"""An opt-in in-process fair (FIFO ticket) wrapper around a file lock."""

from __future__ import annotations

import itertools
import time
from collections import deque
from threading import Condition, get_ident
from typing import TYPE_CHECKING, Final, NoReturn

from ._error import Timeout

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType

    from ._api import BaseFileLock

if TYPE_CHECKING:
    import sys

    if sys.version_info >= (3, 11):  # pragma: no cover (py311+)
        from typing import Self
    else:  # pragma: no cover (<py311)
        from typing_extensions import Self

#: Upper bound of one condition wait, so a ``cancel_check`` is polled even when no state change wakes the waiter.
_WAIT_SLICE: Final[float] = 0.05


class FairFileLock:
    """
    Serialize same-process threads FIFO in front of a wrapped file lock.

    Every non-reentrant :meth:`acquire` draws a ticket and waits for its turn in ticket order before it drives the
    wrapped lock, so threads of this process are served first-come-first-served instead of racing the OS primitive.
    The wrapped lock is still acquired for real: cross-process exclusion keeps coming from it and is never bypassed.

    Fairness scope: the queue is per ``FairFileLock`` instance and orders only threads of the current process that
    acquire through this same wrapper. Two wrappers around the same lock file keep independent queues, and other
    processes (or threads using the wrapped lock directly) are unaffected by the ticket order - they are serialized
    solely by the underlying lock, as before.

    A single acquire call spends one timeout budget across both phases: waiting for its ticket and acquiring the
    wrapped lock. A waiter whose budget expires, or whose ``cancel_check`` fires, drops its ticket and lets the queue
    advance; the same happens when the wrapped acquire fails. Reentry by the current holder takes no ticket and only
    bumps the hold count; each :meth:`release` undoes exactly one acquisition, and the wrapped lock is released only
    when the count reaches zero.
    """

    def __init__(self, lock: BaseFileLock) -> None:
        """
        Wrap ``lock`` with an in-process FIFO queue.

        :param lock: the file lock that keeps providing the real cross-process exclusion

        """
        self._lock = lock
        self._condition = Condition()
        self._tickets: deque[int] = deque()
        self._ticket_numbers = itertools.count()
        self._owner_ident: int | None = None
        self._owner_count = 0

    @property
    def inner(self) -> BaseFileLock:
        """The wrapped file lock."""
        return self._lock

    @property
    def lock_file(self) -> str:
        """Path to the lock file of the wrapped lock."""
        return self._lock.lock_file

    @property
    def is_locked(self) -> bool:
        """True when the wrapped lock is currently held."""
        return self._lock.is_locked

    @property
    def waiting(self) -> int:
        """The number of tickets currently queued (the holder itself is not queued)."""
        with self._condition:
            return len(self._tickets)

    def acquire(
        self,
        timeout: float | None = None,
        poll_interval: float | None = None,
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> FairAcquireReturnProxy:
        """
        Queue for a ticket, then acquire the wrapped lock, all within one timeout budget.

        :param timeout: maximum wait across the queue wait and the wrapped acquire, ``None`` means use the wrapped
            lock's default :attr:`~filelock.BaseFileLock.timeout`; a negative value waits without a limit
        :param poll_interval: forwarded to the wrapped lock's acquire
        :param cancel_check: a callable returning ``True`` when this acquisition should stop waiting; checked while
            queued and forwarded to the wrapped acquire, raising :class:`~filelock.Timeout` when it fires

        :returns: a context object that releases one hold count when the context is exited

        :raises Timeout: if the budget expires or ``cancel_check`` fires before the wrapped lock is held

        """
        if timeout is None:
            timeout = self._lock.timeout
        deadline = None if timeout < 0 else time.perf_counter() + timeout
        ident = get_ident()
        with self._condition:
            if self._owner_ident == ident:
                # Reentry draws no ticket and shares no budget: the holder already won its turn.
                self._owner_count += 1
                return FairAcquireReturnProxy(lock=self)
            ticket = next(self._ticket_numbers)
            self._tickets.append(ticket)
            try:
                while self._tickets[0] != ticket or self._owner_ident is not None:
                    if cancel_check is not None and cancel_check():
                        _raise_timeout(self.lock_file)
                    remaining = _remaining(deadline, self.lock_file)
                    self._condition.wait(_WAIT_SLICE if remaining is None else min(remaining, _WAIT_SLICE))
            except BaseException:
                # A canceled or timed-out waiter frees its ticket so the queue behind it keeps moving.
                self._tickets.remove(ticket)
                self._condition.notify_all()
                raise
        # Only the head ticket reaches here and only while no thread holds the wrapper, so at most one thread at a
        # time drives the wrapped lock's counter and descriptor; the wrapped lock alone arbitrates other processes.
        try:
            self._lock.acquire(
                timeout=-1 if deadline is None else max(deadline - time.perf_counter(), 0.0),
                poll_interval=poll_interval,
                cancel_check=cancel_check,
            )
        except BaseException:
            # A failed wrapped acquire must not strand the queue: drop the head ticket and wake the next waiter.
            with self._condition:
                self._tickets.remove(ticket)
                self._condition.notify_all()
            raise
        with self._condition:
            self._tickets.popleft()
            self._owner_ident = ident
            self._owner_count = 1
        return FairAcquireReturnProxy(lock=self)

    def release(self) -> None:
        """
        Release one hold count of the current thread; the wrapped lock is released when the count reaches zero.

        :raises RuntimeError: if the current thread does not hold this fair lock

        """
        ident = get_ident()
        with self._condition:
            if self._owner_ident != ident:
                msg = "cannot release a fair lock not held by this thread"
                raise RuntimeError(msg)
            self._owner_count -= 1
            if self._owner_count > 0:
                return  # only this acquisition's count is released; the lock stays held
            # Release the wrapped lock before opening the gate: the next ticket must not observe it still held.
            self._lock.release()
            self._owner_ident = None
            self._condition.notify_all()

    def __enter__(self) -> Self:
        """Acquire the fair lock."""
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release one hold count."""
        self.release()


def _raise_timeout(lock_file: str) -> NoReturn:
    raise Timeout(lock_file)


def _remaining(deadline: float | None, lock_file: str) -> float | None:
    """Budget left until ``deadline``; ``None`` when unbounded, :class:`Timeout` when the budget is spent."""
    if deadline is None:
        return None
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise Timeout(lock_file)
    return remaining


# acquire() returns this wrapper instead of self, mirroring AcquireReturnProxy, so entering the with-statement does
# not acquire a second time; exiting releases exactly the one hold count this acquire added.
class FairAcquireReturnProxy:
    """A context-aware object that releases one hold count of the fair lock when the context exits."""

    def __init__(self, lock: FairFileLock) -> None:
        self.lock = lock

    def __enter__(self) -> FairFileLock:
        return self.lock

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.lock.release()


__all__ = [
    "FairAcquireReturnProxy",
    "FairFileLock",
]
