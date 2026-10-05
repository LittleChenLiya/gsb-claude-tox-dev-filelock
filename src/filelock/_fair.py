"""
In-process fair wrapper around a cross-process file lock.

FairFileLock serializes the threads of one process on numbered tickets
before they touch the underlying BaseFileLock. The ticket holder then
acquires the real cross-process lock, so the OS-level mutual exclusion is
never bypassed. This only makes acquisition order fair within the process;
cross-process ordering stays whatever the backend provides.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections import deque
from typing import TYPE_CHECKING

from ._error import Timeout

if TYPE_CHECKING:
    from collections.abc import Callable

    from ._api import BaseFileLock

try:
    from typing import Self
except ImportError:  # pragma: <3.11 cover
    from typing_extensions import Self

__all__ = ["FairFileLock"]


class _FairAcquireProxy:
    """Handle for one acquire; releases exactly that depth on exit."""

    def __init__(self, wrapper: FairFileLock, acquired_by: int) -> None:
        self._wrapper = wrapper
        self._acquired_by = acquired_by
        self._released = False

    def __enter__(self) -> FairFileLock:
        return self._wrapper

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.release()

    def release(self) -> None:
        self._wrapper._release_proxy(self)  # ruff:ignore[private-member-access]  # proxy forwards to its wrapper


class FairFileLock:
    """
    In-process FIFO fair wrapper around a BaseFileLock.

    Threads that contend on the same instance take numbered tickets and are
    served in ticket order. The served thread then acquires the underlying
    cross-process file lock, so cross-process mutual exclusion is preserved
    and never bypassed. Reentrant acquires by the thread that already holds
    the wrapper only bump a counter and take no new ticket.

    The whole acquisition shares one timeout budget: the deadline is computed
    once when acquire is entered and covers both the wait for a ticket and
    the wait for the underlying lock. Giving up while waiting (timeout or
    cancellation) removes the ticket; a failed underlying acquisition also
    advances the queue head so the next ticket can proceed.

    Fairness scope: FIFO ordering is guaranteed only among threads of the
    same process using the same wrapper instance. Across processes the
    underlying file lock's own ordering applies and is not changed by this
    wrapper.

    :param lock: the cross-process file lock to wrap. It is acquired and
        released through its public API; the wrapper never reaches past it.
    """

    def __init__(self, lock: BaseFileLock) -> None:
        self._lock = lock
        self._cond = threading.Condition(threading.Lock())
        self._next_ticket = 0
        self._queue: deque[int] = deque()
        self._holder: int | None = None
        self._count = 0

    @property
    def lock_file(self) -> str:
        """The path of the underlying file lock."""
        return self._lock.lock_file

    @property
    def is_locked(self) -> bool:
        """Whether the wrapper currently holds the lock (any depth)."""
        return self._count > 0

    @property
    def timeout(self) -> float:
        """Default timeout forwarded to the underlying lock."""
        return self._lock.timeout

    @property
    def blocking(self) -> bool:
        """Default blocking flag forwarded to the underlying lock."""
        return self._lock.blocking

    @property
    def poll_interval(self) -> float:
        """Default poll interval forwarded to the underlying lock."""
        return self._lock.poll_interval

    def __enter__(self) -> Self:
        self.acquire()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.release()

    def acquire(
        self,
        timeout: float | None = None,
        poll_interval: float | None = None,
        *,
        blocking: bool | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> _FairAcquireProxy:
        """
        Acquire the lock in FIFO ticket order.

        The timeout budget is computed once and shared between the ticket
        wait and the underlying lock acquisition. A reentrant acquire by the
        holding thread takes no ticket and only increments the depth.

        :returns: a context handle that releases exactly this acquisition's
            depth on exit.

        :raises Timeout: if the ticket wait or the underlying acquisition
            exceeds the shared timeout budget, or if cancel_check returns
            True.
        """
        if timeout is None:
            timeout = self._lock.timeout
        if blocking is None:
            blocking = self._lock.blocking
        if not blocking:
            timeout = 0
        deadline = time.perf_counter() + timeout if timeout >= 0 else None

        me = threading.get_ident()
        with self._cond:
            if self._holder == me:
                # Reentrant: no new ticket, just depth.
                self._count += 1
                return _FairAcquireProxy(self, me)

            ticket = self._next_ticket
            self._next_ticket += 1
            self._queue.append(ticket)
            try:
                self._wait_for_turn(ticket, deadline, cancel_check)
            except BaseException:
                # Timed out or canceled while queued: drop the ticket.
                self._remove_ticket(ticket)
                raise

            # Served and free: acquire the underlying lock outside the cond
            # so the queue can keep forming behind us without blocking waiters.
        self._acquire_underlying_or_advance(ticket, deadline, poll_interval, cancel_check, blocking=blocking)
        with self._cond:
            self._remove_ticket(ticket)
            self._holder = me
            self._count = 1
            self._cond.notify_all()
        return _FairAcquireProxy(self, me)

    def _wait_for_turn(
        self,
        ticket: int,
        deadline: float | None,
        cancel_check: Callable[[], bool] | None,
    ) -> None:
        while self._queue[0] != ticket or self._holder is not None:
            remaining = self._remaining(deadline)
            if remaining is not None and remaining <= 0:
                raise Timeout(self.lock_file)
            if cancel_check is not None and cancel_check():
                raise Timeout(self.lock_file)
            self._cond.wait(remaining)

    def _acquire_underlying_or_advance(
        self,
        ticket: int,
        deadline: float | None,
        poll_interval: float | None,
        cancel_check: Callable[[], bool] | None,
        *,
        blocking: bool,
    ) -> None:
        try:
            remaining = self._remaining(deadline)
            self._lock.acquire(
                timeout=remaining if remaining is not None else -1,
                poll_interval=poll_interval,
                blocking=blocking,
                cancel_check=cancel_check,
            )
        except BaseException:
            # Underlying acquisition failed (timeout/cancel): advance the head
            # so the next ticket gets its turn instead of waiting behind us.
            with self._cond:
                self._remove_ticket(ticket)
                self._cond.notify_all()
            raise

    def release(self) -> None:
        """Release one depth if the current thread is the holder."""
        me = threading.get_ident()
        with self._cond:
            if self._holder != me or self._count <= 0:
                return
            self._count -= 1
            if self._count == 0:
                self._holder = None
                should_release = True
            else:
                should_release = False
        if should_release:
            # Release the underlying lock before waking waiters so they find it free.
            self._lock.release()
            with self._cond:
                self._cond.notify_all()

    def _release_proxy(self, proxy: _FairAcquireProxy) -> None:
        if proxy._released:  # ruff:ignore[private-member-access]  # idempotent handle
            return
        proxy._released = True  # ruff:ignore[private-member-access]
        with self._cond:
            self._count -= 1
            self._count = max(self._count, 0)
            if self._count == 0:
                self._holder = None
                should_release = True
            else:
                should_release = False
        if should_release:
            # Release the underlying lock before waking waiters so they find it free.
            self._lock.release()
            with self._cond:
                self._cond.notify_all()

    def _remove_ticket(self, ticket: int) -> None:
        with contextlib.suppress(ValueError):
            self._queue.remove(ticket)

    @staticmethod
    def _remaining(deadline: float | None) -> float | None:
        if deadline is None:
            return None
        return deadline - time.perf_counter()
