"""
Multi-file lock group: acquire several file locks as one unit.

The group normalizes its lock members (collapsing equivalent spellings and deduplicating them under the platform's
case rule and the real parent directory rule), sorts them into one deterministic order, and acquires them with a
shared total timeout budget. Two callers that submit the same paths in a different order therefore never wait on
each other: both acquire in the same order instead of holding what the other needs. If any lock cannot be
acquired, every lock the group took in this attempt is released in reverse order, so a failed acquisition never
keeps a lock away from a third party. A lock that was already held before the group joined it is only decremented
by the count the group added; the group never releases an external hold.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import TYPE_CHECKING, Union

from ._api import BaseFileLock, _canonical, _resolve_poll_interval
from ._soft import SoftFileLock
from ._unix import UnixFileLock, has_fcntl
from ._windows import WindowsFileLock

if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import TracebackType
    from typing import Self

__all__ = ["FileLockGroup", "FileLockGroupAcquireProxy"]

#: A group member is either an existing lock instance or a path, which the group turns into a platform FileLock.
LockLike = Union[str, "os.PathLike[str]", BaseFileLock]

if sys.platform == "win32":  # pragma: win32 cover
    _PlatformFileLock: type[BaseFileLock] = WindowsFileLock
elif has_fcntl:
    _PlatformFileLock = UnixFileLock
else:  # pragma: no cover  # only on platforms without fcntl
    _PlatformFileLock = SoftFileLock


def _member_key(member: LockLike) -> str:
    """
    Return the stable dedup key for a group member.

    Equivalent spellings of one path collapse to one key via the same rule the lock registry uses (the real parent
    directory, with the final component kept literal), and the platform's case rule folds Windows names, where
    ``A`` and ``a`` name the same file.
    """
    path = member.lock_file if isinstance(member, BaseFileLock) else os.fspath(member)
    key = _canonical(path)
    if sys.platform == "win32":
        key = key.casefold()
    return key


class FileLockGroup:
    """
    A group of file locks acquired together in a deterministic order.

    Pass any mix of lock paths and existing :class:`~filelock.BaseFileLock` instances. Paths are turned into
    platform file locks, equivalent spellings and duplicate members collapse to one, and the survivors are
    acquired in a stable order. The total *timeout* is one shared budget for the whole acquisition: each lock
    waits only for whatever remains of it. When a lock cannot be acquired, every lock already taken in this
    attempt is released in reverse order, so the failed group never keeps a lock away from anyone else.

    Already-held locks are reentrant: the group's acquisition only adds one recursion count to a lock it joins,
    and leaving the group removes exactly that count. An external holder's count is never released by the group.

    .. code-block:: python

        from filelock import FileLockGroup

        with FileLockGroup(["a.lock", "b.lock"], timeout=10):
            # holds both a.lock and b.lock, acquired in a fixed order
            ...

    """

    def __init__(self, locks: Iterable[LockLike], timeout: float = -1, poll_interval: float = 0.05) -> None:
        self._timeout = float(timeout)
        self._poll_interval = _resolve_poll_interval(poll_interval)
        members: list[BaseFileLock] = []
        seen: set[str] = set()
        for member in locks:
            lock = member if isinstance(member, BaseFileLock) else _PlatformFileLock(member)
            key = _member_key(lock)
            if key in seen:
                continue
            seen.add(key)
            members.append(lock)
        members.sort(key=_member_key)
        self._locks: tuple[BaseFileLock, ...] = tuple(members)
        self._state_lock = threading.RLock()
        self._rounds: list[list[BaseFileLock]] = []

    @property
    def locks(self) -> tuple[BaseFileLock, ...]:
        """The normalized, deduplicated locks in the order the group acquires them."""
        return self._locks

    @property
    def timeout(self) -> float:
        """The total timeout budget for acquiring the whole group, in seconds."""
        return self._timeout

    @timeout.setter
    def timeout(self, value: float) -> None:
        self._timeout = float(value)

    @property
    def poll_interval(self) -> float:
        """The retry interval used while acquiring the group, in seconds."""
        return self._poll_interval

    @poll_interval.setter
    def poll_interval(self, value: float) -> None:
        self._poll_interval = _resolve_poll_interval(value)

    @property
    def is_locked(self) -> bool:
        """Whether at least one acquisition round is currently held."""
        with self._state_lock:
            return bool(self._rounds)

    def acquire(
        self,
        timeout: float | None = None,
        poll_interval: float | None = None,
    ) -> FileLockGroupAcquireProxy:
        """
        Acquire every lock in the group, in order, sharing one total timeout budget.

        :param timeout: total wait for the whole group, ``None`` uses the group default. A negative value blocks
            until every lock is held; a non-negative value bounds the total wait. Each lock gets only the time
            left in the budget, never a fresh timeout of its own.
        :param poll_interval: retry interval, ``None`` uses the group default.

        :returns: a context object that releases the group when exited, so ``with group.acquire():`` does not
            re-enter the group.

        :raises Timeout: if the group cannot be fully acquired within the budget. Every lock taken in this attempt
            is released in reverse order first.

        """
        if timeout is None:
            timeout = self._timeout
        if poll_interval is None:
            poll_interval = self._poll_interval
        poll_interval = _resolve_poll_interval(poll_interval)
        start = time.perf_counter()
        acquired: list[BaseFileLock] = []
        try:
            for lock in self._locks:
                # One shared budget: each lock waits only for what is left of it, never a fresh timeout. A zero
                # budget still grants one attempt, matching FileLock's timeout=0 semantics.
                remaining = timeout if timeout < 0 else max(0.0, timeout - (time.perf_counter() - start))
                lock.acquire(timeout=remaining, poll_interval=poll_interval)
                acquired.append(lock)
        except BaseException:
            # Reverse rollback: release what this attempt took, one release per successful acquisition. A lock
            # that was already held before the group is only decremented by the count we added; an external hold
            # is never touched.
            for held in reversed(acquired):
                held.release()
            raise
        with self._state_lock:
            self._rounds.append(acquired)
        return FileLockGroupAcquireProxy(self)

    def release(self, force: bool = False) -> None:  # ruff:ignore[boolean-type-hint-positional-argument, boolean-default-value-positional-argument]  # public API: positional bool kept for backwards compatibility
        """
        Release the locks the group acquired.

        Without *force*, releases the latest acquisition round, one :meth:`~filelock.BaseFileLock.release` per
        lock in reverse acquisition order. *force* releases every round the group pushed. Either way a lock is
        only decremented by the count the group added; a lock held by someone else before the group joined it
        keeps that hold. A release failure on one lock does not strand the others: the rest still release in
        reverse order, and the first failure propagates afterwards.
        """
        with self._state_lock:
            if not self._rounds:
                return
            if force:
                rounds = self._rounds
                self._rounds = []
            else:
                rounds = [self._rounds.pop()]
        error: BaseException | None = None
        for round_locks in reversed(rounds):
            for lock in reversed(round_locks):
                try:
                    lock.release()
                except BaseException as exc:  # ruff:ignore[blind-except, try-except-in-loop]  # every round must still be released
                    if error is None:
                        error = exc
        if error is not None:
            raise error

    def __enter__(self) -> Self:
        """Acquire the whole group."""
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the latest acquisition round, whatever the body did."""
        self.release()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({[lock.lock_file for lock in self._locks]!r}, timeout={self._timeout!r})"


class FileLockGroupAcquireProxy:
    """
    Context returned by :meth:`FileLockGroup.acquire`.

    Mirrors :class:`filelock.AcquireReturnProxy`: entering the context returns the group instead of acquiring it a
    second time, so ``with group.acquire():`` releases exactly what the call acquired.
    """

    def __init__(self, group: FileLockGroup) -> None:
        self.group: FileLockGroup = group

    def __enter__(self) -> FileLockGroup:
        return self.group

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.group.release()
