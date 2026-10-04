from __future__ import annotations

import contextlib
import logging
import os
import time
from threading import RLock
from typing import TYPE_CHECKING, Final

from ._api import BaseFileLock, _canonical, _raise_cleanup_errors

if TYPE_CHECKING:
    import sys
    from collections.abc import Iterable
    from types import TracebackType

    if sys.version_info >= (3, 11):  # pragma: no cover (py311+)
        from typing import Self
    else:  # pragma: no cover (<py311)
        from typing_extensions import Self

_LOGGER: Final[logging.Logger] = logging.getLogger("filelock")


class FileLockGroup(contextlib.ContextDecorator):
    """
    Acquire several file locks as one unit, in a canonical order every caller shares.

    Two flows that need the same locks deadlock when each takes them in a different order: one holds ``a`` and waits
    for ``b`` while the other holds ``b`` and waits for ``a``. The group removes that interleaving by normalizing the
    given paths once at construction and always acquiring in the resulting canonical order, so the caller's input
    order (or duplicated spellings of one file) cannot change who waits for whom.

    Normalization follows the same canonicalization the lock registry uses: relative, ``./``, and ``..`` spellings
    collapse onto the real parent directory (``abspath`` on Windows, ``realpath`` elsewhere) without following a
    final symlink, and the platform's case rules (``os.path.normcase``) fold case-only variants on case-insensitive
    filesystems. Duplicates keep a single member, so a repeated path is acquired once, not once per spelling.

    One acquisition shares a single timeout budget across all members: the clock starts when
    :meth:`~FileLockGroup.acquire` begins and each member only gets what remains. If any member cannot be acquired,
    the members this call already took are released in reverse acquisition order before the failure propagates. A
    member that was already held reentrantly (its :attr:`~filelock.BaseFileLock.lock_counter` was non-zero) only
    returns the count this acquisition added, so an outer hold survives a failed or finished group acquisition. The
    rollback never force-releases and never deletes a lock file it does not own.

    Entries may be paths or existing :class:`~filelock.BaseFileLock` instances. Pass an instance for a lock the
    current flow already holds reentrantly; a path entry builds a fresh ``lock_type`` instance, which cannot join a
    hold owned by a different instance.

    .. code-block:: python

        # Every caller acquires in the same canonical order, no matter the input order:
        with FileLockGroup(["b.lock", "a.lock"]) as group:
            assert [os.path.basename(p) for p in group.lock_files] == ["a.lock", "b.lock"]

        # A member that cannot be acquired in time rolls back what this call already took, in
        # reverse order; a lock another flow still holds keeps its earlier acquisitions:
        holder = FileLock("b.lock")
        holder.acquire()
        try:
            with FileLockGroup(["a.lock", "b.lock"], timeout=1):
                ...
        except Timeout:
            pass  # a.lock is free again; b.lock still belongs to ``holder``
        holder.release()

    """

    def __init__(
        self,
        lock_files: Iterable[str | os.PathLike[str] | BaseFileLock],
        timeout: float = -1,
        *,
        lock_type: type[BaseFileLock] | None = None,
        poll_interval: float = 0.05,
        blocking: bool = True,
        **lock_options: object,
    ) -> None:
        """
        Create a lock group, normalizing and ordering its members immediately.

        :param lock_files: the lock paths or :class:`~filelock.BaseFileLock` instances to acquire together
        :param timeout: default total budget in seconds for one :meth:`acquire` call, shared by all members.
            A negative value waits indefinitely; ``None`` passed to :meth:`acquire` selects this default.
        :param lock_type: the lock class used for path entries; ``None`` (the default) uses the platform's
            :data:`~filelock.FileLock` alias. Ignored for entries that already are lock instances.
        :param poll_interval: default polling interval forwarded to each member's acquire
        :param blocking: default blocking flag forwarded to each member's acquire
        :param lock_options: extra keyword arguments for the constructor of locks built from path entries

        """
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._blocking = blocking
        self._transition_lock = RLock()
        self._acquired: list[BaseFileLock] = []
        self._holding = False
        if lock_type is None:
            # Deferred: the package __init__ picks the platform backend, and it is still importing this module.
            from . import FileLock  # ruff:ignore[import-outside-top-level]

            lock_type = FileLock
        members: dict[str, BaseFileLock] = {}
        for entry in lock_files:
            instance = entry if isinstance(entry, BaseFileLock) else None
            canonical = _canonical(entry if instance is None else instance.lock_file)
            key = os.path.normcase(canonical)
            if key not in members:
                members[key] = instance if instance is not None else lock_type(canonical, **lock_options)
        self._members: tuple[BaseFileLock, ...] = tuple(members[key] for key in sorted(members))

    @property
    def locks(self) -> tuple[BaseFileLock, ...]:
        """The member locks in canonical acquisition order."""
        return self._members

    @property
    def lock_files(self) -> tuple[str, ...]:
        """The member lock paths in canonical acquisition order."""
        return tuple(lock.lock_file for lock in self._members)

    @property
    def is_locked(self) -> bool:
        """Whether the group currently holds its members."""
        return self._holding and all(lock.is_locked for lock in self._acquired)

    def acquire(
        self,
        timeout: float | None = None,
        poll_interval: float | None = None,
        *,
        blocking: bool | None = None,
    ) -> GroupAcquireReturnProxy:
        """
        Acquire every member in canonical order under one shared timeout budget.

        :param timeout: total budget in seconds for the whole acquisition, ``None`` means use the default
        :param poll_interval: interval between acquisition attempts, ``None`` means use the default
        :param blocking: whether to retry until every member is held or the budget expires, ``None`` means use
            the default. ``False`` makes one attempt per member and raises :class:`~filelock.Timeout` on failure.

        :returns: a context object that releases the group when the context is exited

        :raises RuntimeError: if the group is already acquired
        :raises Timeout: if a member cannot be acquired within the remaining budget; members this call already
            took are released in reverse order before the error propagates

        """
        if timeout is None:
            timeout = self._timeout
        if poll_interval is None:
            poll_interval = self._poll_interval
        if blocking is None:
            blocking = self._blocking
        start_time = time.perf_counter()
        with self._transition_lock:
            if self._holding:
                msg = "lock group is already acquired; release it before acquiring again"
                raise RuntimeError(msg)
            acquired: list[BaseFileLock] = []
            try:
                for lock in self._members:
                    remaining = timeout if timeout < 0 else max(0.0, timeout - (time.perf_counter() - start_time))
                    _LOGGER.debug("Attempting to acquire lock group member %s", lock.lock_file)
                    lock.acquire(timeout=remaining, poll_interval=poll_interval, blocking=blocking)
                    acquired.append(lock)
            except BaseException as acquire_error:
                failures = self._release_all(acquired)
                if failures:
                    _raise_cleanup_errors(
                        "lock group acquire failed and its rollback could not release every lock",
                        acquire_error,
                        *(error for _, error in failures),
                    )
                raise
            self._acquired = acquired
            self._holding = True
        return GroupAcquireReturnProxy(group=self)

    def release(self) -> None:
        """
        Release every held member in reverse acquisition order.

        Each member is released exactly once per group acquisition, so a member with an outer reentrant hold keeps
        the count it had before the group acquired it. A member whose release fails stays registered as held, so a
        later :meth:`release` retries it; the first failure propagates after every member had its attempt.
        """
        with self._transition_lock:
            failures = self._release_all(self._acquired)
            # Failed members stay held; restore their original (acquisition) order for the retry.
            self._acquired = [lock for lock, _ in reversed(failures)]
            self._holding = bool(self._acquired)
        if failures:
            errors = [error for _, error in failures]
            if len(errors) == 1:
                raise errors[0]
            _raise_cleanup_errors("lock group release failed", errors[0], *errors[1:])

    @staticmethod
    def _release_all(locks: list[BaseFileLock]) -> list[tuple[BaseFileLock, Exception]]:
        """Release *locks* in reverse order, returning ``(lock, error)`` pairs for the releases that failed."""
        failures: list[tuple[BaseFileLock, Exception]] = []
        for lock in reversed(locks):
            try:
                lock.release()
            except Exception as error:  # ruff:ignore[blind-except, try-except-in-loop]  # members all get their attempt; failures surface together
                failures.append((lock, error))
        return failures

    def __enter__(self) -> Self:
        """
        Acquire the group.

        :returns: the group object

        """
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release every member in reverse order; a release failure chains onto any active body failure."""
        self.release()


# acquire() returns this wrapper instead of self so entering the with-statement does not acquire the group a second
# time, mirroring why BaseFileLock.acquire returns AcquireReturnProxy (issue #37).
class GroupAcquireReturnProxy:
    """A context-aware object that releases the lock group when the context exits."""

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


__all__ = ["FileLockGroup", "GroupAcquireReturnProxy"]
