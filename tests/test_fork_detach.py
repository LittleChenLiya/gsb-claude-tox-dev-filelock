"""
Explicit fork-ownership protection: a lock object that detects a process identity change must drop the
inherited hold without unlocking or unlinking the parent process's lock.

Strategy
--------
Two layers, gated by platform capability:

* **State fixtures** (run everywhere, including Windows, which has no ``fork``): a fake backend records every
  call that would unlock or delete the lock, and ``os.getpid`` is patched to simulate the identity change a
  fork child observes. These tests verify filelock's own behavior only -- the inherited reentrant counter is
  cleared, the inherited descriptor copies are closed, and zero unlock/unlink calls reach the backend. They
  deliberately claim *nothing* about kernel lock state, because no real fork happened; where a real file is
  involved the same process stands in for both parent and child, so a closed descriptor really is closed here.
* **Native verification** (only where ``os.fork`` and ``os.register_at_fork`` exist): a real fork child
  releases an inherited lock object whose at-fork reset was bypassed, and the parent then proves its kernel
  lock survived by probing with a contending child. On platforms without fork this test skips and the fixtures
  above are the only evidence -- they are not presented as kernel verification.

Run just this module with::

    python -m pytest tests/test_fork_detach.py -q
"""

from __future__ import annotations

import gc
import os
from errno import EBADF
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Final, NoReturn

import pytest

from filelock import AsyncSoftFileLock, BaseFileLock, FileLock, SoftFileLock
from filelock._api import _FORK_OBJECTS, _OWNED_DESCRIPTORS
from tests.capability_marks import NEEDS_FORK
from tests.fork_helpers import exit_child, fork_process
from tests.test_fork_ownership import _probe_lock

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

_FORK_WARNING: Final[pytest.MarkDecorator] = pytest.mark.filterwarnings(
    "ignore:.*multi-threaded, use of fork.*:DeprecationWarning"
)


class _CountingBackend(BaseFileLock):
    """A fake backend that records the two calls a fork child must never make against the parent's lock."""

    calls: ClassVar[dict[str, int]] = {"release": 0, "unlink": 0}

    def _acquire(self) -> None:
        # A real descriptor, so the detach's close of the inherited copy is observable via fstat.
        fd = os.open(self.lock_file, os.O_RDWR | os.O_CREAT, 0o666)
        self._mark_descriptor_owned(fd)

    def _release(self) -> None:
        # Unlock and unlink are what _release means across the real backends; count both, then clean up.
        type(self).calls["release"] += 1
        type(self).calls["unlink"] += 1
        fd = self._context.lock_file_fd
        self._mark_descriptor_released()
        if fd is not None:
            os.close(fd)


def _assert_fd_closed(fd: int) -> None:
    """The inherited descriptor copy is closed; compare errnos, not the localized error message."""
    raised_errno = None
    try:
        os.fstat(fd)
    except OSError as exception:
        raised_errno = exception.errno
    assert raised_errno == EBADF


@pytest.fixture
def backend_calls() -> dict[str, int]:
    _CountingBackend.calls = {"release": 0, "unlink": 0}
    return _CountingBackend.calls


def test_identity_change_clears_reentrancy_without_unlock_or_unlink(
    tmp_path: Path, mocker: MockerFixture, backend_calls: dict[str, int]
) -> None:
    acquired_fds: list[int] = []
    lock = _CountingBackend(
        str(tmp_path / "held.lock"), thread_local=False, is_singleton=False, on_acquired=acquired_fds.append
    )
    lock.acquire()
    lock.acquire()
    assert lock.lock_counter == 2

    mocker.patch("os.getpid", return_value=os.getpid() + 1_000_001)  # the fork child's identity
    lock.release(force=True)

    # The parent's reentrant counter is not this process's hold state: it is cleared, not decremented.
    assert (lock.lock_counter, lock.is_locked) == (0, False)
    # The detach never unlocked or unlinked the parent's lock through the backend.
    assert backend_calls == {"release": 0, "unlink": 0}
    # The inherited descriptor copy was closed.
    _assert_fd_closed(acquired_fds[0])


def test_identity_change_collect_closes_inherited_copy(
    tmp_path: Path, mocker: MockerFixture, backend_calls: dict[str, int]
) -> None:
    acquired_fds: list[int] = []
    lock = _CountingBackend(
        str(tmp_path / "held.lock"), thread_local=False, is_singleton=False, on_acquired=acquired_fds.append
    )
    lock.acquire()

    mocker.patch("os.getpid", return_value=os.getpid() + 1_000_001)  # the fork child's identity
    del lock
    gc.collect()

    assert backend_calls == {"release": 0, "unlink": 0}
    _assert_fd_closed(acquired_fds[0])


def test_detached_lock_follows_new_object_rules(
    tmp_path: Path, mocker: MockerFixture, backend_calls: dict[str, int]
) -> None:
    path = str(tmp_path / "held.lock")
    lock = _CountingBackend(path, thread_local=False, is_singleton=False)
    lock.acquire()

    mocker.patch("os.getpid", return_value=os.getpid() + 1_000_001)  # the fork child's identity
    lock.release()

    # The inherited object keeps refusing to acquire; the child takes a new instance through the usual rules.
    with pytest.raises(RuntimeError, match="inherited across fork"):
        lock.acquire(timeout=0)
    fresh = _CountingBackend(path, thread_local=False, is_singleton=False)
    fresh.acquire(timeout=0)
    assert fresh.is_locked
    fresh.release()
    # Only the fresh instance's own release reached the backend.
    assert backend_calls == {"release": 1, "unlink": 1}


@pytest.mark.parametrize("lock_type", [pytest.param(FileLock, id="native"), pytest.param(SoftFileLock, id="soft")])
def test_real_backend_detach_makes_no_unlock_or_unlink_call(
    tmp_path: Path, mocker: MockerFixture, lock_type: type[BaseFileLock]
) -> None:
    path = tmp_path / "held.lock"
    lock = lock_type(str(path), thread_local=False, is_singleton=False)
    lock.acquire()
    assert path.exists()
    release_spy = mocker.spy(lock, "_release")

    mocker.patch("os.getpid", return_value=os.getpid() + 1_000_001)  # the fork child's identity
    lock.release(force=True)

    assert (lock.lock_counter, lock.is_locked) == (0, False)
    # One process stands in for both parent and child, so the closed descriptor really is closed here; what this
    # fixture proves is that filelock issued no unlock/unlink of its own. Kernel-level preservation of the
    # parent's hold is the fork test below, on platforms that have fork.
    assert release_spy.call_count == 0
    assert path.exists()  # the unlink half of _release (Windows native, soft marker) never ran


@pytest.mark.asyncio
async def test_async_lock_detaches_inherited_hold(tmp_path: Path, mocker: MockerFixture) -> None:
    path = tmp_path / "held.lock"
    lock = AsyncSoftFileLock(str(path), thread_local=False, is_singleton=False)
    await lock.acquire()
    await lock.acquire()
    assert lock.lock_counter == 2
    release_spy = mocker.spy(lock, "_release")

    mocker.patch("os.getpid", return_value=os.getpid() + 1_000_001)  # the fork child's identity
    await lock.release(force=True)

    assert (lock.lock_counter, lock.is_locked) == (0, False)
    assert release_spy.call_count == 0
    assert path.exists()


@pytest.mark.asyncio
async def test_async_lock_collect_detaches_inherited_hold(tmp_path: Path, mocker: MockerFixture) -> None:
    path = tmp_path / "held.lock"
    lock = AsyncSoftFileLock(str(path), thread_local=False, is_singleton=False)
    await lock.acquire()

    mocker.patch("os.getpid", return_value=os.getpid() + 1_000_001)  # the fork child's identity
    del lock
    gc.collect()

    assert path.exists()  # the child's finalizer closed its inherited copy without unlinking the marker


@NEEDS_FORK  # pragma: needs fork
@_FORK_WARNING
@pytest.mark.parametrize("lock_type", [pytest.param(FileLock, id="native"), pytest.param(SoftFileLock, id="soft")])
def test_fork_child_detach_preserves_parent_lock(tmp_path: Path, lock_type: type[BaseFileLock]) -> None:
    path = str(tmp_path / "parent.lock")
    lock = lock_type(path, thread_local=False, is_singleton=False)
    lock.acquire()
    lock.acquire()
    assert lock.lock_counter == 2

    # Bypass the at-fork reset for this one object so the child exercises the lazy detach: unpin the object and
    # its descriptor from the registries the fork hooks would have reset.
    _FORK_OBJECTS.pop(id(lock), None)
    if (token := lock._context.lock_file_fd_token) is not None:
        _OWNED_DESCRIPTORS.pop(token, None)
    inherited_fd = lock._context.lock_file_fd
    assert inherited_fd is not None  # the lock is held, so its descriptor is set

    def detach_child() -> NoReturn:
        lock.release(force=True)
        ok = (lock.lock_counter, lock.is_locked) == (0, False)
        try:
            os.fstat(inherited_fd)
        except OSError as exception:
            ok = ok and exception.errno == EBADF
        else:
            ok = False
        exit_child(0 if ok else 1)

    child_pid = fork_process(detach_child)
    _, status = os.waitpid(child_pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0

    # The parent's hold survived the child's detach: a contender still cannot acquire, and a soft lock's marker
    # was never unlinked. The parent's own release then frees the path for the same contender.
    assert not _probe_lock(lock_type, path)
    if lock_type is SoftFileLock:
        assert Path(path).exists()
    lock.release(force=True)
    assert _probe_lock(lock_type, path)
