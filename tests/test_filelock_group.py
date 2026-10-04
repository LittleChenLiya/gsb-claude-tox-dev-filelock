from __future__ import annotations

import os
import sys
import threading
import time
from multiprocessing import get_context
from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import pytest

from filelock import FileLock, FileLockGroup, SoftFileLock, Timeout
from tests.process_helpers import cleanup_processes

if TYPE_CHECKING:
    from multiprocessing.process import BaseProcess
    from multiprocessing.queues import Queue
    from multiprocessing.synchronize import Barrier

if sys.version_info >= (3, 11):  # pragma: >=3.11 cover
    from builtins import ExceptionGroup  # pragma: >=3.11 cover
else:  # pragma: <3.11 cover
    from exceptiongroup import ExceptionGroup


class _RecordingLock(SoftFileLock):
    """A soft lock that records the basename of every physical acquire and release, in order."""

    acquired: ClassVar[list[str]] = []
    released: ClassVar[list[str]] = []

    def _acquire(self) -> None:
        super()._acquire()
        if self.is_locked:  # contended polls retry _acquire; record only the attempt that won
            type(self).acquired.append(Path(self.lock_file).name)

    def _release(self) -> None:
        type(self).released.append(Path(self.lock_file).name)
        super()._release()


class _FlakyReleaseLock(SoftFileLock):
    """A soft lock whose release fails while :attr:`fail_release` is set."""

    fail_release: ClassVar[bool] = False

    def _release(self) -> None:
        if type(self).fail_release:
            msg = "simulated release failure"
            raise OSError(msg)
        super()._release()


@pytest.fixture(autouse=True)
def _reset_recording() -> None:
    _RecordingLock.acquired = []
    _RecordingLock.released = []
    _FlakyReleaseLock.fail_release = False


def test_group_dedupes_equivalent_path_spellings(tmp_path: Path) -> None:
    target = tmp_path / "a.lock"
    spellings = [
        target,
        str(target),
        tmp_path / "." / "a.lock",
        tmp_path / "subdir" / ".." / "a.lock",
    ]
    group = FileLockGroup(spellings)
    assert len(group.locks) == 1
    with group:
        # one physical acquisition for all spellings, not one per spelling
        assert group.locks[0].lock_counter == 1
    assert not group.is_locked


def test_group_dedupes_case_variants_per_platform_rules(tmp_path: Path) -> None:
    group = FileLockGroup([tmp_path / "a.lock", tmp_path / "A.LOCK"])
    case_folded = os.path.normcase("a.lock") == os.path.normcase("A.LOCK")
    assert len(group.locks) == (1 if case_folded else 2)


def test_group_keeps_instance_entries_and_dedupes_paths_against_them(tmp_path: Path) -> None:
    instance = SoftFileLock(tmp_path / "a.lock")
    group = FileLockGroup([instance, str(tmp_path / "a.lock"), tmp_path / "b.lock"], lock_type=SoftFileLock)
    assert len(group.locks) == 2
    assert group.locks[0] is instance


def test_group_orders_members_canonically_regardless_of_input_order(tmp_path: Path) -> None:
    group = FileLockGroup([tmp_path / "c.lock", tmp_path / "a.lock", tmp_path / "b.lock"])
    assert [Path(path).name for path in group.lock_files] == ["a.lock", "b.lock", "c.lock"]
    same = FileLockGroup([tmp_path / "b.lock", tmp_path / "c.lock", tmp_path / "a.lock"])
    assert same.lock_files == group.lock_files


def test_group_acquires_in_order_and_releases_in_reverse(tmp_path: Path) -> None:
    group = FileLockGroup([tmp_path / "b.lock", tmp_path / "a.lock", tmp_path / "c.lock"], lock_type=_RecordingLock)
    with group:
        assert _RecordingLock.acquired == ["a.lock", "b.lock", "c.lock"]
        assert group.is_locked
    assert _RecordingLock.released == ["c.lock", "b.lock", "a.lock"]
    assert not group.is_locked


def test_group_rolls_back_in_reverse_order_when_a_member_fails(tmp_path: Path) -> None:
    blocker = SoftFileLock(tmp_path / "c.lock")
    blocker.acquire()
    try:
        group = FileLockGroup(
            [tmp_path / "a.lock", tmp_path / "b.lock", tmp_path / "c.lock"],
            timeout=0.2,
            lock_type=_RecordingLock,
        )
        with pytest.raises(Timeout):
            group.acquire()
        assert _RecordingLock.acquired == ["a.lock", "b.lock"]
        # the rollback unwinds only what this call took, last acquired first
        assert _RecordingLock.released == ["b.lock", "a.lock"]
        assert not group.is_locked
        # the blocker never lost its own hold to the rollback
        assert blocker.is_locked
        with pytest.raises(Timeout):
            SoftFileLock(tmp_path / "c.lock").acquire(timeout=0)
    finally:
        blocker.release()


def test_group_timeout_frees_acquired_members_for_a_third_party(tmp_path: Path) -> None:
    holder_b = FileLock(tmp_path / "b.lock")
    holder_b.acquire()
    try:
        group = FileLockGroup([tmp_path / "a.lock", tmp_path / "b.lock"], timeout=0.3)
        with pytest.raises(Timeout):
            group.acquire()
        assert not group.is_locked
        # the rollback released a.lock, so an unrelated contender takes it at once
        with FileLock(tmp_path / "a.lock").acquire(timeout=0):
            pass
        # b.lock still belongs to its original holder
        assert holder_b.is_locked
        with pytest.raises(Timeout):
            FileLock(tmp_path / "b.lock").acquire(timeout=0)
    finally:
        holder_b.release()


def test_group_shares_one_timeout_budget_across_members(tmp_path: Path) -> None:
    # not thread-local, so the timer thread may end this hold
    holder_a = FileLock(tmp_path / "a.lock", thread_local=False)
    holder_b = FileLock(tmp_path / "b.lock")
    holder_a.acquire()
    holder_b.acquire()
    # a.lock becomes available a third of the way into the budget; b.lock never does
    timer = threading.Timer(0.3, holder_a.release)
    try:
        timer.start()
        start = time.perf_counter()
        with pytest.raises(Timeout):
            FileLockGroup([tmp_path / "a.lock", tmp_path / "b.lock"], timeout=0.6).acquire()
        elapsed = time.perf_counter() - start
        # a shared budget spends ~0.6s in total; a per-member budget would spend ~0.3s on a plus ~0.6s on b
        assert 0.45 < elapsed < 0.85
    finally:
        timer.cancel()
        holder_b.release()
        if holder_a.is_locked:
            holder_a.release()


def test_group_preserves_reentrant_counts_on_success(tmp_path: Path) -> None:
    lock_a = FileLock(tmp_path / "a.lock")
    lock_a.acquire()
    lock_a.acquire()
    try:
        with FileLockGroup([lock_a, tmp_path / "b.lock"]):
            assert lock_a.lock_counter == 3
        # only the count the group added was returned; the outer hold survives
        assert lock_a.lock_counter == 2
        assert lock_a.is_locked
        # b.lock was fully released
        with FileLock(tmp_path / "b.lock").acquire(timeout=0):
            pass
    finally:
        lock_a.release(force=True)


def test_group_preserves_reentrant_counts_on_rollback(tmp_path: Path) -> None:
    lock_a = FileLock(tmp_path / "a.lock")
    lock_a.acquire()
    lock_a.acquire()
    blocker = FileLock(tmp_path / "c.lock")
    blocker.acquire()
    try:
        group = FileLockGroup([lock_a, tmp_path / "b.lock", tmp_path / "c.lock"], timeout=0.2)
        with pytest.raises(Timeout):
            group.acquire()
        assert lock_a.lock_counter == 2
        assert lock_a.is_locked
        # b.lock was rolled back all the way
        with FileLock(tmp_path / "b.lock").acquire(timeout=0):
            pass
    finally:
        lock_a.release(force=True)
        blocker.release()


def test_group_release_failure_keeps_member_held_for_retry(tmp_path: Path) -> None:
    lock = _FlakyReleaseLock(tmp_path / "a.lock")
    group = FileLockGroup([lock])
    group.acquire()
    _FlakyReleaseLock.fail_release = True
    with pytest.raises(OSError, match="simulated release failure"):
        group.release()
    # the failed member stays registered as held, so a later release retries it
    assert group.is_locked
    _FlakyReleaseLock.fail_release = False
    group.release()
    assert not group.is_locked
    assert not lock.is_locked


def test_group_release_groups_multiple_failures(tmp_path: Path) -> None:
    group = FileLockGroup(
        [_FlakyReleaseLock(tmp_path / "a.lock"), _FlakyReleaseLock(tmp_path / "b.lock")],
    )
    group.acquire()
    _FlakyReleaseLock.fail_release = True
    try:
        with pytest.raises(ExceptionGroup, match="lock group release failed"):
            group.release()
        assert group.is_locked
    finally:
        _FlakyReleaseLock.fail_release = False
    group.release()
    assert not group.is_locked


def test_group_rollback_failure_groups_errors(tmp_path: Path) -> None:
    blocker = SoftFileLock(tmp_path / "b.lock")
    blocker.acquire()
    flaky = _FlakyReleaseLock(tmp_path / "a.lock")
    _FlakyReleaseLock.fail_release = True
    try:
        group = FileLockGroup([flaky, tmp_path / "b.lock"], timeout=0.2, lock_type=SoftFileLock)
        with pytest.raises(ExceptionGroup, match="rollback could not release every lock"):
            group.acquire()
        # the member whose rollback release failed is still held by this flow
        assert flaky.is_locked
    finally:
        _FlakyReleaseLock.fail_release = False
        blocker.release()
    flaky.release()


def test_group_reacquire_while_held_raises(tmp_path: Path) -> None:
    group = FileLockGroup([tmp_path / "a.lock"])
    group.acquire()
    try:
        with pytest.raises(RuntimeError, match="already acquired"):
            group.acquire()
    finally:
        group.release()
    assert not group.is_locked


def test_group_empty_is_a_no_op() -> None:
    group = FileLockGroup([])
    assert group.locks == ()
    with group.acquire():
        assert group.is_locked
    assert not group.is_locked


def _group_worker(barrier: Barrier, first: Path, second: Path, hold: float, events: Queue) -> None:
    barrier.wait(timeout=10)
    with FileLockGroup([first, second], timeout=15):
        events.put(("enter", os.getpid(), time.monotonic()))
        time.sleep(hold)
        events.put(("exit", os.getpid(), time.monotonic()))


def test_group_two_processes_serialize_across_input_orders(tmp_path: Path) -> None:
    """Two native processes submit [a, b] and [b, a]; started together, they serialize instead of deadlocking."""
    a, b = tmp_path / "a.lock", tmp_path / "b.lock"
    context = get_context("spawn")
    barrier = context.Barrier(2)
    events = context.Queue()
    forward: BaseProcess = context.Process(target=_group_worker, args=(barrier, a, b, 0.3, events))
    backward: BaseProcess = context.Process(target=_group_worker, args=(barrier, b, a, 0.3, events))
    with cleanup_processes([forward, backward]):
        forward.start()
        backward.start()
        collected = sorted((events.get(timeout=15) for _ in range(4)), key=itemgetter(2))
        forward.join(timeout=15)
        backward.join(timeout=15)
        assert forward.exitcode == 0
        assert backward.exitcode == 0
    # both processes ran their critical section to completion, and the sections never overlapped
    assert [kind for kind, _, _ in collected] == ["enter", "exit", "enter", "exit"]
