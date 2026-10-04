from __future__ import annotations

import multiprocessing as mp
import sys
import threading
import time
from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from filelock import BaseFileLock, FileLock, FileLockGroup, SoftFileLock, Timeout

if TYPE_CHECKING:
    from collections.abc import Callable
    from multiprocessing.queues import Queue
    from multiprocessing.synchronize import Barrier, Event

# --------------------------------------------------------------------------- #
# Normalization, deduplication and deterministic order
# --------------------------------------------------------------------------- #


def test_group_normalizes_dedupes_and_orders(tmp_path: Path) -> None:
    a = tmp_path / "a.lock"
    b = tmp_path / "b.lock"
    group = FileLockGroup([str(b), str(a), str(a), str(b)])
    assert [Path(lock.lock_file) for lock in group.locks] == [a, b]


def test_equivalent_spellings_dedupe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    group = FileLockGroup(["a.lock", str(tmp_path / "a.lock"), "./a.lock"])
    assert len(group.locks) == 1
    assert Path(group.locks[0].lock_file).resolve() == (tmp_path / "a.lock").resolve()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows file names are case-insensitive")
def test_windows_case_insensitive_dedup(tmp_path: Path) -> None:
    upper = tmp_path / "A.LOCK"
    group = FileLockGroup([str(upper), str(upper).lower()])
    assert len(group.locks) == 1


def test_group_accepts_generator_and_mixed_members(tmp_path: Path) -> None:
    a = tmp_path / "a.lock"
    b = tmp_path / "b.lock"
    instance = FileLock(b)
    group = FileLockGroup(str(p) for p in (a, b))
    assert len(group.locks) == 2
    group_mixed = FileLockGroup([str(a), instance])
    assert group_mixed.locks[1] is instance


def test_group_acquires_in_deterministic_order(tmp_path: Path) -> None:
    a = tmp_path / "a.lock"
    b = tmp_path / "b.lock"
    c = tmp_path / "c.lock"
    order: list[str] = []

    def record(path: Path) -> Callable[[int], None]:
        def callback(_fd: int) -> None:
            order.append(str(path))

        return callback

    locks = [FileLock(path, on_acquired=record(path)) for path in (c, a, b)]
    group = FileLockGroup(locks)
    with group:
        pass
    assert order == [str(a), str(b), str(c)]


# --------------------------------------------------------------------------- #
# Context management and reentrancy
# --------------------------------------------------------------------------- #


def test_context_manager_acquires_and_releases(tmp_path: Path) -> None:
    a = tmp_path / "a.lock"
    b = tmp_path / "b.lock"
    group = FileLockGroup([str(a), str(b)])
    with group:
        assert group.is_locked
        assert all(lock.is_locked for lock in group.locks)
    assert not group.is_locked
    assert all(not lock.is_locked for lock in group.locks)
    # A third party can take over once the group exited.
    assert FileLock(a).acquire(timeout=0.1) is not None
    FileLock(a).release()


def test_acquire_proxy_does_not_reenter(tmp_path: Path) -> None:
    a = tmp_path / "a.lock"
    group = FileLockGroup([str(a)])
    with group.acquire() as held:
        assert held is group
        assert group.is_locked
    assert not group.is_locked


def test_context_manager_releases_on_body_exception(tmp_path: Path) -> None:
    a = tmp_path / "a.lock"
    b = tmp_path / "b.lock"
    group = FileLockGroup([str(a), str(b)])
    msg = "boom"
    with pytest.raises(RuntimeError, match=msg), group:
        raise RuntimeError(msg)
    assert FileLock(a).acquire(timeout=0.1) is not None
    FileLock(a).release()
    assert FileLock(b).acquire(timeout=0.1) is not None
    FileLock(b).release()


def test_nested_group_acquire_tracks_rounds(tmp_path: Path) -> None:
    a = tmp_path / "a.lock"
    b = tmp_path / "b.lock"
    group = FileLockGroup([str(a), str(b)])
    with group:
        with group:
            assert all(lock.lock_counter == 2 for lock in group.locks)
        assert all(lock.lock_counter == 1 for lock in group.locks)
    assert all(lock.lock_counter == 0 for lock in group.locks)
    assert not group.is_locked


def test_reentrant_hold_keeps_external_count(tmp_path: Path) -> None:
    a = tmp_path / "a.lock"
    b = tmp_path / "b.lock"
    external = FileLock(str(a))
    external.acquire()
    assert external.lock_counter == 1
    with FileLockGroup([external, str(b)]):
        assert external.lock_counter == 2
    assert external.lock_counter == 1
    external.release()
    assert external.lock_counter == 0


def test_release_without_acquire_is_noop(tmp_path: Path) -> None:
    group = FileLockGroup([str(tmp_path / "a.lock")])
    group.release()
    group.release(force=True)


def test_force_releases_all_rounds(tmp_path: Path) -> None:
    a = tmp_path / "a.lock"
    group = FileLockGroup([str(a)])
    group.acquire()
    group.acquire()
    assert all(lock.lock_counter == 2 for lock in group.locks)
    group.release(force=True)
    assert all(lock.lock_counter == 0 for lock in group.locks)
    assert not group.is_locked


# --------------------------------------------------------------------------- #
# Acquisition failure: reverse rollback and the shared budget
# --------------------------------------------------------------------------- #


def _hold_in_thread(lock: FileLock, ready: threading.Event, release: threading.Event) -> threading.Thread:
    def hold() -> None:
        with lock:
            ready.set()
            release.wait()

    thread = threading.Thread(target=hold, daemon=True)
    thread.start()
    ready.wait()
    return thread


def test_group_rollback_frees_locks_for_third_party(tmp_path: Path) -> None:
    a = tmp_path / "a.lock"
    b = tmp_path / "b.lock"
    holder_ready = threading.Event()
    holder_release = threading.Event()
    thread = _hold_in_thread(FileLock(str(b)), holder_ready, holder_release)
    try:
        with pytest.raises(Timeout):
            FileLockGroup([str(a), str(b)], timeout=0.4).acquire()
        # b kept the group out, so the group must have released a again: a third party acquires it.
        third = FileLock(str(a))
        assert third.acquire(timeout=0.1) is not None
        third.release()
    finally:
        holder_release.set()
        thread.join()


def test_group_releases_in_reverse_order_on_failure(tmp_path: Path) -> None:
    a = tmp_path / "a.lock"
    b = tmp_path / "b.lock"
    c = tmp_path / "c.lock"
    order: list[str] = []
    locks = [FileLock(str(path)) for path in (a, b, c)]
    for lock in locks:
        original = lock.release

        def spy(
            *args: object,
            _original: Callable[..., None] = original,
            _lock: BaseFileLock = lock,
            **kwargs: object,
        ) -> None:
            order.append(_lock.lock_file)
            _original(*args, **kwargs)

        lock.release = spy  # type: ignore[method-assign]
    holder_ready = threading.Event()
    holder_release = threading.Event()
    thread = _hold_in_thread(FileLock(str(c)), holder_ready, holder_release)
    try:
        with pytest.raises(Timeout):
            FileLockGroup(locks, timeout=0.3).acquire()
        assert order == [str(b), str(a)]
    finally:
        holder_release.set()
        thread.join()


def test_total_timeout_budget_is_shared(tmp_path: Path) -> None:
    a = tmp_path / "a.lock"
    b = tmp_path / "b.lock"
    holder_ready = threading.Event()
    holder_release = threading.Event()
    thread = _hold_in_thread(FileLock(str(b)), holder_ready, holder_release)
    budget = 0.5
    try:
        start = time.monotonic()
        with pytest.raises(Timeout):
            FileLockGroup([str(a), str(b)], timeout=budget).acquire()
        elapsed = time.monotonic() - start
        # The budget is shared: a is taken instantly, b waits only for what remains of the one budget.
        assert elapsed >= budget * 0.8
        assert elapsed < budget * 2
    finally:
        holder_release.set()
        thread.join()


def test_failure_keeps_external_soft_marker(tmp_path: Path) -> None:
    a = tmp_path / "a.lock"
    b = tmp_path / "b.lock"
    external = SoftFileLock(str(a))
    external.acquire()
    marker = Path(str(a))
    assert marker.exists()
    holder_ready = threading.Event()
    holder_release = threading.Event()
    thread = _hold_in_thread(FileLock(str(b)), holder_ready, holder_release)
    try:
        with pytest.raises(Timeout):
            FileLockGroup([external, str(b)], timeout=0.3).acquire()
        # The group only reentered external's hold; it must not have unlinked someone else's marker.
        assert marker.exists()
        assert external.lock_counter == 1
    finally:
        holder_release.set()
        thread.join()
    external.release()
    assert not marker.exists()


def test_repr_lists_members(tmp_path: Path) -> None:
    a = tmp_path / "a.lock"
    group = FileLockGroup([str(a)], timeout=3)
    rendered = repr(group)
    assert "FileLockGroup" in rendered
    assert a.name in rendered
    assert "3.0" in rendered


# --------------------------------------------------------------------------- #
# Two callers, reversed input order, started through a barrier: no deadlock
# --------------------------------------------------------------------------- #


def _group_worker(
    lock_dir: str,
    names: list[str],
    barrier: Barrier,
    hold: float,
    queue: Queue,
) -> None:
    try:
        group = FileLockGroup([str(Path(lock_dir) / name) for name in names], timeout=30)
        barrier.wait()
        with group:
            queue.put(("acquired", names[0], time.monotonic()))
            time.sleep(hold)
            queue.put(("released", names[0], time.monotonic()))
    except BaseException as exc:  # pragma: no cover - diagnostic path on failure
        queue.put(("error", names[0], f"{type(exc).__name__}: {exc}"))


def test_subprocess_groups_do_not_deadlock(tmp_path: Path) -> None:
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(2)
    queue = ctx.Queue()
    hold = 0.5
    first = ctx.Process(target=_group_worker, args=(str(tmp_path), ["a.lock", "b.lock"], barrier, hold, queue))
    second = ctx.Process(target=_group_worker, args=(str(tmp_path), ["b.lock", "a.lock"], barrier, hold, queue))
    first.start()
    second.start()
    first.join(timeout=40)
    second.join(timeout=40)
    assert first.exitcode == 0, f"first worker failed with {first.exitcode}"
    assert second.exitcode == 0, f"second worker failed with {second.exitcode}"
    events: dict[str, dict[str, float]] = {}
    for _ in range(4):
        kind, tag, stamp = queue.get(timeout=5)
        assert kind != "error", f"worker {tag} failed: {stamp}"
        events.setdefault(tag, {})[kind] = stamp
    assert len(events) == 2
    intervals = sorted(events.values(), key=itemgetter("acquired"))
    # Both acquire in the same fixed order, so the second group only starts once the first released.
    assert intervals[1]["acquired"] >= intervals[0]["released"] - 0.1


def _rollback_worker(lock_dir: str, event: Event, queue: Queue) -> None:
    try:
        group = FileLockGroup([str(Path(lock_dir) / "a.lock"), str(Path(lock_dir) / "b.lock")], timeout=1)
        event.wait()
        try:
            group.acquire()
        except Timeout:
            queue.put("timeout")
        else:  # pragma: no cover - diagnostic path on failure
            queue.put("acquired")
    except BaseException as exc:  # pragma: no cover - diagnostic path on failure
        queue.put(f"error: {type(exc).__name__}: {exc}")


def _holder_worker(lock_dir: str, ready: Event, release: Event) -> None:
    with FileLock(str(Path(lock_dir) / "b.lock")):
        ready.set()
        release.wait()


def test_subprocess_rollback_frees_lock_for_third_party(tmp_path: Path) -> None:
    ctx = mp.get_context("spawn")
    holder_ready = ctx.Event()
    holder_release = ctx.Event()
    queue: Queue = ctx.Queue()
    holder = ctx.Process(target=_holder_worker, args=(str(tmp_path), holder_ready, holder_release))
    rollback = ctx.Process(target=_rollback_worker, args=(str(tmp_path), holder_ready, queue))
    holder.start()
    rollback.start()
    holder_ready.wait()
    rollback.join(timeout=20)
    assert rollback.exitcode == 0
    assert queue.get(timeout=5) == "timeout"
    # The group released a on its way out: a third party acquires it even though b is still held.
    third = FileLock(str(tmp_path / "a.lock"))
    assert third.acquire(timeout=1) is not None
    third.release()
    holder_release.set()
    holder.join(timeout=10)
    assert holder.exitcode == 0
