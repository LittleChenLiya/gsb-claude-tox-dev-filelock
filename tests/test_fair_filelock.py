from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from filelock import FairFileLock, FileLock, Timeout

if TYPE_CHECKING:
    from pathlib import Path


def test_context_manager_and_release(tmp_path: Path) -> None:
    lock = FairFileLock(FileLock(tmp_path / "a.lock"))
    assert not lock.is_locked
    with lock:
        assert lock.is_locked
    assert not lock.is_locked


def test_reentrant_acquire_takes_no_ticket(tmp_path: Path) -> None:
    lock = FairFileLock(FileLock(tmp_path / "b.lock"))
    with lock:
        assert len(lock._queue) == 0
        lock.acquire()
        assert len(lock._queue) == 0
        lock.acquire()
        assert len(lock._queue) == 0
        assert lock._count == 3


def test_release_drops_one_depth_only(tmp_path: Path) -> None:
    lock = FairFileLock(FileLock(tmp_path / "c.lock"))
    with lock:
        proxy = lock.acquire()
        assert lock._count == 2
        proxy.release()
        assert lock._count == 1
        assert lock.is_locked
    assert not lock.is_locked


def test_fifo_ticket_order_with_barrier(tmp_path: Path) -> None:
    lock = FairFileLock(FileLock(tmp_path / "d.lock"))
    order: list[str] = []
    timeouts: list[str] = []
    barrier = threading.Barrier(3)

    def first() -> None:
        with lock.acquire(timeout=2.0):
            order.append("first")
            barrier.wait()
            time.sleep(0.3)

    def second() -> None:
        barrier.wait()
        try:
            with lock.acquire(timeout=0.15):
                order.append("second")
        except Timeout:
            timeouts.append("second")

    def third() -> None:
        barrier.wait()
        try:
            with lock.acquire(timeout=2.0):
                order.append("third")
        except Timeout:
            timeouts.append("third")

    threads = [threading.Thread(target=first), threading.Thread(target=second), threading.Thread(target=third)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert order == ["first", "third"]
    assert timeouts == ["second"]


def test_second_timeout_third_advances(tmp_path: Path) -> None:
    lock = FairFileLock(FileLock(tmp_path / "e.lock"))
    acquired: list[str] = []
    failed: list[str] = []
    barrier = threading.Barrier(3)

    def holder() -> None:
        with lock.acquire(timeout=2.0):
            acquired.append("holder")
            barrier.wait()
            time.sleep(0.3)

    def waiter_short() -> None:
        barrier.wait()
        try:
            with lock.acquire(timeout=0.1):
                acquired.append("short")
        except Timeout:
            failed.append("short")

    def waiter_long() -> None:
        barrier.wait()
        try:
            with lock.acquire(timeout=2.0):
                acquired.append("long")
        except Timeout:
            failed.append("long")

    threads = [
        threading.Thread(target=holder),
        threading.Thread(target=waiter_short),
        threading.Thread(target=waiter_long),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert acquired == ["holder", "long"]
    assert failed == ["short"]


def test_exclusive_while_held(tmp_path: Path) -> None:
    lock = FairFileLock(FileLock(tmp_path / "f.lock"))
    with lock:
        result: list[str] = []

        def contender() -> None:
            try:
                with lock.acquire(timeout=0.1):
                    result.append("acquired")
            except Timeout:
                result.append("timeout")

        thread = threading.Thread(target=contender)
        thread.start()
        thread.join()
        assert result == ["timeout"]


def test_cancel_check_removes_ticket(tmp_path: Path) -> None:
    lock = FairFileLock(FileLock(tmp_path / "g.lock"))
    with lock:
        barrier = threading.Barrier(2)
        cancelled: list[str] = []

        def waiter() -> None:
            barrier.wait()
            try:
                lock.acquire(timeout=2.0, cancel_check=lambda: True)
            except Timeout:
                cancelled.append("waiter")

        thread = threading.Thread(target=waiter)
        thread.start()
        barrier.wait()
        thread.join()
        assert cancelled == ["waiter"]
        assert len(lock._queue) == 0


def test_shared_timeout_budget(tmp_path: Path) -> None:
    lock = FairFileLock(FileLock(tmp_path / "h.lock"))
    with lock:
        result: list[str] = []

        def contender() -> None:
            start = time.perf_counter()
            try:
                lock.acquire(timeout=0.15)
            except Timeout:
                result.append(f"timeout:{time.perf_counter() - start:.2f}")

        thread = threading.Thread(target=contender)
        thread.start()
        thread.join()
        assert len(result) == 1
        assert result[0].startswith("timeout:")
        elapsed = float(result[0].split(":")[1])
        assert elapsed < 0.5


def test_underlying_failure_advances_head(tmp_path: Path) -> None:
    lock = FairFileLock(FileLock(tmp_path / "i.lock"))
    order: list[str] = []
    barrier = threading.Barrier(2)

    def blocker() -> None:
        # Hold the underlying lock from another thread via a second wrapper on
        # the same path, so the head waiter's underlying acquire fails.
        other = FairFileLock(FileLock(tmp_path / "i.lock"))
        with other.acquire(timeout=2.0):
            barrier.wait()
            time.sleep(0.3)

    def waiter() -> None:
        barrier.wait()
        try:
            with lock.acquire(timeout=0.1):
                order.append("acquired")
        except Timeout:
            order.append("timeout")

    t1 = threading.Thread(target=blocker)
    t2 = threading.Thread(target=waiter)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert order == ["timeout"]
    assert len(lock._queue) == 0
