"""Short-thread tests for the in-process fair (FIFO ticket) lock wrapper."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import pytest

from filelock import FairFileLock, FileLock, Timeout

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    """Poll ``predicate`` until it holds, failing the test when it does not within ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            pytest.fail("condition was not met in time")
        time.sleep(0.005)


def test_fair_ticket_order_timeout_and_exclusion(tmp_path: Path) -> None:
    """Three threads queue by ticket; the second times out, the third still advances; holds stay exclusive."""
    fair = FairFileLock(FileLock(str(tmp_path / "fair.lock")))
    barrier = threading.Barrier(3)
    gates = [threading.Event() for _ in range(3)]
    entered: list[int] = []
    timed_out: list[int] = []
    state = threading.Lock()
    active = 0
    max_active = 0

    fair.acquire()
    assert fair.waiting == 0

    def worker(index: int, timeout: float) -> None:
        nonlocal active, max_active
        barrier.wait()  # all three threads leave the start line together
        gates[index].wait()  # but draw their tickets in ticket order
        try:
            with fair.acquire(timeout=timeout):
                with state:
                    active += 1
                    max_active = max(max_active, active)
                entered.append(index)
                time.sleep(0.05)
                with state:
                    active -= 1
        except Timeout:
            timed_out.append(index)

    timeouts = (10.0, 0.3, 10.0)
    threads = [threading.Thread(target=worker, args=(index, timeouts[index])) for index in range(3)]
    for thread in threads:
        thread.start()
    for index, gate in enumerate(gates):
        gate.set()
        wait_until(lambda index=index: fair.waiting == index + 1)  # ticket index is queued before index + 1 is drawn
    # The holder keeps the lock until the short-budget waiter (ticket 1) has expired and dropped its ticket.
    wait_until(lambda: timed_out == [1])
    wait_until(lambda: fair.waiting == 2)
    fair.release()
    for thread in threads:
        thread.join()

    assert entered == [0, 2]  # ticket 2 advances past the timed-out ticket 1
    assert timed_out == [1]
    assert max_active == 1  # the hold stayed exclusive the whole time
    assert fair.waiting == 0
    assert not fair.is_locked


def test_fair_reentry_adds_no_ticket_and_release_is_paired(tmp_path: Path) -> None:
    """Reentry bumps only the hold count; each release undoes one acquisition, not the whole hold."""
    fair = FairFileLock(FileLock(str(tmp_path / "reentrant.lock")))
    fair.acquire()
    fair.acquire()  # reentrant: no new ticket is drawn
    assert fair.waiting == 0

    entered: list[str] = []

    def contender() -> None:
        with fair.acquire(timeout=5.0):
            entered.append("contender")

    thread = threading.Thread(target=contender)
    thread.start()
    wait_until(lambda: fair.waiting == 1)

    fair.release()  # releases only this acquisition's count
    assert fair.is_locked  # the remaining count still holds the lock
    assert entered == []

    fair.release()  # the count reaches zero, so the queued ticket may proceed
    thread.join()
    assert entered == ["contender"]
    assert not fair.is_locked

    with pytest.raises(RuntimeError, match="not held by this thread"):
        fair.release()


def test_fair_inner_acquire_failure_advances_queue(tmp_path: Path) -> None:
    """A head ticket whose wrapped acquire fails frees the queue, so the next ticket gets its turn."""
    competitor = FileLock(str(tmp_path / "busy.lock"))
    competitor.acquire(timeout=5.0)
    fair = FairFileLock(FileLock(competitor.lock_file))
    gates = {"first": threading.Event(), "second": threading.Event()}
    results: dict[str, str] = {}

    def worker(name: str, timeout: float) -> None:
        gates[name].wait()
        try:
            with fair.acquire(timeout=timeout):
                results[name] = "acquired"
        except Timeout:
            results[name] = "timeout"

    first = threading.Thread(target=worker, args=("first", 0.2))
    second = threading.Thread(target=worker, args=("second", 5.0))
    first.start()
    gates["first"].set()
    wait_until(lambda: fair.waiting == 1)
    second.start()
    gates["second"].set()
    wait_until(lambda: fair.waiting == 2)

    # The head ticket's wrapped acquire fails against the competitor; its ticket must not strand the queue.
    wait_until(lambda: results.get("first") == "timeout")
    wait_until(lambda: fair.waiting == 1)
    competitor.release()
    first.join()
    second.join()

    assert results == {"first": "timeout", "second": "acquired"}
    assert not fair.is_locked


def test_fair_shared_budget_covers_queue_wait(tmp_path: Path) -> None:
    """The queue wait and the wrapped acquire draw on one budget: a queued waiter times out on its own budget."""
    fair = FairFileLock(FileLock(str(tmp_path / "budget.lock")))
    fair.acquire()
    elapsed: list[float] = []

    def waiter() -> None:
        start = time.perf_counter()
        with pytest.raises(Timeout):
            fair.acquire(timeout=0.3)
        elapsed.append(time.perf_counter() - start)

    thread = threading.Thread(target=waiter)
    thread.start()
    wait_until(lambda: fair.waiting == 1)
    thread.join()
    fair.release()

    assert len(elapsed) == 1
    assert elapsed[0] == pytest.approx(0.3, abs=0.25)


def test_fair_context_manager_and_proxy(tmp_path: Path) -> None:
    """Both the lock itself and the acquire proxy work as context managers, releasing exactly one count."""
    fair = FairFileLock(FileLock(str(tmp_path / "ctx.lock")))
    with fair:
        assert fair.is_locked
        with fair.acquire():  # reentrant through the proxy: still no ticket
            assert fair.waiting == 0
            assert fair.is_locked
        assert fair.is_locked  # the proxy released only its own count
    assert not fair.is_locked
