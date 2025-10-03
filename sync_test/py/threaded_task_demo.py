#!/usr/bin/env python3
"""Minimal runnable example for multi-threaded tasks."""
from __future__ import annotations

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Iterable, Optional


@dataclass
class Task:
    """Represents a unit of work executed in its own thread."""

    name: str
    func: Callable[[], object]

    def run(self) -> object:
        # Attach the thread name so we can see which worker ran this task.
        current = threading.current_thread().name
        print(f"[{self.name}] starting on {current}")
        result = self.func()
        print(f"[{self.name}] finished on {current}")
        return result


def run_tasks(tasks: Iterable[Task], max_workers: Optional[int] = None) -> None:
    """Execute tasks concurrently using a shared pool of worker threads."""

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(task.run): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"[{task.name}] error: {exc}")
            else:
                print(f"[{task.name}] result: {result}")


def make_demo_task(index: int) -> Task:
    """Create a demo task that sleeps for a random short duration."""

    def work() -> str:
        duration = random.uniform(0.5, 1.5)
        time.sleep(duration)
        return f"slept for {duration:.2f}s"

    return Task(name=f"task-{index}", func=work)


if __name__ == "__main__":
    demo_tasks = [make_demo_task(i) for i in range(4)]
    run_tasks(demo_tasks, max_workers=2)
