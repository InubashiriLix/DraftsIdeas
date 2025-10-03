#!/usr/bin/env python3
"""Minimal synchronous task model example."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
U = TypeVar("U")


@dataclass
class SyncTask:
    """Represents a synchronous pipeline that pulls from a source and pushes into a sink."""

    name: str
    source: Callable[[], Iterable[T]]
    transform: Callable[[T], U]
    sink: Callable[[U], None]

    def run(self) -> None:
        for item in self.source():
            self.sink(self.transform(item))


def sample_source() -> Iterable[str]:
    return ["alpha", "beta", "gamma"]


def uppercase_transform(item: str) -> str:
    return item.upper()


def print_sink(item: str) -> None:
    print(f"Synced: {item}")


if __name__ == "__main__":
    demo_task = SyncTask(
        name="demo-sync",
        source=sample_source,
        transform=uppercase_transform,
        sink=print_sink,
    )
    demo_task.run()
