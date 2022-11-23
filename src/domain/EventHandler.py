from dataclasses import dataclass
from typing import TypeVar, Callable

A = TypeVar("A")
B = TypeVar("B")
fAB = Callable[[A], B]


@dataclass(frozen=True)
class EventHandler:
    handler: fAB

    def then(self, that: fAB) -> fAB:
        def inner(it: A) -> B:
            self.handler(it)
            return that(it)

        return inner
