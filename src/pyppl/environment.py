import contextlib
from dataclasses import InitVar, dataclass, field
from typing import Any


@dataclass(kw_only=True)
class BaseEnv:
    """Base class for environment"""

    scopes: list = field(default_factory=list, init=False)
    initial_vals: InitVar[Any] = None

    def __post_init__(self, initial_vals):
        if initial_vals is None:
            initial_vals = self.scope_factory()
        self.scopes.append(initial_vals)

    @staticmethod
    def scope_factory():
        """Construct a new scope."""
        raise NotImplementedError()

    def add_scope(self):
        """
        Add a scope to the stack.
        """
        self.scopes.append(self.scope_factory())

    def remove_scope(self):
        """
        Remove a scope from the stack.
        """
        self.scopes.pop()

    @contextlib.contextmanager
    def local_scope(self):
        try:
            self.add_scope()
            yield self
        finally:
            self.remove_scope()


class NameEnv(BaseEnv):
    def __init__(self):
        pass
