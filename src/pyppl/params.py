from typing import Any


class ParamVector(dict):
    """A vector of program parameters"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            super().__setitem__(k, float(v))

    @classmethod
    def zeros_like(cls, other: "ParamVector") -> "ParamVector":
        return cls({k: 0 for k in other})

    def squared_l2_norm(self) -> float:
        """Get the squared L2 norm of this vector"""
        return sum(self[k] * self[k] for k in self.keys())

    def _check_keys_match(self, other: "ParamVector") -> None:
        """Check that the keys match between this vector and the other vector

        Args:
            other: another vector

        Raises:
            ValueError: if keys do not match
        """
        this_keys = set(self.keys())
        other_keys = set(other.keys())
        if this_keys != other_keys:
            raise ValueError(
                f"keys in vectors do not match (this: {this_keys}, other: {other_keys})"
            )

    def __add__(self, other: "ParamVector") -> "ParamVector":
        self._check_keys_match(other)
        return ParamVector((k, self[k] + other[k]) for k in self.keys())

    def __sub__(self, other: "ParamVector") -> "ParamVector":
        self._check_keys_match(other)
        return ParamVector((k, self[k] - other[k]) for k in self.keys())

    def __mul__(self, x: int | float) -> "ParamVector":
        return ParamVector((k, self[k] * x) for k in self.keys())

    def __truediv__(self, x: int | float) -> "ParamVector":
        one_over_x = 1 / x
        return ParamVector((k, self[k] * one_over_x) for k in self.keys())

    def __neg__(self) -> "ParamVector":
        return ParamVector((k, -self[k]) for k in self.keys())

    def __setitem__(self, key: Any, value: Any) -> None:
        if key not in self:
            raise ValueError(f"no such key {key} in vector (keys: {list(self.keys())})")
        return super().__setitem__(key, float(value))
