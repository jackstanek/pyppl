import random
from typing import Any, Collection


class ParamVector(dict):
    """A vector of program parameters"""

    def __init__(self, *args, **kwargs):
        """Initializes the ParamVector.

        Ensures all initial values are converted to floats.

        Args:
            *args: Variable length argument list to pass to dict constructor.
            **kwargs: Arbitrary keyword arguments to pass to dict constructor.
        """
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            # Convert all initial values to float
            super().__setitem__(k, float(v))

    @classmethod
    def zero(cls, params: Collection[str]) -> "ParamVector":
        """Initialize a zero vector with the given parameter names.

        Args:
            params: collection of names of variables

        Returns:
            parameter vector with the given names, with all values set to zero
        """
        return cls((k, 0) for k in params)

    @classmethod
    def random(cls, params: Collection[str]) -> "ParamVector":
        """Initialize a random vector with the given parameter names.

        Args:
            params: collection of names of variables

        Returns:
            parameter vector with the given names, with all values set random
            values in [0, 1]
        """
        return cls((k, random.random()) for k in params)

    def squared_l2_norm(self) -> float:
        """Get the squared L2 norm of this vector

        Returns:
            float: The squared L2 norm of the vector.
        """
        return sum(self[k] * self[k] for k in self.keys())

    def _check_keys_match(self, other: "ParamVector") -> None:
        """Check that the keys match between this vector and the other vector.

        Args:
            other (ParamVector): Another vector to compare keys with.

        Raises:
            ValueError: If keys do not match.
        """
        this_keys = set(self.keys())
        other_keys = set(other.keys())
        if this_keys != other_keys:
            raise ValueError(
                f"keys in vectors do not match (this: {this_keys}, other: {other_keys})"
            )

    def __add__(self, other: "ParamVector") -> "ParamVector":
        """Adds two ParamVector instances.

        Args:
            other (ParamVector): The other ParamVector to add.

        Returns:
            ParamVector: A new ParamVector instance representing the sum.

        Raises:
            ValueError: If keys in vectors do not match.
        """
        self._check_keys_match(other)
        return ParamVector((k, self[k] + other[k]) for k in self.keys())

    def __radd__(self, other: Any) -> "ParamVector":
        """Handles reverse addition (e.g., 0 + ParamVector).

        This is crucial for compatibility with the sum() function when the start value is 0.

        Args:
            other (Any): The other operand, typically an int, float, or ParamVector.

        Returns:
            ParamVector: A new ParamVector instance representing the sum if `other` is 0 or a ParamVector.
            NotImplemented: If the operation is not supported with the given `other` type.
        """
        if isinstance(other, (int, float)):
            if other == 0:
                # If adding 0 from the left, return a new ParamVector with the same content
                return ParamVector(self)
            else:
                # If 'other' is a non-zero int/float, and it's on the left,
                # this operation does not make sense for a ParamVector.
                return NotImplemented
        # If 'other' is a ParamVector, delegate to __add__
        if isinstance(other, ParamVector):
            return self.__add__(other)
        return NotImplemented  # For any other unsupported type

    def __iadd__(self, other: "ParamVector") -> "ParamVector":
        """Performs in-place addition (+=) of two ParamVector instances.

        Modifies the current instance.

        Args:
            other (ParamVector): The other ParamVector to add.

        Returns:
            ParamVector: The modified current instance (self).
        """
        self._check_keys_match(other)
        for k in self.keys():
            # Use regular addition for the values, and __setitem__ will handle float conversion
            self[k] = self[k] + other[k]
        return self

    def __sub__(self, other: "ParamVector") -> "ParamVector":
        """Subtracts one ParamVector from another.

        Args:
            other (ParamVector): The other ParamVector to subtract.

        Returns:
            ParamVector: A new ParamVector instance representing the difference.

        Raises:
            ValueError: If keys in vectors do not match.
        """
        self._check_keys_match(other)
        return ParamVector((k, self[k] - other[k]) for k in self.keys())

    def __isub__(self, other: "ParamVector") -> "ParamVector":
        """Performs in-place subtraction (-=) of two ParamVector instances.

        Modifies the current instance.

        Args:
            other (ParamVector): The other ParamVector to subtract.

        Returns:
            ParamVector: The modified current instance (self).
        """
        self._check_keys_match(other)
        for k in self.keys():
            self[k] = self[k] - other[k]
        return self

    def __mul__(self, x: int | float) -> "ParamVector":
        """Multiplies all values in the ParamVector by a scalar x.

        Args:
            x (int | float): The scalar value to multiply by.

        Returns:
            ParamVector: A new ParamVector instance with scaled values.
        """
        return ParamVector((k, self[k] * x) for k in self.keys())

    def __rmul__(self, x: int | float) -> "ParamVector":
        """Handles reverse multiplication (e.g., scalar * ParamVector).

        Args:
            x (int | float): The scalar value to multiply by.

        Returns:
            ParamVector: A new ParamVector instance with scaled values.
        """
        return self.__mul__(x)

    def __imul__(self, x: int | float) -> "ParamVector":
        """Performs in-place multiplication (*=) by a scalar x.

        Modifies the current instance.

        Args:
            x (int | float): The scalar value to multiply by.

        Returns:
            ParamVector: The modified current instance (self).
        """
        for k in self.keys():
            self[k] = self[k] * x
        return self

    def __truediv__(self, x: int | float) -> "ParamVector":
        """Divides all values in the ParamVector by a scalar x.

        Args:
            x (int | float): The scalar value to divide by.

        Returns:
            ParamVector: A new ParamVector instance with divided values.

        Raises:
            ZeroDivisionError: If x is 0.
        """
        if x == 0:
            raise ZeroDivisionError("division by zero")
        one_over_x = 1 / x
        return ParamVector((k, self[k] * one_over_x) for k in self.keys())

    def __itruediv__(self, x: int | float) -> "ParamVector":
        """Performs in-place division (/=) by a scalar x.

        Modifies the current instance.

        Args:
            x (int | float): The scalar value to divide by.

        Returns:
            ParamVector: The modified current instance (self).

        Raises:
            ZeroDivisionError: If x is 0.
        """
        if x == 0:
            raise ZeroDivisionError("division by zero")
        for k in self.keys():
            self[k] = self[k] / x
        return self

    def __neg__(self) -> "ParamVector":
        """Negates all values in the ParamVector.

        Returns:
            ParamVector: A new ParamVector instance with negated values.
        """
        return ParamVector((k, -self[k]) for k in self.keys())

    def __setitem__(self, key: Any, value: Any) -> None:
        """Sets the value for an existing key.

        Args:
            key (Any): The key of the item to set.
            value (Any): The new value for the item. Will be converted to float.

        Raises:
            ValueError: If the key does not exist in the vector.
        """
        if key not in self:
            raise ValueError(f"no such key {key} in vector (keys: {list(self.keys())})")
        # Ensure the value is converted to float before storing
        super().__setitem__(key, float(value))
