import abc
import contextlib
from functools import cached_property
import random
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class Environment:
    """Naming environment for expression evaluation"""

    params: Dict[str, float]
    scopes: List[Dict[str, Any]] = field(default_factory=list)

    def __init__(
        self,
        params: Optional[Dict[str, float]] = None,
        initial_vals: Optional[Dict[str, Any]] = None,
    ):
        """Initializes an environment"""
        if params is not None:
            self.params = params
        if initial_vals is None:
            initial_vals = {}
        self.scopes = [initial_vals]

    def add_scope(self):
        """Add a scope to the stack"""
        self.scopes.append({})

    def remove_scope(self):
        """Remove a scope from the stack"""
        self.scopes.pop()

    def add_binding(self, name: str, val: Any):
        """Add a binding to the local scope
        Args:
            name: variable name to add to scope
            val: value to bind name to
        """
        local_scope = self.scopes[-1]
        if name in local_scope:
            raise ValueError(f"name {name} already bound in local scope")
        local_scope[name] = val

    def get_binding(self, name: str):
        """Look up a binding.

        Args:
            name: name to look up binding for
        """
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        raise ValueError(f"name {name} not bound")

    def get_param(self, name: str) -> float:
        """Look up a parameter.

        Args:
            name: name of the parameter

        Returns:
            value of the parameter
        """
        return self.params[name]

    @contextlib.contextmanager
    def temp_scope(self):
        try:
            self.add_scope()
            yield self
        finally:
            self.remove_scope()

    @contextlib.contextmanager
    def temp_binding(self, name: str, val):
        try:
            self.add_scope()
            self.add_binding(name, val)
            yield self
        finally:
            self.remove_scope()


# --- Base AST Node Classes ---


@dataclass(frozen=True)
class ASTNode(abc.ABC):
    """
    Abstract base class for all Abstract Syntax Tree nodes.
    Provides a common interface for AST elements.
    """

    @abc.abstractmethod
    def infer(self, env: Environment, val: "PureNode") -> float:
        """Infer the probability of a value for this program"""


# --- Pure (p) Classes ---


@dataclass(frozen=True)
class PureNode(ASTNode):
    """
    Abstract base class for expression (p) nodes.
    """

    def eval(self, env: Environment) -> "PureNode":
        """Evaluate this pure expression"""
        return self

    def __bool__(self) -> bool:
        raise ValueError(f"truthiness check on non-boolean value {self}")

    def infer(self, env: Environment, val: "PureNode") -> float:
        return float(self.eval(env) == val.eval(env))


def var(name: str) -> "PureNode":
    """Construct a variable node"""
    return VariableNode(name)


def boolean(val: bool) -> "PureNode":
    """Construct a true or false node"""
    if val:
        return TrueNode()
    return FalseNode()


@dataclass(frozen=True)
class VariableNode(PureNode):
    """
    Represents a variable 'x' in a expression.
    Grammar: x
    """

    name: str

    def __post_init__(self):
        """
        Initializes a VariableNode.

        Args:
            name: The name of the variable (e.g., 'a', 'b').
        """
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Variable name must be a non-empty string.")

    def eval(self, env: Environment) -> PureNode:
        return env.get_binding(self.name)


@dataclass(frozen=True)
class TrueNode(PureNode):
    """
    Represents the boolean literal 'tt' (true) in a expression.
    Grammar: tt
    """

    def __bool__(self) -> bool:
        return True


@dataclass(frozen=True)
class FalseNode(PureNode):
    """
    Represents the boolean literal 'ff' (false) in a expression.
    Grammar: ff
    """

    def __bool__(self) -> bool:
        return False


@dataclass(frozen=True)
class IfElseNode(PureNode):
    """
    Represents an 'if p then p else p' conditional expression.
    Grammar: if p then p else p
    """

    condition: PureNode
    true_branch: PureNode
    false_branch: PureNode

    def __post_init__(self):
        """
        Initializes an IfElseNode.

        Args:
            condition: The expression to evaluate (an instance of PureNode).
            true_branch: The expression to execute if condition is true (an instance of PureNode).
            false_branch: The expression to execute if condition is false (an instance of PureNode).
        """
        if not isinstance(self.condition, PureNode):
            raise TypeError("Condition must be an instance of PureNode.")
        if not isinstance(self.true_branch, PureNode):
            raise TypeError("True branch must be an instance of PureNode.")
        if not isinstance(self.false_branch, PureNode):
            raise TypeError("False branch must be an instance of PureNode.")

    def eval(self, env: Environment) -> PureNode:
        cond_val = self.condition.eval(env)
        if bool(cond_val):  # Use bool() explicitly as PureNode has a custom __bool__
            return self.true_branch.eval(env)
        return self.false_branch.eval(env)


@dataclass(frozen=True)
class ConsNode(PureNode):
    """
    Represents a 'cons p p' expression.
    Grammar: cons p p
    """

    head: PureNode
    tail: PureNode

    def __post_init__(self):
        """
        Initializes a ConsNode.

        Args:
            head: The first expression (an instance of PureNode).
            tail: The second expression (an instance of PureNode).
        """
        # Type validation moved to __post_init__
        if not isinstance(self.head, PureNode):
            raise TypeError("Head must be an instance of PureNode.")
        if not isinstance(self.tail, PureNode):
            raise TypeError("Tail must be an instance of PureNode.")

    def eval(self, env: Environment) -> PureNode:
        return ConsNode(self.head.eval(env), self.tail.eval(env))


@dataclass(frozen=True)
class NilNode(PureNode):
    """
    Represents a 'nil' value.
    Grammar: nil
    """


# --- Expression (e) Classes ---


@dataclass(frozen=True)
class ExpressionNode(ASTNode):
    """
    Abstract base class for expression (e) nodes.
    """

    @abc.abstractmethod
    def sample(self, env: Environment) -> PureNode:
        """Sample a value from the program distribution"""

    @abc.abstractmethod
    def possible_vals(self, env: Environment) -> set[PureNode]:
        """Get possible values for this expression in a given context"""

    @cached_property
    def params(self) -> set[str]:
        """Get symbolic parameters of this expression and its subexpressions"""
        return set()

    def gradient(self, env: Environment, val: PureNode) -> dict[str, float]:
        """Compute the gradient of the denotation for a particular value"""
        return {p: self.deriv(env, p, val) for p in self.params}

    def deriv(self, env: Environment, param: str, val: PureNode) -> float:
        """Compute the derivative of the denotation for some parameter

        Args:
            env: naming environment to use
            param: name of the parameter to differentiate
            val: value at which to take the derivative

        Returns:
            value of the derivative at that point
        """
        return 0.0

    def sample_toplevel(self, k: int = 1) -> List[PureNode]:
        """Sample at the top-level with an empty environment

        Args:
            k: number of samples to evaluate

        Return:
            Resulting value from sampling
        """
        return [self.sample(Environment()) for _ in range(k)]


@dataclass(frozen=True)
class ReturnNode(ExpressionNode):
    """
    Represents a 'return p' expression.
    Grammar: return p
    """

    value: PureNode

    def __post_init__(self):
        """
        Initializes a ReturnNode.

        Args:
            value: The pure value to be returned (an instance of PureNode).
        """
        if not isinstance(self.value, PureNode):
            raise TypeError("Return value must be an instance of PureNode.")

    def possible_vals(self, env: Environment) -> set[PureNode]:
        return {self.value.eval(env)}

    def sample(self, env: Environment) -> PureNode:
        return self.value.eval(env)

    def infer(self, env: Environment, val: PureNode) -> float:
        return self.value.infer(env, val)


@dataclass(frozen=True)
class FlipNode(ExpressionNode):
    """
    Represents a 'flip theta' expression.
    Grammar: flip theta
    """

    theta: float | str

    def __post_init__(self):
        """
        Initializes a FlipNode.

        Args:
            theta: The probability value (a float between 0.0 and 1.0).
        """
        if isinstance(self.theta, float) and not (0.0 <= self.theta <= 1.0):
            raise ValueError("Theta must be between 0.0 and 1.0 (inclusive).")

    def get_theta(self, env: Environment) -> float:
        """Get the value of the parameter."""
        if isinstance(self.theta, str):
            return env.get_param(self.theta)
        return self.theta

    def sample(self, env: Environment) -> PureNode:
        return boolean(random.random() < self.get_theta(env))

    def possible_vals(self, env: Environment) -> set[PureNode]:
        return {TrueNode(), FalseNode()}

    def infer(self, env: Environment, val: PureNode) -> float:
        val = val.eval(env)
        if isinstance(val, TrueNode):
            return self.get_theta(env)
        if isinstance(val, FalseNode):
            return 1 - self.get_theta(env)
        return 0.0

    def deriv(self, env: Environment, param: str, val: PureNode) -> float:
        if isinstance(self.theta, float) or self.theta != param:
            return 0.0

        val = val.eval(env)
        if isinstance(val, TrueNode):
            grad = 1.0
        elif isinstance(val, FalseNode):
            grad = -1.0
        else:
            grad = 0.0
        return grad

    @cached_property
    def params(self) -> set[str]:
        if isinstance(self.theta, str):
            return {self.theta}
        return set()


@dataclass(frozen=True)
class SequenceNode(ExpressionNode):
    """
    Represents a 'x <- e; e' sequential expression.
    Grammar: x <- e; e
    """

    variable_name: str
    assignment_expr: ExpressionNode
    next_expr: ExpressionNode

    def __post_init__(self):
        """
        Initializes a SequenceNode.

        Args:
            variable_name: The name of the variable 'x' to which the result of assignment_expr is bound.
            assignment_expr: The expression 'e' whose result is assigned to 'x' (an instance of ExpressionNode).
            next_expr: The subsequent expression 'e' to execute (an instance of ExpressionNode).
        """
        if not isinstance(self.variable_name, str) or not self.variable_name:
            raise ValueError("Variable name must be a non-empty string.")
        if not isinstance(self.assignment_expr, ExpressionNode):
            raise TypeError(
                "Assignment expression must be an instance of ExpressionNode."
            )
        if not isinstance(self.next_expr, ExpressionNode):
            raise TypeError("Next expression must be an instance of ExpressionNode.")

    def sample(self, env: Environment) -> PureNode:
        bind_val = self.assignment_expr.sample(env)
        env.add_binding(self.variable_name, bind_val)
        return self.next_expr.sample(env)

    def possible_vals(self, env: Environment) -> set[PureNode]:
        poss = set()
        for val in self.assignment_expr.possible_vals(env):
            with env.temp_binding(self.variable_name, val):
                poss |= self.next_expr.possible_vals(env)
        return poss

    def infer(self, env: Environment, val: PureNode) -> float:
        # The denotation of a sequence is the sum of the products of the
        # probabilities of each possible intermediate value resulting in the
        # given value being produced:
        # sum_{v in val} [[assign]](env, v) * [[next]](env[x |-> v], val)
        prob = 0.0
        for poss_val in self.assignment_expr.possible_vals(env):
            with env.temp_binding(self.variable_name, poss_val):
                bound_val_prob = self.assignment_expr.infer(env, poss_val)
                prob += bound_val_prob * self.next_expr.infer(env, val)
        return prob

    def deriv(self, env: Environment, param: str, val: PureNode) -> float:
        # The derivative of a sequence is computed with the product rule using
        # the formula given in infer().

        # Sum derivatives of each possible value
        deriv = 0.0
        for poss_val in self.assignment_expr.possible_vals(env):
            with env.temp_binding(self.variable_name, poss_val):
                del_e2 = self.next_expr.deriv(env, param, val)
                e2 = self.next_expr.infer(env, val)
            del_e1 = self.assignment_expr.deriv(env, param, poss_val)
            e1 = self.assignment_expr.infer(env, poss_val)
            deriv += del_e1 * e2 + e1 * del_e2  # Product rule
        return deriv

    @cached_property
    def params(self) -> set[str]:
        return self.assignment_expr.params | self.next_expr.params
