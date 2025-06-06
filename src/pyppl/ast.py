import abc
import contextlib
import random
from collections.abc import Generator
from dataclasses import dataclass
from functools import cached_property
from typing import Any, List

from pyppl.environment import BaseEnv
from pyppl.params import ParamVector


@dataclass(frozen=True)
class ASTNode(abc.ABC):
    """
    Abstract base class for all Abstract Syntax Tree nodes.
    Provides a common interface for AST elements.
    """


@dataclass
class EvalEnv(BaseEnv):
    """Naming environment for expression evaluation"""

    params: ParamVector
    scope_factory = dict

    @cached_property
    def param_names(self) -> set[str]:
        """Names of defined parameters in the environment"""
        return self.params.param_names

    def get_param(self, name: str) -> float:
        """
        Look up a parameter.

        Args:
            name: name of the parameter

        Returns:
            value of the parameter
        """
        return self.params[name]

    def add_binding(self, name: str, val: Any):
        """
        Add a binding to the local scope.

        Args:
            name: variable name to add to scope
            val: value to bind name to
        """
        local_scope = self.scopes[-1]
        if name in local_scope:
            raise ValueError(f"name {name} already bound in local scope")
        local_scope[name] = val

    def get_binding(self, name: str) -> Any:
        """
        Look up a binding.

        Args:
            name: name to look up binding for

        Returns:
            value bound to the given name
        """
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        raise ValueError(f"name {name} not bound")

    @contextlib.contextmanager
    def local_binding(self, name: str, val):
        """
        Create a binding within a context.

        Args:
            name: the name to bind
            val: the value to bind to the name
        """
        try:
            self.add_scope()
            self.add_binding(name, val)
            yield self
        finally:
            self.remove_scope()


@dataclass(frozen=True)
class Program(ASTNode):
    """
    Base class for programs (definitions followed by an effectful expression)
    """

    defns: dict[str, "ExpressionNode"]
    expr: "EffectfulNode"

    def env(self, params: ParamVector) -> EvalEnv:
        """Create a fresh environment with the given parameters.

        Args:
            params: vector containing the parameters

        Returns:
            new environment with the global bindings and given parameters
        """
        return EvalEnv(params=params, initial_vals=self.defns)

    def infer(self, params: ParamVector, val: "PureNode") -> float:
        """Perform inference over the whole program.

        Args:
            params: fixed parameter values
            val: value to infer

        Return:
            probability assigned to the value by the program
        """
        env = EvalEnv(params=params, initial_vals=self.defns)
        return self.expr.infer(env, val)

    @cached_property
    def params(self) -> set[str]:
        return self.expr.params.union(*(b.params for b in self.defns.values()))

    def sample(self, params: ParamVector, k: int = 1) -> List["PureNode"]:
        """Sample at the top-level with an environment containing the global definitions

        Args:
            k: number of samples to evaluate

        Return:
            Resulting value from sampling

        """
        undefined_params = self.expr.params - params.param_names
        if undefined_params:
            raise UndefinedParamError(
                f"""undefined parameters: {",".join(undefined_params)}"""
            )
        samples = []
        for _ in range(k):
            env = EvalEnv(params=params, initial_vals=self.defns)
            samples.append(self.expr.sample(env))
        return samples

    def gradient(self, params: ParamVector, val: "PureNode") -> ParamVector:
        """Compute the gradient of the denotation for a particular value"""
        env = self.env(params)
        return ParamVector({p: self.expr.deriv(env, p, val) for p in self.params})


@dataclass(frozen=True)
class ExpressionNode(ASTNode):
    """
    Base class for expressions in the language
    """

    @abc.abstractmethod
    def infer(self, env: EvalEnv, val: "PureNode") -> float:
        """Infer the probability of a value for this program"""

    @cached_property
    @abc.abstractmethod
    def params(self) -> set[str]:
        """Set of parameter names in this sub-expression"""

    @property
    def subexpressions(self) -> Generator["ExpressionNode"]:
        """Yield the subexpressions of this expression"""
        yield from ()


# --- Pure (p) Classes ---


@dataclass(frozen=True)
class PureNode(ExpressionNode):
    """
    Abstract base class for expression (p) nodes.
    """

    def eval(self, env: EvalEnv) -> "PureNode":
        """Evaluate this pure expression"""
        return self

    def __bool__(self) -> bool:
        raise ValueError(f"truthiness check on non-boolean value {self}")

    def infer(self, env: EvalEnv, val: "PureNode") -> float:
        return float(self.eval(env) == val.eval(env))

    @cached_property
    def params(self) -> set[str]:
        return set()


def var(name: str) -> "VariableNode":
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

    def eval(self, env: EvalEnv) -> PureNode:
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

    def eval(self, env: EvalEnv) -> PureNode:
        cond_val = self.condition.eval(env)
        if bool(cond_val):  # Use bool() explicitly as PureNode has a custom __bool__
            return self.true_branch.eval(env)
        return self.false_branch.eval(env)

    @property
    def subexpressions(self) -> Generator[ExpressionNode]:
        yield from (self.condition, self.true_branch, self.false_branch)


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

    def eval(self, env: EvalEnv) -> PureNode:
        return ConsNode(self.head.eval(env), self.tail.eval(env))

    @property
    def subexpressions(self) -> Generator[ExpressionNode]:
        yield from (self.head, self.tail)


@dataclass(frozen=True)
class NilNode(PureNode):
    """
    Represents a 'nil' value.
    Grammar: nil
    """


# --- Expression (e) Classes ---


class UndefinedParamError(Exception):
    """Thrown when a parameter is not defined in the execution environment."""


@dataclass(frozen=True)
class EffectfulNode(ExpressionNode):
    """
    Abstract base class for expression (e) nodes.
    """

    @abc.abstractmethod
    def sample(self, env: EvalEnv) -> PureNode:
        """Sample a value from the program distribution"""

    @abc.abstractmethod
    def possible_vals(self, env: EvalEnv) -> set[PureNode]:
        """Get possible values for this expression in a given context"""

    @cached_property
    def params(self) -> set[str]:
        """Get symbolic parameters of this expression and its subexpressions"""
        return set()

    def gradient(self, env: EvalEnv, val: PureNode) -> ParamVector:
        """Compute the gradient of the denotation for a particular value"""
        return ParamVector({p: self.deriv(env, p, val) for p in self.params})

    def deriv(self, env: EvalEnv, param: str, val: PureNode) -> float:
        """Compute the derivative of the denotation for some parameter

        Args:
            env: naming environment to use
            param: name of the parameter to differentiate
            val: value at which to take the derivative

        Returns:
            value of the derivative at that point
        """
        return 0.0


@dataclass(frozen=True)
class ReturnNode(EffectfulNode):
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

    def possible_vals(self, env: EvalEnv) -> set[PureNode]:
        return {self.value.eval(env)}

    def sample(self, env: EvalEnv) -> PureNode:
        return self.value.eval(env)

    def infer(self, env: EvalEnv, val: PureNode) -> float:
        return self.value.infer(env, val)

    @property
    def subexpressions(self) -> Generator[ExpressionNode]:
        yield from (self.value,)


@dataclass(frozen=True)
class FlipNode(EffectfulNode):
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

    def get_theta(self, env: EvalEnv) -> float:
        """Get the value of the parameter."""
        if isinstance(self.theta, str):
            return env.get_param(self.theta)
        return self.theta

    def sample(self, env: EvalEnv) -> PureNode:
        return boolean(random.random() < self.get_theta(env))

    def possible_vals(self, env: EvalEnv) -> set[PureNode]:
        return {TrueNode(), FalseNode()}

    def infer(self, env: EvalEnv, val: PureNode) -> float:
        val = val.eval(env)
        if isinstance(val, TrueNode):
            return self.get_theta(env)
        if isinstance(val, FalseNode):
            return 1 - self.get_theta(env)
        return 0.0

    def deriv(self, env: EvalEnv, param: str, val: PureNode) -> float:
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
class SequenceNode(EffectfulNode):
    """
    Represents a 'x <- e; e' sequential expression.
    Grammar: x <- e; e
    """

    variable_name: str
    assignment_expr: EffectfulNode
    next_expr: EffectfulNode

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
        if not isinstance(self.assignment_expr, EffectfulNode):
            raise TypeError(
                "Assignment expression must be an instance of ExpressionNode."
            )
        if not isinstance(self.next_expr, EffectfulNode):
            raise TypeError("Next expression must be an instance of ExpressionNode.")

    def sample(self, env: EvalEnv) -> PureNode:
        bind_val = self.assignment_expr.sample(env)
        env.add_binding(self.variable_name, bind_val)
        return self.next_expr.sample(env)

    def possible_vals(self, env: EvalEnv) -> set[PureNode]:
        poss = set()
        for val in self.assignment_expr.possible_vals(env):
            with env.local_binding(self.variable_name, val):
                poss |= self.next_expr.possible_vals(env)
        return poss

    def infer(self, env: EvalEnv, val: PureNode) -> float:
        # The denotation of a sequence is the sum of the products of the
        # probabilities of each possible intermediate value resulting in the
        # given value being produced:
        # sum_{v in val} [[assign]](env, v) * [[next]](env[x |-> v], val)
        prob = 0.0
        for poss_val in self.assignment_expr.possible_vals(env):
            with env.local_binding(self.variable_name, poss_val):
                bound_val_prob = self.assignment_expr.infer(env, poss_val)
                prob += bound_val_prob * self.next_expr.infer(env, val)
        return prob

    def deriv(self, env: EvalEnv, param: str, val: PureNode) -> float:
        # The derivative of a sequence is computed with the product rule using
        # the formula given in infer().

        # Sum derivatives of each possible value
        deriv = 0.0
        for poss_val in self.assignment_expr.possible_vals(env):
            with env.local_binding(self.variable_name, poss_val):
                del_e2 = self.next_expr.deriv(env, param, val)
                e2 = self.next_expr.infer(env, val)
            del_e1 = self.assignment_expr.deriv(env, param, poss_val)
            e1 = self.assignment_expr.infer(env, poss_val)
            deriv += del_e1 * e2 + e1 * del_e2  # Product rule
        return deriv

    @cached_property
    def params(self) -> set[str]:
        return self.assignment_expr.params | self.next_expr.params

    @property
    def subexpressions(self) -> Generator[ExpressionNode]:
        yield from (self.assignment_expr, self.next_expr)
