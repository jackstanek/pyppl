import abc
import itertools
import random


class Environment:
    """Naming environment for expression evaluation"""

    def __init__(self):
        """Initializes an environment"""
        self.scopes = [{}]

    def add_scope(self):
        """Add a scope to the stack"""
        self.scopes.append({})

    def remove_scope(self):
        """Remove a scope from the stack"""
        self.scopes.pop()

    def add_binding(self, name: str, val):
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


# --- Base AST Node Classes ---


class ASTNode(abc.ABC):
    """
    Abstract base class for all Abstract Syntax Tree nodes.
    Provides a common interface for AST elements.
    """


# --- Pure (p) Classes ---


class PureNode(ASTNode):
    """
    Abstract base class for expression (p) nodes.
    """

    def eval(self, env: Environment) -> "PureNode":
        """Evaluate this pure expression"""
        return self

    def __bool__(self) -> bool:
        raise ValueError(f"truthiness check on non-boolean value {self}")

    def possible_values(self, env: Environment) -> set["PureNode"]:
        """Determine possible values that this node could evaluate to"""
        return set()


def var(name: str) -> "PureNode":
    """Construct a variable node"""
    return VariableNode(name)


def boolean(val: bool) -> "PureNode":
    """Construct a true or false node"""
    if val:
        return TrueNode()
    return FalseNode()


class VariableNode(PureNode):
    """
    Represents a variable 'x' in a expression.
    Grammar: x
    """

    def __init__(self, name: str):
        """
        Initializes a VariableNode.

        Args:
            name: The name of the variable (e.g., 'a', 'b').
        """
        if not isinstance(name, str) or not name:
            raise ValueError("Variable name must be a non-empty string.")
        self.name = name

    def __repr__(self):
        return f"VariableNode(name='{self.name}')"

    def eval(self, env: Environment) -> PureNode:
        return env.get_binding(self.name)

    def possible_values(self, env: Environment):
        return env.get_binding(self.name)


class TrueNode(PureNode):
    """
    Represents the boolean literal 'tt' (true) in a expression.
    Grammar: tt
    """

    def __init__(self):
        # No specific data needed for a constant 'true' value
        pass

    def __repr__(self):
        return "TrueNode()"

    def __bool__(self) -> bool:
        return True

    def possible_values(self, env: Environment) -> set[PureNode]:
        return set([self])


class FalseNode(PureNode):
    """
    Represents the boolean literal 'ff' (false) in a expression.
    Grammar: ff
    """

    def __init__(self):
        # No specific data needed for a constant 'false' value
        pass

    def __repr__(self):
        return "FalseNode()"

    def __bool__(self) -> bool:
        return False

    def possible_values(self, env: Environment) -> set[PureNode]:
        return set([self])


class IfElseNode(PureNode):
    """
    Represents an 'if p then p else p' conditional expression.
    Grammar: if p then p else p
    """

    def __init__(
        self, condition: PureNode, true_branch: PureNode, false_branch: PureNode
    ):
        """
        Initializes an IfElseNode.

        Args:
            condition: The expression to evaluate (an instance of PureNode).
            true_branch: The expression to execute if condition is true (an instance of PureNode).
            false_branch: The expression to execute if condition is false (an instance of PureNode).
        """
        if not isinstance(condition, PureNode):
            raise TypeError("Condition must be an instance of PureNode.")
        if not isinstance(true_branch, PureNode):
            raise TypeError("True branch must be an instance of PureNode.")
        if not isinstance(false_branch, PureNode):
            raise TypeError("False branch must be an instance of PureNode.")

        self.condition = condition
        self.true_branch = true_branch
        self.false_branch = false_branch

    def __repr__(self):
        return (
            f"IfElseNode(condition={self.condition}, "
            f"true_branch={self.true_branch}, "
            f"false_branch={self.false_branch})"
        )

    def eval(self, env: Environment) -> PureNode:
        cond_val = self.condition.eval(env)
        if cond_val:
            return self.true_branch.eval(env)
        return self.false_branch.eval(env)

    def possible_values(self, env: Environment) -> set[PureNode]:
        return self.true_branch.possible_values(
            env
        ) | self.false_branch.possible_values(env)


class ConsNode(PureNode):
    """
    Represents a 'cons p p' expression.
    Grammar: cons p p
    """

    def __init__(self, head: PureNode, tail: PureNode):
        """
        Initializes a ConsNode.

        Args:
            head: The first expression (an instance of PureNode).
            tail: The second expression (an instance of PureNode).
        """
        if not isinstance(head, PureNode):
            raise TypeError("Head must be an instance of PureNode.")
        if not isinstance(tail, PureNode):
            raise TypeError("Tail must be an instance of PureNode.")
        self.head = head
        self.tail = tail

    def __repr__(self):
        return f"ConsNode(head={self.head}, tail={self.tail})"

    def eval(self, env: Environment) -> PureNode:
        return ConsNode(self.head.eval(env), self.tail.eval(env))

    def possible_values(self, env: Environment) -> set[PureNode]:
        return set(
            ConsNode(phead, ptail)
            for phead, ptail in itertools.product(
                self.head.possible_values(env), self.tail.possible_values(env)
            )
        )


class NilNode(PureNode):
    """
    Represents a 'nil' value.
    Grammar: nil
    """

    def __init__(self):
        # No specific data needed for a constant 'nil' value
        pass

    def __repr__(self):
        return "NilNode()"

    def possible_values(self, env: Environment) -> set[PureNode]:
        return set([self])


# --- Expression (e) Classes ---


class ExpressionNode(ASTNode):
    """
    Abstract base class for expression (e) nodes.
    """

    @abc.abstractmethod
    def sample(self, env: Environment) -> PureNode:
        """Sample a value from the program distribution"""

    def sample_toplevel(self, k=1) -> list[PureNode]:
        """Sample at the top-level with an empty environment

        Args:
            k: number of samples to evaluate

        Return:
            Resulting value from sampling
        """
        return [self.sample(Environment()) for _ in range(k)]


class ReturnNode(ExpressionNode):
    """
    Represents a 'return p' expression.
    Grammar: return p
    """

    def __init__(self, value: PureNode):
        """
        Initializes a ReturnNode.

        Args:
            value: The pure value to be returned (an instance of PureNode).
        """
        if not isinstance(value, PureNode):
            raise TypeError("Return value must be an instance of PureNode.")
        self.value = value

    def __repr__(self):
        return f"ReturnNode(value={self.value})"

    def sample(self, env: Environment) -> PureNode:
        return self.value.eval(env)


class FlipNode(ExpressionNode):
    """
    Represents a 'flip theta' expression.
    Grammar: flip theta
    """

    def __init__(self, theta: float):
        """
        Initializes a FlipNode.

        Args:
            theta: The probability value (a float between 0.0 and 1.0).
        """
        if not isinstance(theta, (int, float)):
            raise TypeError("Theta must be a number.")
        if not (0.0 <= theta <= 1.0):
            raise ValueError("Theta must be between 0.0 and 1.0 (inclusive).")
        self.theta = float(theta)

    def __repr__(self):
        return f"FlipNode(theta={self.theta})"

    def sample(self, env: Environment) -> PureNode:
        return boolean(random.random() < self.theta)


class SequenceNode(ExpressionNode):
    """
    Represents a 'x <- e; e' sequential expression.
    Grammar: x <- e; e
    """

    def __init__(
        self,
        variable_name: str,
        assignment_expr: ExpressionNode,
        next_expr: ExpressionNode,
    ):
        """
        Initializes a SequenceNode.

        Args:
            variable_name: The name of the variable 'x' to which the result of assignment_expr is bound.
            assignment_expr: The expression 'e' whose result is assigned to 'x' (an instance of ExpressionNode).
            next_expr: The subsequent expression 'e' to execute (an instance of ExpressionNode).
        """
        if not isinstance(variable_name, str) or not variable_name:
            raise ValueError("Variable name must be a non-empty string.")
        if not isinstance(assignment_expr, ExpressionNode):
            raise TypeError(
                "Assignment expression must be an instance of ExpressionNode."
            )
        if not isinstance(next_expr, ExpressionNode):
            raise TypeError("Next expression must be an instance of ExpressionNode.")

        self.variable_name = variable_name
        self.assignment_expr = assignment_expr
        self.next_expr = next_expr

    def __repr__(self):
        return (
            f"SequenceNode(variable_name='{self.variable_name}', "
            f"assignment_expr={self.assignment_expr}, "
            f"next_expr={self.next_expr})"
        )

    def sample(self, env: Environment) -> PureNode:
        bind_val = self.assignment_expr.sample(env)
        env.add_binding(self.variable_name, bind_val)
        return self.next_expr.sample(env)
