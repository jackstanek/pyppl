from pyppl import ast
from pyppl.environment import BaseEnv


class UnboundNameError(Exception):
    """Represents a name which doesn't have a binding"""

    def __init__(self, name: str, *args: object) -> None:
        self.name = name
        super().__init__(*args)


class NameEnv(BaseEnv):
    """Environment for name analysis"""

    scope_factory = set

    def add_name(self, name: str):
        """
        Add a name to the current scope
        """
        self.scopes[-1].add(name)

    def check_binding(self, name: str):
        """
        Check that a name is bound, and raise UnboundNameError otherwise
        """
        for scope in reversed(self.scopes):
            if name in scope:
                return
        raise UnboundNameError(name)


def name_analysis(prog: ast.Program):
    """
    Perform name analysis on the program.

    Check that names are accessed in the proper scopes.
    """
    env = NameEnv()
    for defn_name in prog.defns.keys():
        env.add_name(defn_name)

    for defn_val in prog.defns.values():
        name_analysis_expr(env, defn_val)

    name_analysis_expr(env, prog.expr)


def name_analysis_expr(env: NameEnv, expr: ast.ExpressionNode):
    """
    Perform name analysis on an expression
    """
    if isinstance(expr, ast.VariableNode):
        env.check_binding(expr.name)
    elif isinstance(expr, ast.SequenceNode):
        name_analysis_expr(env, expr.assignment_expr)
        with env.local_scope():
            env.add_name(expr.variable_name)
            name_analysis_expr(env, expr.next_expr)
    else:
        for subexpr in expr.subexpressions:
            name_analysis_expr(env, subexpr)
