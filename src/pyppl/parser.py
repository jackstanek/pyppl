from lark import Lark, Transformer, v_args

from pyppl import ast

pyppl_parser = Lark(
    r"""
    value             : eff_expr
    eff_expr          : VAR_OR_PARAM_NAME "<-" non_bind_eff_expr ";" eff_expr -> bind_expr
                      | non_bind_eff_expr
    non_bind_eff_expr : "flip" param -> flip_expr
                      | "return" pure_expr -> return_expr
                      | "(" eff_expr ")"
    param             : FLOAT -> float_param
                      | VAR_OR_PARAM_NAME -> sym_param
    pure_expr         : "if" pure_expr "then" pure_expr "else" non_if_pure_expr -> if_then_else
                      | non_if_pure_expr
    non_if_pure_expr  : "true" -> true
                      | "false" -> false
                      | "cons" pure_expr pure_expr -> cons
                      | "nil" -> nil
                      | "(" pure_expr ")"
                      | VAR_OR_PARAM_NAME -> var
    # The negative lookahead `(?!...)` ensures that the regex will not match
    # if the current position is at the start of any of the listed keywords.
    VAR_OR_PARAM_NAME : /(?!if\b|then\b|else\b|true\b|false\b|cons\b|nil\b|flip\b|return\b)[a-zA-Z_][a-zA-Z0-9_]*/

    %import common.WS
    %import common.FLOAT
    %ignore WS

    """,
    start="value",
    parser="lalr",
)


@v_args(inline=True)  # This decorator passes the children directly to the methods
class PypplTransformer(Transformer):
    """
    Transforms a Lark parse tree into an Abstract Syntax Tree (AST)
    using the defined ASTNode classes.
    """

    # Top-level rule
    def value(self, eff_expr):
        """Handles the 'value' rule, the starting point of the grammar.

        Args:
            eff_expr: The transformed effectful expression.

        Returns:
            The transformed effectful expression.
        """
        return eff_expr

    # Effectful Expressions (e)
    def bind_expr(self, var_name, assignment_expr, next_expr):
        """Handles the 'bind_expr' rule to create a SequenceNode.

        This rule corresponds to the syntax `x <- e; e`.

        Args:
            var_name: The name of the variable being bound.
            assignment_expr: The expression whose result is assigned to `var_name`.
            next_expr: The subsequent expression to execute.

        Returns:
            A `SequenceNode` representing the bind expression.
        """
        return ast.SequenceNode(str(var_name), assignment_expr, next_expr)

    def flip_expr(self, param):
        """Handles the 'flip_expr' rule to create a FlipNode.

        This rule corresponds to the syntax `flip theta`.

        Args:
            param: The probability parameter (float or variable name).

        Returns:
            A `FlipNode` representing the flip expression.
        """
        return ast.FlipNode(param)

    def return_expr(self, pure_expr):
        """Handles the 'return_expr' rule to create a ReturnNode.

        This rule corresponds to the syntax `return p`.

        Args:
            pure_expr: The transformed pure expression to be returned.

        Returns:
            A `ReturnNode` representing the return expression.
        """
        return ast.ReturnNode(pure_expr)

    # Parameters
    def float_param(self, value):
        """Handles the 'float_param' rule, converting the token to a float.

        Args:
            value: The token representing the float value.

        Returns:
            The float value of the parameter.
        """
        return float(value)

    def sym_param(self, name):
        """Handles the 'sym_param' rule for symbolic parameters.

        Args:
            name: The token representing the symbolic name.

        Returns:
            The string value of the symbolic parameter.
        """
        return str(name)

    # Pure Expressions (p)
    def if_then_else(self, condition, true_branch, false_branch):
        """Handles the 'if_then_else' rule to create an IfElseNode.

        This rule corresponds to the syntax `if p then p else p`.

        Args:
            condition: The transformed condition expression.
            true_branch: The transformed expression for the true branch.
            false_branch: The transformed expression for the false branch.

        Returns:
            An `IfElseNode` representing the conditional expression.
        """
        return ast.IfElseNode(condition, true_branch, false_branch)

    def true(self):
        """Handles the 'true' literal.

        Returns:
            A `TrueNode` representing the boolean true.
        """
        return ast.TrueNode()

    def false(self):
        """Handles the 'false' literal.

        Returns:
            A `FalseNode` representing the boolean false.
        """
        return ast.FalseNode()

    def cons(self, head, tail):
        """Handles the 'cons' rule to create a ConsNode.

        This rule corresponds to the syntax `cons p p`.

        Args:
            head: The transformed head expression.
            tail: The transformed tail expression.

        Returns:
            A `ConsNode` representing the cons expression.
        """
        return ast.ConsNode(head, tail)

    def nil(self):
        """Handles the 'nil' literal.

        Returns:
            A `NilNode` representing the nil value.
        """
        return ast.NilNode()

    def var(self, name):
        """Handles the 'var' rule to create a VariableNode.

        This rule corresponds to variable names in pure expressions.

        Args:
            name: The token representing the variable name.

        Returns:
            A `VariableNode` representing the variable.
        """
        return ast.VariableNode(str(name))

    # Terminal rules
    def VAR_OR_PARAM_NAME(self, token):
        """Handles the VAR_OR_PARAM_NAME terminal.

        Args:
            token: The token representing the variable or parameter name.

        Returns:
            The string value of the token.
        """
        return str(token)

    def FLOAT(self, token):
        """Handles the FLOAT terminal.

        Args:
            token: The token representing the float value.

        Returns:
            The float value of the token.
        """
        return float(token)

    # Default for rules that just pass through their child (e.g., non_bind_eff_expr, pure_expr)
    # Lark automatically handles these if no specific method is defined,
    # but defining them can make the transformer more explicit.
    def eff_expr(self, child):
        """Passes through the transformed child for rules that don't create new AST nodes.

        Args:
            child: The transformed child node.

        Returns:
            The transformed child node.
        """
        return child

    def non_bind_eff_expr(self, child):
        """Passes through the transformed child for rules that don't create new AST nodes.

        Args:
            child: The transformed child node.

        Returns:
            The transformed child node.
        """
        return child

    def pure_expr(self, child):
        """Passes through the transformed child for rules that don't create new AST nodes.

        Args:
            child: The transformed child node.

        Returns:
            The transformed child node.
        """
        return child

    def non_if_pure_expr(self, child):
        """Passes through the transformed child for rules that don't create new AST nodes.

        Args:
            child: The transformed child node.

        Returns:
            The transformed child node.
        """
        return child


def parse(input: str) -> ast.ExpressionNode:
    """Parse a program into an abstract syntax tree.

    Args:
        input: string containing program text

    Returns:
        parsed syntax tree

    Raises:
        lark.UnexpectedInput: on syntax error
    """
    tree = pyppl_parser.parse(input)
    return PypplTransformer().transform(tree)
