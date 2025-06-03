import random

from pyppl import ast
from pyppl.params import ParamVector


def init_params(expr: ast.ExpressionNode) -> ParamVector:
    """Initialize parameters to random values."""
    return ParamVector({k: random.random()} for k in expr.params)


def main():
    prog = ast.SequenceNode(
        "x",
        ast.FlipNode(0.5),
        ast.ReturnNode(
            ast.IfElseNode(
                ast.var("x"), ast.ConsNode(ast.var("x"), ast.NilNode()), ast.NilNode()
            )
        ),
    )

    print(prog.sample_toplevel(k=10))


if __name__ == "__main__":
    main()
