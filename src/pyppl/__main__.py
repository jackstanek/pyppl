import argparse
import random

from pyppl import ast
from pyppl.params import ParamVector


def init_params(expr: ast.ExpressionNode) -> ParamVector:
    """Initialize parameters to random values."""
    return ParamVector({k: random.random()} for k in expr.params)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True, dest="command")
    generate_parser = subparsers.add_parser("generate")
    learn_parser = subparsers.add_parser("learn")
    args = parser.parse_args()

    print(args)



if __name__ == "__main__":
    main()
