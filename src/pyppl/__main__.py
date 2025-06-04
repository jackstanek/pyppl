import argparse
import random

from pyppl import ast
from pyppl.parser import parse
from pyppl.params import ParamVector


def init_params(expr: ast.ExpressionNode) -> ParamVector:
    """Initialize parameters to random values."""
    return ParamVector({k: random.random()} for k in expr.params)


def param_val(param: str) -> tuple[str, float]:
    """Convert a string in the format "p=v" to a tuple (p, v)

    Args:
        param: the param=value string

    Returns:
    """
    try:
        p, v = param.split("=")
        return (p, float(v))
    except ValueError as err:
        raise ValueError("improper parameter string format") from err


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True, dest="command")
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument(
        "program", type=argparse.FileType("r"), help="path to program source"
    )
    # generate_parser.add_argument("output_dir", help="directory to write samples to")
    generate_parser.add_argument(
        "--n-samples", "-n", type=int, default=10, help="number of samples to generate"
    )
    generate_parser.add_argument(
        "--param",
        "-p",
        action="append",
        dest="params",
        type=param_val,
        help="parameter value to use (format: <param>=<value>)",
    )
    # learn_parser = subparsers.add_parser("learn")
    args = parser.parse_args()

    if args.command == "generate":
        # os.makedirs(args.output_dir, exist_ok=True)
        with args.program:
            prog_ast = parse(args.program.read())
            if args.params:
                env = ast.Environment(args.params)
            else:
                env = None
            for sample in prog_ast.sample_toplevel(env=env, k=args.n_samples):
                print(sample)


if __name__ == "__main__":
    main()
