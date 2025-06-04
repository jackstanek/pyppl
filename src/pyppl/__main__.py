import argparse
import pickle
import random
from typing import Any

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



# Define the updated classes provided by the user
class PickleLoader:
    """Wrapper for loading pickle files"""

    def __init__(self, path: str):
        self.path = path
        self.pickle_file = None # Initialize to None

    def load(self) -> Any:
        """Loads an object from the pickle file.

        This method should be called within a 'with' statement.

        Raises:
            RuntimeError: If the method is called outside of a 'with' statement.

        Returns:
            Any: The object loaded from the pickle file.
        """
        # Ensure pickle_file is not None before calling pickle.load
        if self.pickle_file is None:
            raise RuntimeError("PickleLoader must be used within a 'with' statement.")
        return pickle.load(self.pickle_file)

    def __enter__(self):
        # The 'open' function will be mocked by pytest
        self.pickle_file = open(self.path, mode="rb")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pickle_file: # Ensure file was opened before attempting to close
            self.pickle_file.close()
        return exc_type is None


class PickleDumper:
    """Wrapper for dumping pickle files"""

    def __init__(self, path: str):
        self.path = path
        self.pickle_file = None # Initialize to None
        self.pickler = None # Initialize to None

    def dump(self, obj: Any):
        """Dumps an object to the pickle file.

        This method should be called within a 'with' statement.

        Args:
            obj (Any): The object to be dumped to the pickle file.

        Raises:
            RuntimeError: If the method is called outside of a 'with' statement.
        """
        # Ensure pickler is not None before calling dump
        if self.pickler is None:
            raise RuntimeError("PickleDumper must be used within a 'with' statement.")
        self.pickler.dump(obj)

    def __enter__(self):
        # The 'open' function will be mocked by pytest
        self.pickle_file = open(self.path, mode="wb")
        self.pickler = pickle.Pickler(self.pickle_file)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pickle_file: # Ensure file was opened before attempting to close
            self.pickle_file.close()
        return exc_type is None


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True, dest="command")
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument(
        "program", type=argparse.FileType("r"), help="path to program source"
    )
    generate_parser.add_argument("data", help="file to write samples to")
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
    learn_parser = subparsers.add_parser("learn")
    learn_parser.add_argument(
        "program", type=argparse.FileType("r"), help="path to program source"
    )
    learn_parser.add_argument("data", help="path to training set")
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
