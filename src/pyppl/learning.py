import math

from pyppl import ast
from pyppl.params import ParamVector


def avg_negative_log_likelihood(
    prog: ast.Program, params: ParamVector, data: list[ast.PureNode]
) -> float:
    """Compute the negative log-likelihood of a collection of data."""
    return -sum(math.log(prog.infer(params, d)) for d in data) / len(data)


def avg_negative_log_likelihood_gradient(
    prog: ast.Program, params: ParamVector, data: list[ast.PureNode]
) -> ParamVector:
    """Compute the gradient of the negative log-likelhood function.

    Args:
        prog: an expression representing the generating program
        params: mapping of parameter names to values
        data: the training set

    Returns:
        negative log-likelihood of observing the training set given the program
        and the parameters
    """
    grad = sum(
        prog.gradient(params, val) / prog.infer(params, val) for val in data
    ) / len(data)
    if isinstance(grad, (float, int)):
        raise ValueError("empty training set or type error")
    return grad


def optimize(
    prog: ast.Program,
    data: list[ast.PureNode],
    epochs: int = 100,
    learning_rate: float = 0.01,
) -> ParamVector:
    """Optimize parameters to maximize the likelihood of the training set"""
    params = ParamVector.random(prog.params)
    for epoch in range(epochs):
        nll = avg_negative_log_likelihood(prog, params, data)
        print(f"epoch: {epoch}; nll: {nll}")
        grad = avg_negative_log_likelihood_gradient(prog, params, data)
        params += learning_rate * grad

    return params
