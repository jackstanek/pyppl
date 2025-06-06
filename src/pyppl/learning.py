import math

from pyppl import ast
from pyppl.params import ParamVector


def avg_negative_log_likelihood(
    prog: ast.ExpressionNode, params: ParamVector, data: list[ast.PureNode]
) -> float:
    """Compute the negative log-likelihood of a collection of data."""
    env = ast.Environment(params)
    return -sum(math.log(prog.infer(env, d)) for d in data) / len(data)


def avg_negative_log_likelihood_gradient(
    prog: ast.ExpressionNode, params: ParamVector, data: list[ast.PureNode]
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
    env = ast.Environment(params)
    grad = sum(prog.gradient(env, val) / prog.infer(env, val) for val in data) / len(
        data
    )
    if isinstance(grad, (float, int)):
        raise ValueError("empty training set or type error")
    return grad


def optimize(
    prog: ast.ExpressionNode,
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
