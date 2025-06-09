"""
Facilities for learning the distribution of some dataset using maximum
likelihood estimation.

Here we try to minimize the average negative log-likelihood of the dataset with
respect to the hypothesis class denoted by a proposed program. This is
equivalent to maximizing the likehlihood function with respect to the
parameters.
"""

import logging
import math

from pyppl import ast
from pyppl.params import ParamVector

LOGGER = logging.getLogger(__name__)


def avg_negative_log_likelihood(
    prog: ast.Program, params: ParamVector, data: list[ast.PureNode]
) -> float:
    """Compute the negative log-likelihood of a collection of data."""
    acc = 0
    n_possible = 0
    for datum in data:
        LOGGER.debug("Computing avg NLL for %s", datum)
        prob = prog.infer(params, datum)
        if prob > 0:
            acc -= math.log(prob)
            n_possible += 1
    return acc / n_possible


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
    grad = ParamVector.zero(params)
    n_possible = 0
    for datum in data:
        prob = prog.infer(params, datum)
        if prob > 0:
            grad -= prog.gradient(params, datum) / prob
            n_possible += 1
    return grad / n_possible


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
        params -= learning_rate * grad

    return params
