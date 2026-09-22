"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable

# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(x: float, y: float) -> float:
    return x*y

def id(x: float) -> float:
    return x

def add(x: float, y: float) -> float:
    return x+y

def neg(x: float) -> float:
    return -1*x

def lt(x: float, y: float) -> float:
    if(x < y):
        return 1.0
    return 0.0

def eq(x: float, y: float) -> float:
    if(x == y):
        return 1.0
    return 0.0

def max(x: float, y: float) -> float:
    if(x < y):
        return y
    return x

def is_close(x: float, y: float, tol: float = 1e-2) -> bool:
    return abs(x - y) < tol

def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))

def relu(x: float) -> float:
    return x if x > 0.0 else 0.0


def log(x: float) -> float:
    return math.log(x)


def exp(x: float) -> float:
    return math.exp(x)


def inv(x: float) -> float:
    return 1.0 / x


def log_back(x: float, d: float) -> float:
    return d / x


def inv_back(x: float, d: float) -> float:
    return -d / (x * x)


def relu_back(x: float, d: float) -> float:
    return d if x > 0.0 else 0.0

# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.

def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order map.

    Args:
    ----
        fn: Function from one value to one value.

    Returns:
    -------
        A function that takes a list/iterable, applies `fn` to each element,
        and returns a list or iterable of results.

    """
    def _map(ls: Iterable[float]) -> Iterable[float]:
        return [fn(x) for x in ls]

    return _map

def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order zipWith (or map2).

    Args:
    ----
        fn: Two-argument function that combines two floats into one float.

    Returns:
    -------
        A function that takes two iterables, applies `fn` pairwise,
        and returns an iterable/list of results.

    """
    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        return [fn(x, y) for x, y in zip(ls1, ls2)]

    return _zipWith

def reduce(fn: Callable[[float,float], float], start: float) -> Callable[[Iterable[float]], float]:
    """Higher-order reduce.

    Args:
    ----
        fn: Two-argument function that combines an accumulator and a value.
        start: Initial value for the accumulator.

    Returns:
    -------
        A function that takes an iterable and reduces it to a single float value.

    """
    def _reduce(ls: Iterable[float]) -> float:
        val = start
        for x in ls:
            val = fn(val, x)
        return val

    return _reduce

def negList(ls: Iterable[float]) -> Iterable[float]:
    return map(neg)(ls)

def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float] :
    return zipWith(add)(ls1, ls2)

def sum(ls: Iterable[float]) -> float:
    return reduce(add, 0.0)(ls)

def prod(ls: Iterable[float]) -> float:
    return reduce(mul, 1.0)(ls)