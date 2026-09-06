from typing import Callable, List, Tuple

import pytest
from hypothesis import given
from hypothesis.strategies import lists

from minitorch import MathTest
import minitorch
from minitorch.operators import (
    add,
    addLists,
    eq,
    id,
    inv,
    inv_back,
    log_back,
    lt,
    max,
    mul,
    neg,
    negList,
    prod,
    relu,
    relu_back,
)

from .strategies import assert_close, small_floats

# ## Task 0.1 Basic hypothesis tests.


@pytest.mark.task0_1
@given(small_floats, small_floats)
def test_same_as_python(x: float, y: float) -> None:
    """Check that the main operators all return the same value of the python version"""
    assert_close(mul(x, y), x * y)
    assert_close(add(x, y), x + y)
    assert_close(neg(x), -x)
    assert_close(max(x, y), x if x > y else y)
    if abs(x) > 1e-5:
        assert_close(inv(x), 1.0 / x)


@pytest.mark.task0_1
@given(small_floats)
def test_relu(a: float) -> None:
    if a > 0:
        assert relu(a) == a
    if a < 0:
        assert relu(a) == 0.0


@pytest.mark.task0_1
@given(small_floats, small_floats)
def test_relu_back(a: float, b: float) -> None:
    if a > 0:
        assert relu_back(a, b) == b
    if a < 0:
        assert relu_back(a, b) == 0.0


@pytest.mark.task0_1
@given(small_floats)
def test_id(a: float) -> None:
    assert id(a) == a


@pytest.mark.task0_1
@given(small_floats)
def test_lt(a: float) -> None:
    """Check that a - 1.0 is always less than a"""
    assert lt(a - 1.0, a) == 1.0
    assert lt(a, a - 1.0) == 0.0


@pytest.mark.task0_1
@given(small_floats)
def test_max(a: float) -> None:
    assert max(a - 1.0, a) == a
    assert max(a, a - 1.0) == a
    assert max(a + 1.0, a) == a + 1.0
    assert max(a, a + 1.0) == a + 1.0


@pytest.mark.task0_1
@given(small_floats)
def test_eq(a: float) -> None:
    assert eq(a, a) == 1.0
    assert eq(a, a - 1.0) == 0.0
    assert eq(a, a + 1.0) == 0.0


# ## Task 0.2 - Property Testing

# Implement the following property checks
# that ensure that your operators obey basic
# mathematical rules.


@pytest.mark.task0_2
@given(small_floats)
def test_sigmoid(a: float) -> None:
    """Check properties of the sigmoid function, specifically
    * It is always between 0.0 and 1.0.
    * one minus sigmoid is the same as sigmoid of the negative
    * It crosses 0 at 0.5
    * It is  strictly increasing.
    """
    # TODO: Implement for Task 0.2.
    # 1. It is always between 0.0 and 1.0
    val = minitorch.operators.sigmoid(a)
    assert 0.0 <= val <= 1.0

    # 2. one minus sigmoid is the same as sigmoid of the negative
    assert minitorch.operators.sigmoid(-a) == pytest.approx(1.0 - val, abs=1e-5)

    # 3. It crosses 0 at 0.5
    assert minitorch.operators.sigmoid(0.0) == pytest.approx(0.5, abs=1e-5)

    # 4. Strictly increasing (<= handles tail saturation, < tests the active region)
    delta = 1.0
    assert minitorch.operators.sigmoid(a) <= minitorch.operators.sigmoid(a + delta)
    if -5.0 <= a <= 5.0:
        assert minitorch.operators.sigmoid(a) < minitorch.operators.sigmoid(a + delta)
        
        
@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_transitive(a: float, b: float, c: float) -> None:
    """Test the transitive property of less-than (a < b and b < c implies a < c)"""
    # TODO: Implement for Task 0.2.
    #Test the transitive property of less-than (a < b and b < c implies a < c)
    if minitorch.operators.lt(a, b) and minitorch.operators.lt(b, c):
        assert minitorch.operators.lt(a, c)


@pytest.mark.task0_2
def test_symmetric() -> None:
    """Write a test that ensures that :func:`minitorch.operators.mul` is symmetric, i.e.
    gives the same value regardless of the order of its input.
    """
    # TODO: Implement for Task 0.2.
    test_cases = [
        (0.0, 5.0),
        (-3.0, 4.0),
        (2.5, -7.1),
        (1e-4, 1e4),
        (-0.5, -0.2),
        (42.0, 13.37),
    ]
    for a, b in test_cases:
        assert minitorch.operators.mul(a, b) == minitorch.operators.mul(b, a)


@pytest.mark.task0_2
def test_distribute() -> None:
    r"""Write a test that ensures that your operators distribute, i.e.
    :math:`z \times (x + y) = z \times x + z \times y`
    """
    # TODO: Implement for Task 0.2.
    test_cases = [
        (2.0, 3.0, 4.0),
        (-1.5, 2.5, -3.0),
        (0.0, 5.0, 10.0),
        (0.5, 0.25, 0.75),
        (-2.0, -4.0, 5.0),
    ]

    for z, x, y in test_cases:
        lhs = minitorch.operators.mul(z, minitorch.operators.add(x, y))
        rhs = minitorch.operators.add(
            minitorch.operators.mul(z, x), minitorch.operators.mul(z, y)
        )
        assert lhs == pytest.approx(rhs)


@pytest.mark.task0_2
def test_other() -> None:
    """Write a test that ensures some other property holds for your functions."""
    # TODO: Implement for Task 0.2.
    test_values = [-10.5, -1.0, -0.001, 0.0, 0.001, 2.5, 42.0]

    for x in test_values:
        r1 = minitorch.operators.relu(x)
        r2 = minitorch.operators.relu(r1)

        # Non-negative property
        assert r1 >= 0.0
        # Idempotent property
        assert r1 == pytest.approx(r2)


# ## Task 0.3  - Higher-order functions

# These tests check that your higher-order functions obey basic
# properties.


@pytest.mark.task0_3
@given(small_floats, small_floats, small_floats, small_floats)
def test_zip_with(a: float, b: float, c: float, d: float) -> None:
    x1, x2 = addLists([a, b], [c, d])
    y1, y2 = a + c, b + d
    assert_close(x1, y1)
    assert_close(x2, y2)


@pytest.mark.task0_3
@given(
    lists(small_floats, min_size=5, max_size=5),
    lists(small_floats, min_size=5, max_size=5),
)
def test_sum_distribute(ls1: List[float], ls2: List[float]) -> None:
    """Write a test that ensures that the sum of `ls1` plus the sum of `ls2`
    is the same as the sum of each element of `ls1` plus each element of `ls2`.
    """
    # TODO: Implement for Task 0.3.
    raise NotImplementedError("Need to implement for Task 0.3")


@pytest.mark.task0_3
@given(lists(small_floats))
def test_sum(ls: List[float]) -> None:
    assert_close(sum(ls), minitorch.operators.sum(ls))


@pytest.mark.task0_3
@given(small_floats, small_floats, small_floats)
def test_prod(x: float, y: float, z: float) -> None:
    assert_close(prod([x, y, z]), x * y * z)


@pytest.mark.task0_3
@given(lists(small_floats))
def test_negList(ls: List[float]) -> None:
    check = negList(ls)
    for i, j in zip(ls, check):
        assert_close(i, -j)


# ## Generic mathematical tests

# For each unit this generic set of mathematical tests will run.


one_arg, two_arg, _ = MathTest._tests()


@given(small_floats)
@pytest.mark.parametrize("fn", one_arg)
def test_one_args(fn: Tuple[str, Callable[[float], float]], t1: float) -> None:
    name, base_fn = fn
    base_fn(t1)


@given(small_floats, small_floats)
@pytest.mark.parametrize("fn", two_arg)
def test_two_args(
    fn: Tuple[str, Callable[[float, float], float]], t1: float, t2: float
) -> None:
    name, base_fn = fn
    base_fn(t1, t2)


@given(small_floats, small_floats)
def test_backs(a: float, b: float) -> None:
    relu_back(a, b)
    inv_back(a + 2.4, b)
    log_back(abs(a) + 4, b)
