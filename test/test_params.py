import pytest

from pyppl.params import ParamVector


def test_add_success():
    """
    Tests successful addition of two ParamVector instances with matching keys.
    """
    v1 = ParamVector({"a": 1.0, "b": 2.0})
    v2 = ParamVector({"a": 3.0, "b": 4.0})
    v3 = v1 + v2
    assert isinstance(v3, ParamVector)  # Check if the result is a ParamVector
    assert v3 == {"a": 4.0, "b": 6.0}


def test_add_mismatched_keys():
    """
    Tests that addition raises ValueError when keys do not match.
    """
    v1 = ParamVector({"a": 1.0})
    v2 = ParamVector({"b": 2.0})
    with pytest.raises(ValueError, match="keys in vectors do not match"):
        _ = v1 + v2


def test_sub_success():
    """
    Tests successful subtraction of two ParamVector instances with matching keys.
    """
    v1 = ParamVector({"a": 5.0, "b": 7.0})
    v2 = ParamVector({"a": 1.0, "b": 3.0})
    v3 = v1 - v2
    assert isinstance(v3, ParamVector)  # Check if the result is a ParamVector
    assert v3 == {"a": 4.0, "b": 4.0}


def test_sub_mismatched_keys():
    """
    Tests that subtraction raises ValueError when keys do not match.
    """
    v1 = ParamVector({"a": 1.0})
    v2 = ParamVector({"b": 2.0})
    with pytest.raises(ValueError, match="keys in vectors do not match"):
        _ = v1 - v2


def test_mul_scalar_int():
    """
    Tests multiplication of ParamVector by an integer scalar.
    """
    v1 = ParamVector({"a": 2.0, "b": 3.0})
    v2 = v1 * 2
    assert isinstance(v2, ParamVector)  # Check if the result is a ParamVector
    assert v2 == {"a": 4.0, "b": 6.0}


def test_mul_scalar_float():
    """
    Tests multiplication of ParamVector by a float scalar.
    """
    v1 = ParamVector({"a": 2.0, "b": 3.0})
    v3 = v1 * 0.5
    assert isinstance(v3, ParamVector)  # Check if the result is a ParamVector
    assert v3 == {"a": 1.0, "b": 1.5}


def test_truediv_scalar_int():
    """
    Tests division of ParamVector by an integer scalar.
    """
    v1 = ParamVector({"a": 4.0, "b": 6.0})
    v2 = v1 / 2
    assert isinstance(v2, ParamVector)  # Check if the result is a ParamVector
    assert v2 == {"a": 2.0, "b": 3.0}


def test_truediv_scalar_float():
    """
    Tests division of ParamVector by a float scalar.
    """
    v1 = ParamVector({"a": 4.0, "b": 6.0})
    v3 = v1 / 0.5
    assert isinstance(v3, ParamVector)  # Check if the result is a ParamVector
    assert v3 == {"a": 8.0, "b": 12.0}


def test_truediv_by_zero():
    """
    Tests that division by zero raises ZeroDivisionError.
    """
    v1 = ParamVector({"a": 1.0})
    with pytest.raises(ZeroDivisionError):
        _ = v1 / 0


def test_neg():
    """
    Tests negation of a ParamVector.
    """
    v1 = ParamVector({"a": 2.0, "b": -3.0})
    v2 = -v1
    assert isinstance(v2, ParamVector)  # Check if the result is a ParamVector
    assert v2 == {"a": -2.0, "b": 3.0}


def test_setitem_existing_key():
    """
    Tests setting a value for an existing key.
    Ensures the value is converted to float.
    """
    v = ParamVector({"a": 1.0, "b": 2.0})
    v["a"] = 10
    assert v == {"a": 10.0, "b": 2.0}
    assert isinstance(v["a"], float)
    v["b"] = 5.5
    assert v == {"a": 10.0, "b": 5.5}
    assert isinstance(v["b"], float)


def test_setitem_new_key_raises_error():
    """
    Tests that setting a value for a non-existent key raises ValueError.
    """
    v = ParamVector({"a": 1.0})
    with pytest.raises(ValueError, match="no such key c in vector"):
        v["c"] = 3.0


def test_initialization_and_type_conversion():
    """
    Tests initial creation of ParamVector and ensures values are floats.
    """
    v = ParamVector(x=10, y=20.5)
    assert v == {"x": 10.0, "y": 20.5}
    assert isinstance(v["x"], float)
    assert isinstance(v["y"], float)


def test_empty_param_vector_operations():
    """
    Tests operations on an empty ParamVector.
    """
    v_empty = ParamVector({})
    assert v_empty + v_empty == {}
    assert v_empty - v_empty == {}
    assert v_empty * 5 == {}
    assert v_empty / 5 == {}
    assert -v_empty == {}
    # Test setitem on empty vector (should still raise error for new key)
    with pytest.raises(ValueError, match="no such key a in vector"):
        v_empty["a"] = 1.0


def test_iadd_success():
    """
    Tests in-place addition (+=) of two ParamVector instances.
    """
    v1 = ParamVector({"a": 1.0, "b": 2.0})
    v2 = ParamVector({"a": 3.0, "b": 4.0})
    v1 += v2
    assert v1 == {"a": 4.0, "b": 6.0}


def test_iadd_mismatched_keys():
    """
    Tests that in-place addition (+=) raises ValueError when keys do not match.
    """
    v1 = ParamVector({"a": 1.0})
    v2 = ParamVector({"b": 2.0})
    with pytest.raises(ValueError, match="keys in vectors do not match"):
        v1 += v2


def test_isub_success():
    """
    Tests in-place subtraction (-=) of two ParamVector instances.
    """
    v1 = ParamVector({"a": 5.0, "b": 7.0})
    v2 = ParamVector({"a": 1.0, "b": 3.0})
    v1 -= v2
    assert v1 == {"a": 4.0, "b": 4.0}


def test_isub_mismatched_keys():
    """
    Tests that in-place subtraction (-=) raises ValueError when keys do not match.
    """
    v1 = ParamVector({"a": 1.0})
    v2 = ParamVector({"b": 2.0})
    with pytest.raises(ValueError, match="keys in vectors do not match"):
        v1 -= v2


def test_imul_scalar():
    """
    Tests in-place multiplication (*=) of ParamVector by a scalar.
    """
    v1 = ParamVector({"a": 2.0, "b": 3.0})
    v1 *= 2
    assert v1 == {"a": 4.0, "b": 6.0}
    v1 *= 0.5
    assert v1 == {"a": 2.0, "b": 3.0}  # 4.0 * 0.5 = 2.0, 6.0 * 0.5 = 3.0


def test_itruediv_scalar():
    """
    Tests in-place division (/=) of ParamVector by a scalar.
    """
    v1 = ParamVector({"a": 4.0, "b": 6.0})
    v1 /= 2
    assert v1 == {"a": 2.0, "b": 3.0}
    v1 /= 0.5
    assert v1 == {"a": 4.0, "b": 6.0}  # 2.0 / 0.5 = 4.0, 3.0 / 0.5 = 6.0


def test_itruediv_by_zero():
    """
    Tests that in-place division (/=) by zero raises ZeroDivisionError.
    """
    v1 = ParamVector({"a": 1.0})
    with pytest.raises(ZeroDivisionError):
        v1 /= 0
