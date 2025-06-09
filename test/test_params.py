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
    v = ParamVector({"x": 10, "y": 20.5})
    assert v == {"x": 10.0, "y": 20.5}
    assert isinstance(v["x"], float)
    assert isinstance(v["y"], float)


def test_setting_custom_type():
    """
    Tests setting custom types for ParamVectors.
    """
    v = ParamVector({"x": 10.0, "y": 9.0}, valtype=int)
    assert v == {"x": 10, "y": 9}
    assert isinstance(v["x"], int)
    assert isinstance(v["y"], int)


def test_using_refined_types():
    """
    Tests setting a refined type for ParamVectors.
    """
    v = ParamVector({"x": 1}, valtype=lambda x: x + 1)
    assert v == {"x": 2}


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


def test_squared_l2_norm():
    """
    Tests the squared_l2_norm method.
    """
    v = ParamVector({"a": 3.0, "b": 4.0})
    assert v.squared_l2_norm() == 25.0
    v_empty = ParamVector({})
    assert v_empty.squared_l2_norm() == 0.0
    v_neg = ParamVector({"x": -2.0, "y": -3.0})
    assert v_neg.squared_l2_norm() == 13.0


# --- In-place operator tests ---


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
    assert v1 == {"a": 2.0, "b": 3.0}


def test_itruediv_scalar():
    """
    Tests in-place division (/=) of ParamVector by a scalar.
    """
    v1 = ParamVector({"a": 4.0, "b": 6.0})
    v1 /= 2
    assert v1 == {"a": 2.0, "b": 3.0}
    v1 /= 0.5
    assert v1 == {"a": 4.0, "b": 6.0}


def test_itruediv_by_zero():
    """
    Tests that in-place division (/=) by zero raises ZeroDivisionError.
    """
    v1 = ParamVector({"a": 1.0})
    with pytest.raises(ZeroDivisionError):
        v1 /= 0


def test_sum_of_param_vectors_no_start():
    """
    Tests summing a list of ParamVectors without a specified start value.
    This relies on the first element being the initial sum.
    """
    v1 = ParamVector({"a": 1.0, "b": 2.0})
    v2 = ParamVector({"a": 3.0, "b": 4.0})
    v3 = ParamVector({"a": 5.0, "b": 6.0})
    result = sum([v1, v2, v3])
    assert isinstance(result, ParamVector)
    assert result == {"a": 9.0, "b": 12.0}


def test_sum_of_param_vectors_with_paramvector_start():
    """
    Tests summing a list of ParamVectors with a ParamVector as the start value.
    """
    v1 = ParamVector({"a": 1.0, "b": 2.0})
    v2 = ParamVector({"a": 3.0, "b": 4.0})
    start_vec = ParamVector({"a": 10.0, "b": 20.0})
    result = sum([v1, v2], start=start_vec)
    assert isinstance(result, ParamVector)
    assert result == {"a": 14.0, "b": 26.0}


def test_sum_of_param_vectors_with_zero_start():
    """
    Tests summing a list of ParamVectors with 0 as the start value.
    This relies on the __radd__ method.
    """
    v1 = ParamVector({"a": 1.0, "b": 2.0})
    v2 = ParamVector({"a": 3.0, "b": 4.0})
    result = sum([v1, v2], start=0)
    assert isinstance(result, ParamVector)
    assert result == {"a": 4.0, "b": 6.0}


def test_sum_empty_list_with_paramvector_start():
    """
    Tests summing an empty list with a ParamVector start value.
    Should return the start value.
    """
    start_vec = ParamVector({"a": 10.0, "b": 20.0})
    result = sum([], start=start_vec)
    assert isinstance(result, ParamVector)
    assert result == start_vec


def test_sum_paramvector_quotients():
    """
    Tests that we can sum over quotients of ParamVectors
    """
    v1 = ParamVector({"a": 2.0, "b": 4.0})
    v2 = ParamVector({"a": 4.0, "b": 8.0})
    assert sum(v / 2 for v in [v1, v2]) == {"a": 3.0, "b": 6.0}


def test_sum_mismatched_keys_in_list():
    """
    Tests that summing a list with ParamVectors having mismatched keys raises ValueError.
    """
    v1 = ParamVector({"a": 1.0, "b": 2.0})
    v2 = ParamVector({"a": 3.0, "c": 4.0})  # Mismatched keys
    with pytest.raises(ValueError, match="keys in vectors do not match"):
        _ = sum([v1, v2])


def test_sum_mismatched_keys_with_start():
    """
    Tests that summing a list with ParamVectors having mismatched keys (even with a matching start)
    raises ValueError during the addition process.
    """
    v1 = ParamVector({"a": 1.0, "b": 2.0})
    v2 = ParamVector({"a": 3.0, "c": 4.0})  # Mismatched keys
    start_vec = ParamVector({"a": 0.0, "b": 0.0})
    with pytest.raises(ValueError, match="keys in vectors do not match"):
        _ = sum([v1, v2], start=start_vec)
