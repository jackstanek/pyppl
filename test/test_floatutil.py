from pyppl.floatutil import clamp


def test_clamp():
    """Test clamping values"""
    for i in range(100):
        x = 2 - 0.03 * i
        if x < 0:
            assert clamp(x) == 0.0
        elif x > 1:
            assert clamp(x) == 1.0
        else:
            assert clamp(x) == x
