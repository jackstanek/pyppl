def clamp(x: float, low: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a float value.

    Args:
        x: value to clamp
        low: lower end of the interval
        hi: upper end of the interval
    """
    return max((min((hi, x)), low))
