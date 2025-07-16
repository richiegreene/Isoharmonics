def to_subscript(number):
    return str(number).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))
