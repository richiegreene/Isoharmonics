import math
from fractions import Fraction
from functools import reduce
from math import gcd

def calculate_edo_step(cents, edo):
    step_size = 1200 / edo
    step = round(cents / step_size)
    error = step * step_size - cents
    step_str = f"-{abs(step)}" if step < 0 else str(step)
    return step_str, error

def calculate_12edo_step(cents):
    step_size = 1200 / 12
    step = round(cents / step_size)
    error = step * step_size - cents
    return step, error

def ratio_to_cents(ratio):
    return 1200 * math.log2(ratio)

def generate_iso_series(fundamental, isoharmonic, partials_above, partials_below):
    series = []
    current_ratio = isoharmonic
    for _ in range(partials_below):
        current_ratio = current_ratio - fundamental
        series.insert(0, current_ratio)
    series.append(isoharmonic)
    current_ratio = isoharmonic
    for _ in range(partials_above):
        current_ratio = current_ratio + fundamental
        series.append(current_ratio)
    return series

def find_gcd(list):
    return reduce(gcd, list)

def find_lcd(denominators):
    def lcm(a, b):
        return a * b // gcd(a, b)
    return reduce(lcm, denominators)

def format_series_segment(series):
    fractions = [Fraction(ratio).limit_denominator() for ratio in series]
    denominators = [frac.denominator for frac in fractions]
    lcd = find_lcd(denominators)
    numerators = [int(frac.numerator * (lcd / frac.denominator)) for frac in fractions]
    
    if lcd == 1:
        return ':'.join(map(str, numerators))
    else:
        return f"({':'.join(map(str, numerators))})/{lcd}"

def simplify_ratio(ratio):
    frac = Fraction(ratio).limit_denominator()
    return f"{frac.numerator}/{frac.denominator}"
