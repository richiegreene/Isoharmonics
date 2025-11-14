import math
from fractions import Fraction
from functools import reduce
from math import gcd
from itertools import combinations_with_replacement
import numpy as np

def cents(x):
    """Converts a ratio to cents."""
    return 1200 * np.log2(x)

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

def get_odd_part_of_number(num):
    if num == 0:
        return 0
    while num > 0 and num % 2 == 0:
        num //= 2
    return num

def get_odd_limit(ratio):
    """Calculates the odd limit of a given ratio."""
    try:
        ratio = Fraction(ratio).limit_denominator(10000)
        n, d = ratio.numerator, ratio.denominator
        
        n_odd_part = get_odd_part_of_number(n)
        d_odd_part = get_odd_part_of_number(d)
            
        return max(n_odd_part, d_odd_part)
    except (ValueError, ZeroDivisionError):
        return 1

def get_integer_limit(ratio):
    """Calculates the integer limit of a given ratio."""
    try:
        ratio = Fraction(ratio).limit_denominator(10000)
        return max(ratio.numerator, ratio.denominator)
    except (ValueError, ZeroDivisionError):
        return 1

def _generate_valid_numbers(limit_value, limit_mode):
    """
    Generates a set of valid numbers based on the limit mode.
    """
    valid_numbers = set()
    if limit_mode == "odd":
        # Heuristic for max number to check.
        max_num_to_check = max(limit_value * 2, 100) # A reasonable heuristic
        for num in range(1, max_num_to_check + 1):
            if get_odd_part_of_number(num) <= limit_value:
                valid_numbers.add(num)
    elif limit_mode == "integer":
        valid_numbers = set(range(1, limit_value + 1))
    # Add other limit modes here if implemented
    return valid_numbers

def generate_ji_tetra_labels(limit_value, equave_ratio, limit_mode="odd", prime_limit=7, max_exponent=4):
    """
    Generates a list of 4-note JI chords (labels) and their 3D coordinates (c1, c2, c3)
    and simplicity for the tetrahedron.
    """
    labels_data = []
    equave_ratio_float = float(equave_ratio)

    valid_numbers = _generate_valid_numbers(limit_value, limit_mode)
            
    if not valid_numbers:
        return []

    sorted_valid_numbers = sorted(list(valid_numbers))
    
    # Find all unique combinations of 4 valid numbers
    for combo in combinations_with_replacement(sorted_valid_numbers, 4):
        i, j, k, l = combo
        
        # Ensure the chord is within the equave
        if l / i > equave_ratio_float:
            continue
            
        # Ensure the components are coprime
        if gcd(gcd(gcd(i, j), k), l) != 1:
            continue

        # Ensure the odd limit of all intervals is within the limit_value (only for odd limit mode)
        if limit_mode == "odd":
            if (get_odd_limit(Fraction(j, i)) > limit_value or
                get_odd_limit(Fraction(k, j)) > limit_value or
                get_odd_limit(Fraction(l, k)) > limit_value):
                continue
            
        # Calculate interval cents
        c1 = cents(j / i)
        c2 = cents(k / j)
        c3 = cents(l / k)
        
        # Calculate simplicity
        if limit_mode == "odd":
            simplicity = max(get_odd_limit(Fraction(j, i)),
                             get_odd_limit(Fraction(k, j)),
                             get_odd_limit(Fraction(l, k)))
        else: # For integer limit, use sum of integers as simplicity
            simplicity = i + j + k + l
        
        # Format label
        label = f"{i}:{j}:{k}:{l}"
        
        labels_data.append(((c1, c2, c3), label, simplicity))
        
    return labels_data
