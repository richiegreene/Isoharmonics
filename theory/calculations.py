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

def get_odd_limit(ratio):
    """Calculates the odd limit of a given ratio."""
    try:
        ratio = Fraction(ratio).limit_denominator(10000)
        n, d = ratio.numerator, ratio.denominator
        
        while n > 0 and n % 2 == 0:
            n //= 2
        while d > 0 and d % 2 == 0:
            d //= 2
            
        return max(n, d)
    except (ValueError, ZeroDivisionError):
        return 1

def generate_ji_triads(odd_limit):
    if odd_limit < 1:
        return []

    # 1. Generate all valid intervals in [1, 2]
    valid_intervals = set()
    odds = [i for i in range(1, odd_limit + 1) if i % 2 != 0]
    for n in odds:
        for d in odds:
            ratio = Fraction(n, d)
            while ratio > 2:
                ratio /= 2
            while ratio < 1:
                ratio *= 2
            if get_odd_limit(ratio) <= odd_limit:
                 valid_intervals.add(ratio)
    
    if get_odd_limit(1) <= odd_limit:
        valid_intervals.add(Fraction(1,1))
    if get_odd_limit(2) <= odd_limit:
        valid_intervals.add(Fraction(2,1))

    sorted_intervals = sorted(list(valid_intervals))

    triads = []
    triad_labels = set()

    # 2. Form triads from intervals
    for i in range(len(sorted_intervals)):
        r1 = sorted_intervals[i]
        for j in range(i, len(sorted_intervals)):
            r2 = sorted_intervals[j]
            # Triad notes are 1, r1, r2 (as intervals from the root)
            # We need to check the third interval r2/r1
            
            r3 = r2 / r1
            if get_odd_limit(r3) <= odd_limit:
                # This is a valid triad with intervals from root R1=r1, R2=r2
                # The intervals between adjacent notes are cx_ratio = r1, cy_ratio = r2/r1
                cx_ratio = r1
                cy_ratio = r3

                cx = 1200 * math.log2(cx_ratio)
                cy = 1200 * math.log2(cy_ratio)

                # Generate label 1 : r1 : r2
                common_denom = r1.denominator * r2.denominator
                a = common_denom
                b = r1.numerator * r2.denominator
                c = r2.numerator * r1.denominator
                
                common_divisor = gcd(gcd(a,b),c)
                sa, sb, sc = a//common_divisor, b//common_divisor, c//common_divisor
                
                sorted_triad = sorted([sa, sb, sc])
                label = f"{sorted_triad[0]}:{sorted_triad[1]}:{sorted_triad[2]}"

                if label not in triad_labels:
                    triads.append(((cx, cy), label))
                    triad_labels.add(label)

    return triads