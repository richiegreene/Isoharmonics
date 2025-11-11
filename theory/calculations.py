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

def generate_ji_triads(odd_limit, equave=Fraction(2,1)):
    if odd_limit < 1:
        return []

    # 1. Generate all valid intervals
    valid_intervals = set([Fraction(1,1)])
    odds = [i for i in range(1, odd_limit + 1) if i % 2 != 0]
    for n in odds:
        for d in odds:
            if n == d: continue
            ratio = Fraction(n, d)
            
            # generate octave variations up to equave
            temp_r = ratio
            while temp_r <= equave:
                if temp_r >= 1:
                    valid_intervals.add(temp_r)
                temp_r *= 2
            
            temp_r = ratio / 2
            while temp_r >= 1:
                valid_intervals.add(temp_r)
                temp_r /= 2

    if get_odd_limit(equave) <= odd_limit:
        valid_intervals.add(equave)

    sorted_intervals = sorted(list(valid_intervals))

    triads = []
    triad_labels = set()

    # 2. Form triads from intervals
    for i in range(len(sorted_intervals)):
        r1 = sorted_intervals[i]
        for j in range(i, len(sorted_intervals)):
            r2 = sorted_intervals[j]
            
            r3 = r2 / r1
            if get_odd_limit(r3) <= odd_limit:
                cx_ratio = r1
                cy_ratio = r3

                if cx_ratio < 1 or cy_ratio < 1: continue

                cx = 1200 * math.log2(cx_ratio)
                cy = 1200 * math.log2(cy_ratio)

                if cx + cy > 1200 * math.log2(equave) + 1e-9: continue

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