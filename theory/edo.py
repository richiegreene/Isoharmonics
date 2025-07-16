from theory.calculations import calculate_12edo_step
from utils.formatters import to_subscript

def assign_12edo_notation(cents):
    step, error = calculate_12edo_step(cents)
    step = step % 12
    octave = 4 + (cents // 1200)
    notation_map = {
        0: "C",
        1: "C\uE262",
        2: "D",
        3: "E\uE260",
        4: "E",
        5: "F",
        6: "F\uE262",
        7: "G",
        8: "A\uE260",
        9: "A",
        10: "B\uE260",
        11: "B",
    }
    note_name = notation_map.get(step, f"Step {step}")
    inverted_error = -error
    if inverted_error == 0.0:
        return f"{note_name}{to_subscript(int(octave))}"
    else:
        rounded_error = round(inverted_error)
        error_str = f"+{rounded_error}" if rounded_error > 0 else f"-{abs(rounded_error)}"
        return f"{note_name}{to_subscript(int(octave))} {error_str}"
