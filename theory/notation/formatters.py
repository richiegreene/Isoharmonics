from utils.constants import *

def printnom(nom):
    return chr((nom + 2) % 7 + 65)

def printupdown(ups):
    return "v" * -ups if ups < 0 else "^" * ups

def printliftdrop(lifts):
    return "\\" * -lifts if lifts < 0 else "/" * lifts

def printsharp(sharps, half):
    result = ""
    if half:
        if sharps % 2 != 0:
            result += HALF_SHARP
            sharps -= 1
        if sharps % 4 != 0:
            if result.endswith(HALF_SHARP):
                result = result[:-1] + THREE_HALF_SHARP
                sharps -= 2
            else:
                result += SHARP
                sharps -= 2
        while sharps > 0:
            result += DOUBLE_SHARP
            sharps -= 4
    else:
        if sharps % 2 != 0:
            result += SHARP
            sharps -= 1
        while sharps > 0:
            result += DOUBLE_SHARP
            sharps -= 2
    return result

def printflat(flats, half):
    result = ""
    if half:
        if flats % 2 != 0:
            result += HALF_FLAT
            flats -= 1
        if flats % 4 != 0:
            if result.endswith(HALF_FLAT):
                result = result[:-1] + THREE_HALF_FLAT
                flats -= 2
            else:
                result += FLAT
                flats -= 2
        while flats > 0:
            result += DOUBLE_FLAT
            flats -= 4
    else:
        flat_count = flats
        while flat_count >= 2:
            result += DOUBLE_FLAT
            flat_count -= 2
        if flat_count == 1:
            result += FLAT
    return result

def printnote(note, halves):
    if note.s_nom == note.f_nom:
        return printnom(note.s_nom)
    else:
        sharp_name = (
            printliftdrop(note.s_lifts)
            + printupdown(note.s_ups)
            + printnom(note.s_nom)
            + printsharp(note.sharps, halves)
        )
        flat_name = (
            printliftdrop(note.f_lifts)
            + printupdown(note.f_ups)
            + printnom(note.f_nom)
            + printflat(note.flats, halves)
        )
        return f"{sharp_name}, {flat_name}"
