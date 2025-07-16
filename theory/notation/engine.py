import math
from theory.notation.note import Note
from theory.notation.formatters import printnote

def fifth(edo):
    return round(math.log(1.5) / math.log(2) * edo)

def majsec(edo, p5):
    return (p5 * 2) - edo

def apotome(edo, p5):
    if (edo % 7 == 0 or edo % 5 == 0) and edo < 36:
        return edo
    return (p5 * 7) - (edo * 4)

def updown(edo, p5):
    if edo < 66:
        return 1
    if edo == 129:
        return 3
    y3 = round(math.log(1.25) / math.log(2) * edo)
    sc = (p5 * 4) - (edo * 2) - y3
    return 1 if sc == 0 else sc

def verysharp(edo, p5):
    return p5 / edo > 0.6 or (edo < 35 and edo % 5 == 0)

def halfacc(a1):
    return a1 % 2 == 0

def basicnotes(notes, edo, p5, p2, penta):
    notes[0].s_nom = 0
    notes[p2].s_nom = 1
    notes[edo - p5].s_nom = 3
    notes[p5].s_nom = 4
    notes[p5 + p2].s_nom = 5
    notes[0].f_nom = 0
    notes[p2].f_nom = 1
    notes[edo - p5].f_nom = 3
    notes[p5].f_nom = 4
    notes[p5 + p2].f_nom = 5
    if not penta:
        notes[2 * p2].s_nom = 2
        notes[p5 + (2 * p2)].s_nom = 6
        notes[2 * p2].f_nom = 2
        notes[p5 + (2 * p2)].f_nom = 6

def trround(x):
    if x - math.floor(x) == 0.5:
        return math.floor(x) if x > 0 else math.ceil(x)
    return round(x)

def setsharpcounts(x, ap, ud, apc, udc, ldc, ra, ru):
    apc[0] = x / ap
    ra[0] = trround(apc[0])
    udc[0] = ((ra[0] * ap) - x) / ud
    ru[0] = trround(udc[0])
    ldc[0] = (ru[0] * ud) - ((ra[0] * ap) - x)
    ru[0] *= -1

def setsharpnotes(note, nom, r_ap, r_ud, ldcount):
    note.s_nom = nom
    note.sharps = r_ap
    note.s_ups = r_ud
    note.s_lifts = ldcount

def sharpnotes(notes, edo, p5, p2, ap, ud, penta):
    apcount = [0]
    udcount = [0]
    r_ap = [0]
    r_ud = [0]
    ldcount = [0]
    for i in range(1, p2):
        setsharpcounts(i, ap, ud, apcount, udcount, ldcount, r_ap, r_ud)
        setsharpnotes(notes[i], 0, r_ap[0], r_ud[0], ldcount[0])
        setsharpnotes(notes[edo - p5 + i], 3, r_ap[0], r_ud[0], ldcount[0])
        setsharpnotes(notes[p5 + i], 4, r_ap[0], r_ud[0], ldcount[0])
    if penta:
        for i in range(1, (edo - p5) - p2):
            setsharpcounts(i, ap, ud, apcount, udcount, ldcount, r_ap, r_ud)
            setsharpnotes(notes[p2 + i], 1, r_ap[0], r_ud[0], ldcount[0])
            setsharpnotes(notes[p5 + p2 + i], 5, r_ap[0], r_ud[0], ldcount[0])
    else:
        for i in range(1, p2):
            setsharpcounts(i, ap, ud, apcount, udcount, ldcount, r_ap, r_ud)
            setsharpnotes(notes[p2 + i], 1, r_ap[0], r_ud[0], ldcount[0])
            setsharpnotes(notes[p5 + p2 + i], 5, r_ap[0], r_ud[0], ldcount[0])
        for i in range(1, (edo - p5) - (p2 * 2)):
            setsharpcounts(i, ap, ud, apcount, udcount, ldcount, r_ap, r_ud)
            setsharpnotes(notes[(p2 * 2) + i], 2, r_ap[0], r_ud[0], ldcount[0])
            setsharpnotes(notes[p5 + (p2 * 2) + i], 6, r_ap[0], r_ud[0], ldcount[0])

def setflatcounts(x, ap, ud, apc, udc, ldc, ra, ru):
    apc[0] = x / ap
    ra[0] = trround(apc[0])
    udc[0] = ((ra[0] * ap) - x) / ud
    ru[0] = trround(udc[0])
    ldc[0] = ((ra[0] * ap) - x) - (ru[0] * ud)

def setflatnotes(note, nom, r_ap, r_ud, ldcount):
    note.f_nom = nom
    note.flats = r_ap
    note.f_ups = r_ud
    note.f_lifts = ldcount

def flatnotes(notes, edo, p5, p2, ap, ud, penta):
    apcount = [0]
    udcount = [0]
    r_ap = [0]
    r_ud = [0]
    ldcount = [0]
    for i in range(1, p2):
        setflatcounts(i, ap, ud, apcount, udcount, ldcount, r_ap, r_ud)
        setflatnotes(notes[p2 - i], 1, r_ap[0], r_ud[0], ldcount[0])
        setflatnotes(notes[p5 - i], 4, r_ap[0], r_ud[0], ldcount[0])
        setflatnotes(notes[p5 + p2 - i], 5, r_ap[0], r_ud[0], ldcount[0])
    if penta:
        for i in range(1, (edo - p5) - p2):
            setflatcounts(i, ap, ud, apcount, udcount, ldcount, r_ap, r_ud)
            setflatnotes(notes[edo - i], 0, r_ap[0], r_ud[0], ldcount[0])
            setflatnotes(notes[edo - p5 - i], 3, r_ap[0], r_ud[0], ldcount[0])
    else:
        for i in range(1, (edo - p5) - (p2 * 2)):
            setflatcounts(i, ap, ud, apcount, udcount, ldcount, r_ap, r_ud)
            setflatnotes(notes[edo - i], 0, r_ap[0], r_ud[0], ldcount[0])
            setflatnotes(notes[edo - p5 - i], 3, r_ap[0], r_ud[0], ldcount[0])
        for i in range(1, p2):
            setflatcounts(i, ap, ud, apcount, udcount, ldcount, r_ap, r_ud)
            setflatnotes(notes[(p2 * 2) - i], 2, r_ap[0], r_ud[0], ldcount[0])
            setflatnotes(notes[p5 + (p2 * 2) - i], 6, r_ap[0], r_ud[0], ldcount[0])

def calculate_single_note(n, m):
    if m < 7 and m != 5:
        return "n/a"
    p5 = fifth(m)
    p2 = majsec(m, p5)
    a1 = apotome(m, p5)
    ud = updown(m, p5)
    penta = verysharp(m, p5)
    halves = halfacc(a1)
    if halves:
        a1 //= 2
    notes = [Note() for _ in range(m)]
    basicnotes(notes, m, p5, p2, penta)
    sharpnotes(notes, m, p5, p2, a1, ud, penta)
    flatnotes(notes, m, p5, p2, a1, ud, penta)
    n = n % m
    return printnote(notes[n], halves)
