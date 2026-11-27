#!/usr/bin/env python3
"""colour_gradients.py — truecolour smoke test for the new frontend.

Seven horizontal gradient bars, full terminal width:
  1..6: black -> base colour -> white   (RED, GREEN, BLUE, YELLOW, CYAN, MAGENTA)
  7:    black -> white (neutral)

Smooth ramps across ~80+ columns require true 24-bit colour; any banding or
quantisation is visible immediately. Verifies pyterm's set_clear_colour and
the target terminal's SGR 48;2 handling.

Run: python3 colour_gradients.py   (holds the screen for 5 s, then exits)
"""
import sys
import time

sys.path.insert(0, "/home/agents/workspace/pyterm/repo")

from pyterm import Terminal, RED, GREEN, BLUE, YELLOW, CYAN, MAGENTA


def lerp(c1, c2, t):
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c1, c2))


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


def bar_colour(base, t):
    """Colour at position t (0..1) across one gradient bar.

    base=None: straight black -> white ramp.
    base set:  black -> base for t in [0, 0.5], base -> white for [0.5, 1].
    """
    if base is None:
        return lerp(BLACK, WHITE, t)
    if t <= 0.5:
        return lerp(BLACK, base, t * 2)
    return lerp(base, WHITE, (t - 0.5) * 2)


def draw_bar(term, row, width, base):
    """Fill one screen row with a gradient; uses pyterm public API throughout."""
    term.put("", 1, row)                      # cursor to row start
    for c in range(1, width + 1):
        rgb = bar_colour(base, (c - 1) / max(width - 1, 1))
        with term.lock:                        # one atomic cell: colour + glyph
            term.set_clear_colour(rgb)
            term.print(" ")


def main():
    term = Terminal()

    bars = [
        ("red", RED),
        ("green", GREEN),
        ("blue", BLUE),
        ("yellow", YELLOW),
        ("cyan", CYAN),
        ("magenta", MAGENTA),
        ("black-white", None),
    ]

    term.clear_screen()
    width = term.width

    for r, (label, base) in enumerate(bars):
        draw_bar(term, r + 1, width, base)

    print("%d gradient bars (%d cols); screen held 5s" % (len(bars), width),
          file=sys.stderr)
    time.sleep(5)


if __name__ == "__main__":
    main()
