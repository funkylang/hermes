#!/usr/bin/env python3
"""layout_test.py — pyterm window layout demonstration (hermes style).

Layout (for 80x24 terminal):
  - Rows 1-22: Output window (scroll region, content scrolls upward)
  - Row 23:    Input header with hermes-style rule: ───── Input ────────────
  - Row 24:    Status bar text

Controls:
  q = quit
  u = scroll up in output window
  d = scroll down in output window
"""
import sys
import os
import time
import termios
import tty

sys.path.insert(0, "/home/agents/workspace/pyterm/repo")

from pyterm import Terminal


INPUT_HEIGHT = 2  # rows for input + status bar at the bottom
OUTPUT_LINES = 80  # number of words to display (will definitely scroll)


def read_key():
    """Read a single key from stdin in raw mode."""
    term_fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(term_fd)
    tty.setraw(term_fd)
    ch = sys.stdin.read(1)
    termios.tcsetattr(term_fd, termios.TCSADRAIN, old_attrs)
    return ch


def draw_hermes_input_header(term, height, width):
    """Draw the hermes-style input header rule: ───── Input ────────────"""
    input_row = height - INPUT_HEIGHT + 1

    # Format: {symbol×5} Input {fill with symbols}
    prefix_symbols = "─" * 5
    label = " Input "
    # Fill with remaining symbols to reach width
    filler_count = max(width - 5 - len(label), 0)
    line = prefix_symbols + label + "─" * filler_count

    # Write with white text on blue background like hermes INPUT kind
    term.put("", 1, input_row)  # clear the row first
    term.set_draw_colour((255, 255, 255))       # white text
    term.set_clear_colour((18, 43, 66))         # blue background
    term.print(line)
    term.reset_colours()


def draw_input_area(term, height, width):
    """Draw the input and status bar area."""
    # Draw hermes-style input header
    draw_hermes_input_header(term, height, width)

    # Status bar
    status_row = height
    term.put("status bar", 1, status_row)


def main():
    term = Terminal()
    height = term.height
    width = term.width
    output_top = 1
    output_bottom = height - INPUT_HEIGHT

    print(f"Layout: {width}x{height}", file=sys.stderr)
    print(f"  Output rows: 1-{output_bottom}", file=sys.stderr)
    print(f"  Input row: {output_bottom + 1} (hermes-style header)", file=sys.stderr)
    print(f"  Status row: {output_bottom + 2} (= {height})", file=sys.stderr)
    print(f"\nControls: q=quit, u=scroll up, d=scroll down", file=sys.stderr)

    term.clear_screen()

    # Create output window with its own scroll region
    out_win = term.window(output_top, output_bottom)
    out_win.clear()

    # Draw initial input area
    draw_input_area(term, height, width)

    print(f"\nWriting {OUTPUT_LINES} words (one per line). Watch rows 1-{output_bottom} scroll.", file=sys.stderr)
    time.sleep(1)

    # Write words one at a time to demonstrate scrolling
    for i in range(1, OUTPUT_LINES + 1):
        out_win.println(f"word_{i}")

    # Redraw input area (ensure it's still there after scrolling)
    draw_input_area(term, height, width)

    print(f"\nDone. The last ~{output_bottom} words should be visible in the output area.", file=sys.stderr)
    print(f"(Words 1-{OUTPUT_LINES - output_bottom + 1} scrolled off the top).", file=sys.stderr)
    print(f"Use 'u' and 'd' to scroll through the content.\n", file=sys.stderr)

    # Main loop: handle keyboard input for scrolling
    while True:
        ch = read_key()

        if ch.lower() == 'q':
            print("\nQuitting...", file=sys.stderr)
            break

        elif ch.lower() == 'u':
            # Scroll up (view earlier content)
            term.scroll_up(1)
            draw_input_area(term, height, width)  # redraw input area after scroll

        elif ch.lower() == 'd':
            # Scroll down (view later content)
            term.scroll_down(1)
            draw_input_area(term, height, width)  # redraw input area after scroll

    # Clean up terminal state before exiting (remove scroll regions, reset colours)
    term.cleanup()
    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()
