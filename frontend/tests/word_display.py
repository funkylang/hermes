#!/usr/bin/env python3
"""word_display.py — layout test: output area + input/menu area.

Layout:
  - Top: output window (rows 1 to H-2), words scroll within it
  - Bottom: fixed rows for input box ("Input") and status bar

On each keypress the next word from sample_text.txt is shown in the output
window. Press 'q' to quit.
"""
import sys
import os
import termios
import tty

sys.path.insert(0, "/home/agents/workspace/pyterm/repo")

from pyterm import Terminal


INPUT_HEIGHT = 2  # rows for input + status bar at the bottom


def read_input_char(term_fd):
    """Read one character from raw-mode stdin (blocking)."""
    return sys.stdin.read(1)


def draw_input_rows(term, height, prompt_text=""):
    """Draw the fixed bottom rows: input box + status bar.

    Uses put() for positioning — safe, doesn't trigger scrolling.
    """
    input_row = height - INPUT_HEIGHT + 1
    status_row = height

    # Input row: vertical line label + prompt text
    term.put("│ " + prompt_text, 1, input_row)

    # Status row
    term.put("status bar", 1, status_row)


def main():
    words_file = os.path.join(os.path.dirname(__file__), "sample_text.txt")
    if not os.path.exists(words_file):
        print(f"Error: {words_file} not found", file=sys.stderr)
        sys.exit(1)

    with open(words_file, 'r') as f:
        words = [w for w in f.read().split() if w.strip()]

    term = Terminal()
    height = term.height
    output_top = 1
    output_bottom = height - INPUT_HEIGHT

    # Clear screen and set up layout
    term.clear_screen()

    # Create the output window (top part of screen)
    out_win = term.window(output_top, output_bottom)
    out_win.clear()

    # Show initial state: blank output + input/status rows
    draw_input_rows(term, height)

    print(f"Displaying {len(words)} words. "
          f"Press keys to advance, 'q' to quit.")

    # Save terminal attributes and switch to raw mode for input
    term_fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(term_fd)
    tty.setraw(term_fd)

    word_idx = 0
    show_msg = False  # whether to re-show the info message next loop

    try:
        while word_idx < len(words):
            ch = read_input_char(term_fd)

            if ch in ('q', '\x03'):  # 'q' or Ctrl-C to quit
                print("\nQuitting.")
                break

            # Show next word in output window
            word = words[word_idx]
            out_win.println(word)
            word_idx += 1
            show_msg = True

        if word_idx >= len(words):
            print("\nAll words displayed. Press any key to quit.")
            read_input_char(term_fd)  # wait for a key then exit

    finally:
        # Restore terminal settings
        termios.tcsetattr(term_fd, termios.TCSADRAIN, old_attrs)


if __name__ == "__main__":
    main()
