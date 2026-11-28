# Frontend Layout Test Notes

## Design (confirmed with user 2026-11-28)

**Layout requirements:**
- Upper screen: Output area that scrolls into scrollback buffer
- Lower screen: Variable-sized input + menu area
  - Input box with vertical line (like current hermes CLI): "│ Input:"
  - Status bar: "status bar" (placeholder for now)

## Implementation (pyterm windows)

**For standard 80x24 terminal:**
- Output window: rows 1 to H-2 (scroll region, content scrolls upward within band)
- Input row: row H-1 (fixed at bottom area)
- Status row: row H (bottom-most row)

**Key pyterm features used:**
- `Terminal()` - manages device state, scroll regions, colors
- `window(top, bottom)` - creates scoped window with own scroll region
- Window.println() - writes text and scrolls within window's band only
- Terminal.put(text, col, row) - positions cursor to absolute location (safe, no scrolling)

**Why this works:**
When output window scrolls content, it only affects its defined scroll region
(rows 1 to H-2). The bottom rows (H-1 and H) are outside that region and remain
fixed while output scrolls above them.

## Test Files Created

1. **layout_test.py** - Non-interactive demo:
   - Writes 50 words automatically
   - Demonstrates scrolling in output area
   - Shows fixed bottom rows staying put
   - Run: `python3 layout_test.py`

2. **word_display.py** - Interactive version:
   - Reads words from sample_text.txt
   - Press any key to display next word in output window
   - Press 'q' to quit
   - Uses raw-mode terminal input (termios)
   - Run: `python3 word_display.py`

3. **sample_text.txt** - Test data with 60+ words for scrolling demo

## How to Test (visual verification needed)

Run these in your terminal to see the actual layout:

```bash
cd ~/workspace/hermes/repo/frontend/tests
python3 layout_test.py      # Auto-demonstrates layout and scrolling
# OR
python3 word_display.py     # Interactive: press keys to advance words
```

**What you should see:**
- Top 22 rows: words appear one at a time, older ones scroll off top
- Row 23: "│ Input:" indicator (always visible)
- Row 24: "status bar" text (always visible)
- Bottom 2 rows stay fixed while top area scrolls
