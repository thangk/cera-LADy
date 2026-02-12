#!/usr/bin/env python3
import curses

def test_keys(stdscr):
    stdscr.clear()
    stdscr.addstr(0, 0, "Press keys (q to quit):")
    stdscr.addstr(1, 0, "Key codes will appear below:")
    row = 3
    
    while True:
        key = stdscr.getch()
        if key == ord('q'):
            break
        
        stdscr.addstr(row, 0, f"Key pressed: {key} ('{chr(key) if 32 <= key <= 126 else 'non-printable'}')")
        row += 1
        if row > 20:
            row = 3
            stdscr.clear()
            stdscr.addstr(0, 0, "Press keys (q to quit):")
            stdscr.addstr(1, 0, "Key codes will appear below:")
        stdscr.refresh()

if __name__ == "__main__":
    curses.wrapper(test_keys)