#!/usr/bin/env python3
import curses

def debug_main(stdscr):
    marked_for_kill = False
    
    while True:
        stdscr.clear()
        stdscr.addstr(0, 0, f"Marked for kill: {marked_for_kill}")
        stdscr.addstr(1, 0, "Press 'k' to mark, ENTER to confirm, other keys to cancel, 'q' to quit")
        stdscr.refresh()
        
        key = stdscr.getch()
        
        if key == ord('q'):
            break
        elif key == ord('k'):
            marked_for_kill = True
        elif key == ord('\n') and marked_for_kill:
            stdscr.addstr(3, 0, "KILL CONFIRMED!")
            stdscr.refresh()
            curses.napms(1000)
            marked_for_kill = False
        elif marked_for_kill:
            marked_for_kill = False

if __name__ == "__main__":
    curses.wrapper(debug_main)