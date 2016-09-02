#! /usr/bin/env python

import curses
import curses.ascii

def my_term(stdscr, my_var):
    curses.start_color()

    if curses.has_colors():
        stdscr.addstr(0, 0, 'Colors!')
        bg = curses.COLOR_BLACK
        curses.init_pair(1, curses.COLOR_RED, bg)
    else:
        stdscr.addstr(0, 0, 'No Colors :(')

    exit = False
    height, width = stdscr.getmaxyx()

    stdscr.addstr(1, 0, my_var, curses.color_pair(1))
    stdscr.move(height - 1, 0)
    stdscr.refresh()

    myplace = 2

    typing = ''
    while not exit:
        ch = stdscr.getch()
        if ch == curses.KEY_MOUSE:
            pass
        elif typing != '' and (ch == curses.KEY_ENTER or ch == 10):
            stdscr.addstr(myplace, 0, typing)
            myplace = (myplace + 1) % height
            stdscr.move(height - 1, 0)
            for i in range(len(typing)):
                stdscr.addstr(' ')
            stdscr.move(height - 1, 0)
            typing = ''
            stdscr.refresh()
        elif curses.ascii.isascii(ch):
            key = curses.ascii.unctrl(ch)
            if typing == '' and (key == 'q' or key == 'Q'):
                return
            else:
                typing += key
                stdscr.addstr(key)
                stdscr.refresh()

test_var = 'HOWDY'
curses.wrapper(my_term, test_var)
