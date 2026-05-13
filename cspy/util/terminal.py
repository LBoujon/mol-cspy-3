import os
import re
import sys

PY3 = sys.version_info[0] > 3
DISABLE_COLOR = False

if "get_ipython" in dir():
    DISABLE_COLOR = True


class TerminalColors:
    _CONTROLS = {
        "reset": "\033[0m",
        "bold": "\033[01m",
        "disable": "\033[02m",
        "underline": "\033[04m",
        "reverse": "\033[07m",
        "invisible": "\033[08m",
        "strikethrough": "\033[09m",
    }

    _FG = {
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "orange": "\033[33m",
        "blue": "\033[34m",
        "purple": "\033[35m",
        "cyan": "\033[36m",
        "lightgrey": "\033[37m",
        "darkgrey": "\033[90m",
        "lightred": "\033[91m",
        "lightgreen": "\033[92m",
        "yellow": "\033[93m",
        "lightblue": "\033[94m",
        "pink": "\033[95m",
        "lightcyan": "\033[96m",
    }

    _BG = {
        "black": "\033[40m",
        "red": "\033[41m",
        "green": "\033[42m",
        "orange": "\033[43m",
        "blue": "\033[44m",
        "purple": "\033[45m",
        "cyan": "\033[46m",
        "lightgrey": "\033[47m",
    }

    @staticmethod
    def format(string, color="", style="", fmt=""):
        if color == "white":
            color = ""
        s = "{{style}}{{color}}{{string:{fmt}}}{{reset}}".format(fmt=fmt if fmt else "")
        s = s.format(
            style=TerminalColors._CONTROLS[style] if style else "",
            color=TerminalColors._FG[color] if color else "",
            string=string,
            reset=TerminalColors._CONTROLS["reset"] if color or style else "",
        )
        return s


def passfail(msg, result):
    tc = TerminalColors
    res = tc.format(
        "Y" if result else "N",
        color="lightgreen" if result else "lightred",
        style="bold",
        fmt="^3s",
    )
    return res + tc.format(msg, color="white", style="", fmt="<42s")


def print_format_table():
    for style in range(8):
        for fg in range(30, 38):
            s1 = ""
            for bg in range(40, 48):
                fmt = ";".join((str(style), str(fg), str(bg)))
                s1 += "\x1b[{}m {} \x1b[0m".format(fmt, fmt)
            print(s1)
        print("\n")


if __name__ == "__main__":
    print()
    print("Results")
    print(passfail("Terminal colors", True))
    print(passfail("Bland output", False))
    print()
