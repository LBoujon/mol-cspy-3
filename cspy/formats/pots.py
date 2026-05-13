import re
from collections import defaultdict
from cspy.util.path import Path

BUCK_REGEX = r"(BUCK)\s+(\w+)\s+(\w+).*$\s^\s*(.*)$"
DBUC_REGEX = r"(DBUC)\s+(\w+)\s+(\w+).*$\s^\s*(.*)$"
ANIS_REGEX = r"(ANIS)\s+(\w+)\s+(\w+).*$\s^\s*(.*)$\s^\s*(.*)$"


def parse_pots(filename):
    anis_fields = (int, int, int, int, int, float)
    contents = Path(filename).read_text()
    # check if pot is damped by looking for DBUC in first line
    if 'DBUC' in contents.split('\n')[0]:
        dbuc = True
    else:
        dbuc = False
    buck = defaultdict(dict)
    anis = defaultdict(dict)

    if dbuc:
        buck_matches = re.findall(DBUC_REGEX, contents, re.MULTILINE)
    else:
        buck_matches = re.findall(BUCK_REGEX, contents, re.MULTILINE)
    for kind, a, b, params in buck_matches:
        buck[a][b] = [float(x) for x in params.split()]

    anis_matches = re.findall(ANIS_REGEX, contents, re.MULTILINE)
    for kind, a, b, l1, l2 in anis_matches:
        anis[a][b] = [
            [t(x) for t, x in zip(anis_fields, l.split())]
            for l in (l1, l2)
            if "ENDS" not in l
        ]

    return buck, anis, dbuc
