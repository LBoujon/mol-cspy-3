import logging
from pathlib import Path
from cspy.chem import Element
from collections import namedtuple
import numpy as np

ZMatEntry = namedtuple("ZMatEntry", "element a r b angle c dihedral")

import re

EXYZ_FIELD_REGEX = re.compile(r"""(\w+)=(".*?"|\S+)""")
LOG = logging.getLogger(__name__)


def parse_zmat_string(contents, filename="string"):
    contents = contents.splitlines()
    zmat = []
    parsers = (str, int, float, int, float, int, float)
    v = {x: None for x in ZMatEntry._fields}
    for i, line in enumerate(contents):
        tokens = line.split()
        v = {x: None for x in ZMatEntry._fields}
        for k, f, parser in zip(ZMatEntry._fields, tokens, parsers):
            v[k] = parser(f)
        zmat.append(ZMatEntry(**v))
    return zmat


def parse_zmat_file(filename, **kwargs):
    contents = Path(filename).read_text()
    return parse_zmat_string(contents, filename=filename, **kwargs)


def to_zmat(mol):
    """Iterate through a molecule in depth first order, creating
    lines for a Z-matrix entry in the style of Gaussian.
    """
    seq, pred = mol.dfs_order()
    zmat = []
    els = mol.elements
    idx_of = {s: i for i, s in enumerate(seq)}
    # -9999 is used as an undefined value
    idx_of[-9999] = -9999
    for n, idx in enumerate(seq):
        r_a = None
        angle = None
        dihedral = None
        pi = idx
        a1 = pred[idx] if n > 0 else -9999
        a2 = -9999
        a3 = -9999
        if a1 > -1:
            r_a = np.linalg.norm(mol.positions[a1, :] - mol.positions[pi, :])
            na = mol.bonds[a1, :].nonzero()[1]
            if n > 1:
                a2 = na[~np.isin(na, [pi, a1, a2, a3])][0]
        if a2 > -1:
            angle = mol.bond_angle([pi, a1, a2])
            nb = mol.bonds[a2, :].nonzero()[1]
            if n > 2:
                a3 = nb[~np.isin(nb, [pi, a1, a2, a3])][0]
        if a3 > -1:
            dihedral = mol.dihedral_angle([pi, a1, a2, a3])

        z = ZMatEntry(
            els[pi],
            idx_of[a1] + 1,
            r_a,
            idx_of[a2] + 1,
            angle,
            idx_of[a3] + 1,
            dihedral,
        )
        zmat.append(z)
    return zmat


def to_zmat_string(mol):
    zmat = to_zmat(mol)
    lines = []
    for el, a, r_a, b, ab, c, abc in zmat:
        line = f"{el}"
        if a > 0:
            line += f" {a:3} {r_a:10.5f}"
        if b > 0:
            line += f" {b:3} {ab:10.5f}"
        if c > 0:
            line += f" {c:3} {abc:10.5f}"
        lines.append(line)
    return "\n".join(lines)
