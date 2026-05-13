import numpy as np
from collections import namedtuple
from .symmetry_operation import (
    SymmetryOperation,
    expanded_symmetry_list,
    reduced_symmetry_list,
)
from cspy.crystal.sg_n_symops import DEFAULT_SG_N_SYMOPS, DMACRYS_SG_N_SYMOPS
from typing import List, Tuple
from typing_extensions import Literal
import os
import json
import logging

LOG = logging.getLogger(__name__)

_sgdata = namedtuple(
    "_sgdata",
    "number short schoenflies "
    "full international pointgroup "
    "choice centering symops inversion_at_origin",
)

with open(os.path.join(os.path.dirname(__file__), "sgdata.json")) as f:
    _sgdata_dict = json.load(f)

SG_FROM_NUMBER = {k: [_sgdata._make(x) for x in v] for k, v in _sgdata_dict.items()}
SG_DEFAULT_SETTING_CHOICE = {
    48: "2",
    50: "2",
    59: "2",
    68: "2",
    70: "2",
    85: "2",
    88: "2",
    86: "2",
    125: "2",
    126: "2",
    129: "2",
    130: "2",
    133: "2",
    134: "2",
    137: "2",
    138: "2",
    141: "2",
    142: "2",
    201: "2",
    203: "2",
    222: "2",
    224: "2",
    227: "2",
    228: "2",
}
SG_FROM_SYMOPS = {tuple(x.symops): x for k, sgs in SG_FROM_NUMBER.items() for x in sgs}
SG_CHOICES = {int(k): [x.choice for x in v] for k, v in SG_FROM_NUMBER.items()}
LATTICE_VARIABLES = {
    "triclinic": ("a", "b", "c", "alpha", "beta", "gamma"),
    "monoclinic": ("a", "b", "c", 90, "beta", 90),
    "orthorhombic": ("a", "b", "c", 90, 90, 90),
    "tetragonal": ("a", "a", "c", 90, 90, 90),
    "rhombohedral": ("a", "a", "a", "alpha", "alpha", "alpha"),
    "hexagonal": ("a", "a", "c", 90, 90, 120),
    "cubic": ("a", "a", "a", 90, 90, 90),
}


def n_symops_from_SG(SG: int, context: Literal["default", "DMACRYS"]) -> int:
    """Take the space group as an integer, and return the
    number of symmetry operators for the default setting in a given context.


    Parameters
    ----------
    SG : int
        Space group number

    context : string
        For which software should we find the default setting for?
        Supported: default, DMACRYS

    Returns
    -------
    n_symops : int
        The number of symmetry operators for that space group in the default
        setting for the context provided

    """

    supported_contexts = ["default", "DMACRYS"]
    if not context in supported_contexts:
        LOG.exception("Context %s not supported.", context)

    if context == "DMACRYS":
        if SG in DMACRYS_SG_N_SYMOPS.keys():
            n_symops = DMACRYS_SG_N_SYMOPS[SG]
        else:
            n_symops = DEFAULT_SG_N_SYMOPS[SG]
            LOG.info(
                "Space group %s missing from DMACRYS_SG_N_SYMOPS. Taking from cspy default instead.",
                SG,
            )
    else:
        n_symops = DEFAULT_SG_N_SYMOPS[SG]

    return n_symops


class SpaceGroup:
    """Represent a crystallographic space group, including
    all necessary symmetry operations in fractional coordinates,
    the international tables number from 1-230, and the international
    tables symbol.
    """

    international_tables_number: int
    symbol: str
    symmetry_operations: List[SymmetryOperation]

    def __init__(self, international_tables_number: int, choice=""):
        if international_tables_number < 1 or international_tables_number > 230:
            raise ValueError("Space group number must be between [1, 230]")
        self.international_tables_number = international_tables_number
        if not choice and international_tables_number in SG_DEFAULT_SETTING_CHOICE:
            choice = SG_DEFAULT_SETTING_CHOICE[international_tables_number]
        if not choice:
            sgdata = SG_FROM_NUMBER[str(international_tables_number)][0]
        else:
            candidates = SG_FROM_NUMBER[str(international_tables_number)]
            for candidate in candidates:
                if choice == candidate.choice:
                    sgdata = candidate
                    break
            else:
                raise ValueError("Could not find choice {}".format(choice))
        self.symbol = sgdata.short
        self.full_symbol = sgdata.full
        self.choice = sgdata.choice
        self.centering = sgdata.centering
        self.point_group = sgdata.pointgroup
        self.schoenflies = sgdata.schoenflies
        self.inversion_at_origin = sgdata.inversion_at_origin
        symops = sgdata.symops
        self.symmetry_operations = [
            SymmetryOperation.from_integer_code(s) for s in symops
        ]

    @property
    def cif_section(self) -> str:
        """Representation of the SpaceGroup in CIF files"""
        return "\n".join(
            "{} {}".format(i, sym.cif_form)
            for i, sym in enumerate(self.symmetry_operations, start=1)
        )

    @property
    def crystal_system(self):
        sg = self.international_tables_number
        if sg <= 0 or sg >= 231:
            raise ValueError("International spacegroup number must be between 1-230")
        if sg <= 2:
            return "triclinic"
        if sg <= 15:
            return "monoclinic"
        if sg <= 74:
            return "orthorhombic"
        if sg <= 142:
            return "tetragonal"
        if sg <= 167:
            return "trigonal"
        if sg <= 194:
            return "hexagonal"
        return "cubic"

    @property
    def lattice_type(self):
        inum = self.international_tables_number
        if inum < 143 or inum > 194:
            return self.crystal_system
        if inum in (146, 148, 155, 160, 161, 166, 167):
            if self.choice == "H":
                return "hexagonal"
            elif self.choice == "R":
                return "rhombohedral"
        else:
            return "hexagonal"

    @property
    def lattice_variables(self):
        return LATTICE_VARIABLES[self.lattice_type]

    @property
    def latt(self) -> int:
        """
        >>> P1 = SpaceGroup(1)
        >>> P21c = SpaceGroup(14)
        >>> I41 = SpaceGroup(14)
        >>> R3bar = SpaceGroup(148)
        >>> P1.latt
        -1
        >>> P21c.latt
        1
        >>> R3bar.latt
        3
        """
        centering_to_latt = {
            "primitive": 1,  # P
            "body": 2,  # I
            "rcenter": 3,  # R
            "face": 4,  # F
            "aface": 5,  # A
            "bface": 6,  # B
            "cface": 7,  # C
        }
        if not self.inversion_at_origin:
            return -centering_to_latt[self.centering]
        return centering_to_latt[self.centering]

    def __len__(self):
        return len(self.symmetry_operations)

    def ordered_symmetry_operations(self):
        # make sure we do the unit symop first
        unity = 0
        for i, s in enumerate(self.symmetry_operations):
            if s.is_identity():
                unity = i
                break
        else:
            return self.symmetry_operations
        other_symops = (
            self.symmetry_operations[:unity] + self.symmetry_operations[unity + 1 :]
        )
        return [self.symmetry_operations[unity]] + other_symops

    def has_hexagonal_rhombohedral_choices(self):
        return self.international_tables_number in (146, 148, 155, 160, 161, 166, 167)

    def apply_all_symops(
        self, coordinates: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """For a given set of coordinates, apply all symmetry
        operations in this space group, yielding a set subject
        to only translational symmetry (i.e. a unit cell).
        Assumes the input coordinates are fractional."""
        nsites = len(coordinates)
        transformed = np.empty((nsites * len(self), 3))
        generator_symop = np.empty(nsites * len(self), dtype=np.int32)

        # make sure we do the unit symop first
        unity = 0
        for i, s in enumerate(self.symmetry_operations):
            if s.integer_code == 16484:
                unity = i
                break
        transformed[0:nsites] = coordinates
        generator_symop[0:nsites] = 16484
        other_symops = (
            self.symmetry_operations[:unity] + self.symmetry_operations[unity + 1 :]
        )
        for i, s in enumerate(other_symops, start=1):
            transformed[i * nsites : (i + 1) * nsites] = s(coordinates)
            generator_symop[i * nsites : (i + 1) * nsites] = s.integer_code
        return generator_symop, transformed

    def __repr__(self):
        return "{} {}: {}".format(
            self.__class__.__name__, self.international_tables_number, self.symbol
        )

    def __eq__(self, other):
        return (
            self.international_tables_number == other.international_tables_number
        ) and (self.choice == other.choice)

    def __hash__(self):
        return hash((self.international_tables_number, self.choice))

    def reduced_symmetry_operations(self):
        return reduced_symmetry_list(self.symmetry_operations, self.latt)

    def subgroups(self):
        """This definition of subgroups is probably not correct"""
        symop_ids = set([x.integer_code for x in self.symmetry_operations])
        subgroups = []
        for symops in SG_FROM_SYMOPS:
            if symop_ids.issuperset(symops):
                sgdata = SG_FROM_SYMOPS[symops]
                sg = SpaceGroup(sgdata.number, choice=sgdata.choice)
                if sg != self:
                    subgroups.append(sg)
        return subgroups

    def gulp_full_symbol(self) -> str:
        """Return full symbol in a format that is compatible with GULP"""
        tokens = []
        s = self.full_symbol.upper()
        if "=" in s:
            s = s.split("=")[-1]
        s = iter(s)
        for ch in s:
            if ch == "_":
                tokens[-1] += next(s)
            elif ch == "/":
                tokens[-1] += ch + next(s)
            elif ch == "-":
                tokens.append(ch + next(s))
            else:
                tokens.append(ch)
        return " ".join(tokens)

    @classmethod
    def from_symmetry_operations(
        cls, symops: List[SymmetryOperation], expand_latt: int = None
    ) -> "SpaceGroup":
        """Find a matching spacegroup for a given set of symmetry
        operations, optionally treating them as a reduced set of
        symmetry operations and expanding them based on the lattice
        type."""
        if expand_latt is not None:
            if not -8 < expand_latt < 8:
                raise ValueError("expand_latt must be between [-7, 7]")
            symops = expanded_symmetry_list(symops, expand_latt)
        encoded = tuple(sorted(s.integer_code for s in symops))
        if encoded not in SG_FROM_SYMOPS:
            raise ValueError(
                "Could not find matching spacegroup for "
                "the following symops:\n{}".format(
                    "\n".join(s.shelx_form for s in sorted(symops))
                )
            )
        else:
            sgdata = SG_FROM_SYMOPS[encoded]
            return SpaceGroup(sgdata.number, choice=sgdata.choice)
