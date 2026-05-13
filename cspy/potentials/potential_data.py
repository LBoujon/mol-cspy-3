from collections import namedtuple
import numpy as np
from os.path import dirname, join
import itertools

KCAL_PER_EV = 23.060541945329334
EV_PER_KCAL = 0.0433641153087705

PminAtomicPotential = namedtuple("PminAtomicPotential", "A B C")
OrientPairPotential = namedtuple("OrientPairPotential", "alpha rho C6")
POTS_DIR = dirname(__file__)

WILLIAMS_HBOND_TERMS = {
    (x, y)
    for x in ("H_W2", "H_W3", "H_W4", "H_Wa")
    for y in ("N_W1", "N_W2", "N_W3", "N_W4", "O_W1", "O_W2", "O_Wa")
}


class PotentialData:
    r"""
    Values are stored in the form

    A_{ij} \exp{-\B_{ij} r_{ij}} + C_{ij} r_{ij}^-6

    expected by neighcrys/dmacrys, where the values we
    store are actually A, 1/B and C, whose units 
    A: eV, 1/B: Ang, and C: eV Ang^6
    whereas pmin uses the form:

    B_{ij} \exp{-\C_{ij} r_{ij}} + A_{ij} r_{ij}^-6

    where units are stored as

    A: \sqrt(kcal/mol)Ang^6, B: Ang and C: \sqrt(kcal/mol)

    These Aij are actually from combining rules:
    A_{ij} = \sqrt{A_i A_j}
    B_{ij} = 0.5 * (B_i + B_j)
    C_{ij} = \sqrt{C_i C_j}

    """
    default_type = {"F": "fit", "W": "w99"}
    cross_terms = {
        "w99rev_pcm_6311": WILLIAMS_HBOND_TERMS,
        "w99rev_6311": WILLIAMS_HBOND_TERMS,
        "w99rev_6311_s": WILLIAMS_HBOND_TERMS,
        "w99rev_631": WILLIAMS_HBOND_TERMS,
    }

    def __init__(self, potential_name):
        potential_name = self.default_type.get(potential_name, potential_name)
        if not hasattr(self, potential_name):
            from cspy.formats.pots import parse_pots

            setattr(
                self,
                potential_name,
                parse_pots(join(POTS_DIR, f"{potential_name}.pots")),
            )
        self._required_cross_terms = self.cross_terms.get(potential_name, set())
        self.data = getattr(self, potential_name, None)
        if self.data is None:
            raise ValueError(f"Unknown potential: {potential_name}")
        self.buck, self.anis, self.dbuc = self.data

    @property
    def required_cross_terms(self):
        return self._required_cross_terms

    def pmin_form(self):
        """
        Assumes combining rules
        >>> pots, cross = PotentialData("fit").pmin_form()
        >>> round(pots['C_F1'].A, 3)
        24.148
        >>> pots, cross = PotentialData("w99rev_631").pmin_form()
        >>> round(cross[('H_W4', "O_Wa")].C, 3)
        0.281
        """
        cross = self.required_cross_terms
        pots = {}
        cross_terms = {}
        for k in self.buck:
            p = self.buck[k][k]
            pots[k] = PminAtomicPotential(
                np.sqrt(p[2] * KCAL_PER_EV), np.sqrt(p[0] * KCAL_PER_EV), 0.5 / p[1]
            )
            if cross:
                for k2 in self.buck[k]:
                    if (k, k2) in cross:
                        cross_terms[(k, k2)] = PminAtomicPotential(
                            p[2] * KCAL_PER_EV, p[0] * KCAL_PER_EV, p[1]
                        )
        return pots, cross_terms

    def orient_form(self):
        """
        Converts from A, B, C to alpha, rho, C6 
        """
        pots = {}
        for type_a, vals in self.buck.items():
            pots[type_a] = {}
            for type_b, (A, B, C, _, _) in vals.items():
                pots[type_a][type_b] = OrientPairPotential(1 / B, B * np.log(A), C)
        return pots

    def gulp_string_form(self, elements=None):
        lines = []
        if elements is not None:
            elements = set(elements)
        else:
            elements = set(x[:2].strip("_") for x in self.buck.keys())
        buck_pots = {}
        for k1, d in self.buck.items():
            el = k1[:2].strip("_")
            if el not in elements:
                continue
            buck_pots[k1] = {
                k2: v for k2, v in d.items() if k2[:2].strip("_") in elements
            }

        lines.append("buck inter")
        for k1, d in buck_pots.items():
            l = k1[:2].replace("_", "") + k1[3]
            for k2, v in d.items():
                A, B, C, r0, r1 = v
                r = k2[:2].replace("_", "") + k2[3]
                lines.append(f"{l:4s} core {r:4s} core {A:12.6f} {B:12.6f} {C:12.6f} {r0:12.6f} {r1:12.6f}")
        return "\n".join(lines)

    def neighcrys_string_form(self, elements=None):
        lines = []
        if elements is not None:
            elements = set(elements)
        else:
            elements = set(x[0] for x in self.buck.keys())
        buck_pots = {}
        for k1, d in self.buck.items():
            el = k1[:2].strip("_")
            if el not in elements:
                continue
            buck_pots[k1] = {
                k2: v for k2, v in d.items() if k2[:2].strip("_") in elements
            }

        anis_pots = {}
        for k1, d in self.anis.items():
            el = k1[:2].strip("_")
            if el not in elements:
                continue
            anis_pots[k1] = {
                k2: v for k2, v in d.items() if k2[:2].strip("_") in elements
            }
        lines = []
        for k1, d in buck_pots.items():
            anis_d = anis_pots.get(k1, None)
            for k2, v in d.items():
                if self.dbuc:
                    lines.append(f"DBUC {k1:4s} {k2:4s}")
                else:
                    lines.append(f"BUCK {k1:4s} {k2:4s}")
                lines.append(" ".join(str(x) for x in v))
                if anis_d and k2 in anis_d:
                    lines.append(f"ANIS {k1:4s} {k2:4s}")
                    for row in anis_d[k2]:
                        lines.append(" ".join(str(x) for x in row))
                lines.append("ENDS")
        return "\n".join(lines)
