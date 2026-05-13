from cspy.util import perfect_square
import re
from cspy.util.re import NUMERIC
import logging
import numpy as np


LOG = logging.getLogger(__name__)
RANK_NAMES = {
    0: "charge",
    1: "dipole",
    2: "quadrupole",
    3: "octapole",
    4: "hexadecapole",
}

_GDMA_MULTIPOLE_COMPONENTS = [
    "Q00",
    "Q10",
    "Q11c",
    "Q11s",
    "Q20",
    "Q21c",
    "Q21s",
    "Q22c",
    "Q22s",
    "Q30",
    "Q31c",
    "Q31s",
    "Q32c",
    "Q32s",
    "Q33c",
    "Q33s",
    "Q40",
    "Q41c",
    "Q41s",
    "Q42c",
    "Q42s",
    "Q43c",
    "Q43s",
    "Q44c",
    "Q44s",
]

_GDMA_MULT_IDX = {x: i for i, x in enumerate(_GDMA_MULTIPOLE_COMPONENTS)}

_GDMA_MULTIPOLE_ORDERING = [
    # charge
    "1",  # R00 | 0
    # dipole
    "z",  # R10 | 1
    "x",  # R11c | 2
    "y",  # R11s | 3
    # quadrupole
    "0.5 * (3*z**2 - r**2)",  # R20 | 4
    "sqrt(3)*x*z",  # R21c | 5
    "sqrt(3)*y*z",  # R21s | 6
    "sqrt(3/4)*(x**2 - y**2)",  # R22c | 7
    "sqrt(3)*x*y",  # R22s | 8
    # octapole
    "0.5 * (5 * z**3 - 3 * z * r**2)",  # R30 | 9
    "sqrt(3/8) * x * (5 * z**2 - r**2)",  # R31c | 10
    "sqrt(3/8) * y * (5 * z**2 - r**2)",  # R31s | 11
    "sqrt(15/4) * z * (x**2 - y**2)",  # R32c | 12
    "sqrt(15) * x * y * z",  # R32s | 13
    "sqrt(5/8) * (x**3 - 3 * x * y**2)",  # R33c | 14
    "sqrt(5/8) * (3 * x**2 * y - y**3)",  # R33s | 15
    # hexadecapole
    "1/8 * (35 * z**4 - 30 * z**2 * r**2 + 3*r**4)",  # R40 | 16
    "sqrt(5/8) * (7 * x * z**3 - 3 * x * z * r**2)",  # R41c | 17
    "sqrt(5/8) * (7 * y * z**3 - 3 * y * z * r**2)",  # R41s | 18
    "sqrt(5/16) * (x**2 - y**2) * (7 * z**2 - r**2)",  # R42c | 19
    "sqrt(5/4) * x * y * (7 * z**2 - r**2)",  # R42s | 20
    "sqrt(35/8) * z * (x**3 - 3 * x * y**2)",  # R43c | 21
    "sqrt(35/8) * z * (3 * x**2 * y - y**3)",  # R43s | 22
    "sqrt(35/64) * (x**4 - 6*x**2 * y**2 + y **4)",  # R44c | 23
    "sqrt(35/4) * (x**3 * y - x * y**3)",  # R4sc | 24
]

# The indices of those solid harmonics with an odd
# power in the corresponding normal coordinate (i.e. xy -> z)
_GDMA_REFLECTION_INDICES = {
    "xy": np.array([1, 5, 6, 9, 12, 13, 17, 18, 21, 22]),
    "yz": np.array([2, 5, 8, 10, 13, 14, 17, 20, 21, 24]),
    "xz": np.array([3, 6, 8, 11, 13, 15, 18, 20, 22, 24]),
}


class Multipole:
    """class to store traceless multipoles for a given site

    Parameters
    ----------
    moments : :obj:`np.ndarray`, optional
        vector of traceless multipole moments
    rank : int, optional
        maximum rank to truncate moments or fill with zeros.
    label : str, optional
        label for this site.

    Defaults to zero for ranks higher than provided
    >>> m = Multipole(rank=1)
    >>> m.charge
    0.0
    >>> m.dipole = [0.0, 1.0, 0.0]
    >>> m.charge = 1.0
    >>> print(m.charge)
    1.0
    >>> print(m.dipole[1])
    1.0

    Discard higher ranks with rank keyword argument
    >>> m = Multipole(rank=0, moments=[1.0, 5.0, 3.0, 0.1])
    >>> len(m.moments)
    1

    """

    def __init__(self, moments=None, rank=4, label=""):
        self.label = label
        assert rank > -1
        nmoments = (rank + 1) ** 2
        self.moments = np.zeros(nmoments, dtype=np.float64)

        if moments is not None:
            moments = np.asarray(moments, dtype=np.float64)
            nmoments_provided = len(moments)
            if not perfect_square(nmoments_provided):
                raise ValueError(
                    "Invalid length of values for multipole site moments"
                    ": {} is not a square number".format(nmoments_provided)
                )
            nmoments_to_copy = min(nmoments, nmoments_provided)
            self.moments[:nmoments_to_copy] = moments[:nmoments_to_copy]
        else:
            n = rank + 1
            self.moments = np.zeros((n * n), dtype=np.float64)

    def __repr__(self):
        res = "Multipole("
        res += ",".join(
            "{}={}".format(RANK_NAMES[l], getattr(self, RANK_NAMES[l]))
            for l in range(self.rank + 1)
        )
        return res + ")"

    def transformed(self, T):
        """Transform the multipole moments to a new basis

        Parameters
        ----------
        T : :obj:`np.ndarray`
            Transformation matrix, assumed to be a transformation
            to an orthogonal (cartesian) axis system with no scale
            change

        Returns
        :obj:`Multipole`
            A copy of this multipole object with moments transformed
            to the new basis.
        """
        from .conversions import traceless_to_cartesian, expand_tensor, tensor_to_gdma

        res = Multipole(rank=self.rank, moments=np.zeros_like(self.moments))
        c, d, q, o, h = traceless_to_cartesian(self.moments)
        res.moments[0] = c
        res.moments[1:4] = tensor_to_gdma(np.dot(T, d))
        # for higher order multipoles we convert
        quadrupole = np.einsum("ik,jl,kl->ij", T, T, expand_tensor(q))
        res.moments[4:9] = tensor_to_gdma(quadrupole)
        octapole = np.einsum("il,jm,kn,lmn->ijk", T, T, T, expand_tensor(o))
        res.moments[9:16] = tensor_to_gdma(octapole)
        hexadecapole = np.einsum("ai,bj,ck,dl,ijkl->abcd", T, T, T, T, expand_tensor(h))
        res.moments[16:25] = tensor_to_gdma(hexadecapole)
        return res

    def reflected(self, plane="xy"):
        """Reflect the multipole moments about a plane

        Parameters
        ----------
        plane : str, optional
            Reflection plane, one of "xy", "xz", "yz".

        Returns
        :obj:`Multipole`
            A copy of this multipole object with moments
            reflected about the given plane.

        >>> m = Multipole(rank=1, moments=[1.0, 5.0, 3.0, 0.1])
        >>> print(m.dipole)
        [5.  3.  0.1]
        >>> m = m.reflected(plane='xy')
        >>> print(m.dipole)
        [-5.   3.   0.1]
        >>> m = m.reflected(plane='zy')
        >>> print(m.dipole)
        [-5.  -3.   0.1]
        """
        plane = "".join(sorted(plane))
        indices = _GDMA_REFLECTION_INDICES[plane]
        res = Multipole(rank=self.rank, moments=self.moments)
        indices = indices[indices < len(res.moments)]
        res.moments[indices] *= -1
        return res

    @property
    def rank(self):
        """ Maximum angular momenta.

        Returns
        -------
        int
            The highest order of multipoles contained in this object.
        """
        return int(np.sqrt(len(self.moments))) - 1

    @rank.setter
    def rank(self, rank):
        """ Decrease or increase the maximum rank (l)
        of this multipole by truncation or filling with zeros.
        Does nothing if `rank` is the same as the current max rank.

        Parameters
        ----------
        rank : int
            The new maximum rank of this Multipole.

        >>> m = Multipole(rank=2)
        >>> len(m.moments)
        9
        >>> m.rank = 3
        >>> len(m.moments)
        16
        >>> m.rank = 1
        >>> len(m.moments)
        4
        """
        if rank == self.rank:
            return
        elif rank > self.rank:
            moments = np.zeros((rank + 1) ** 2, dtype=np.float64)
            moments[: len(self.moments)] = self.moments[:]
            self.moments = moments
        else:
            moments = np.zeros((rank + 1) ** 2, dtype=np.float64)
            nmoments = (rank + 1) ** 2
            moments[:] = self.moments[:nmoments]
            self.moments = moments

    @property
    def charge(self):
        """ Site charge.

        Returns
        -------
        float
            The magnitude of the point charge at this
            site i.e. l = 0 multipole.
        """
        return self.moments[0]

    @charge.setter
    def charge(self, value):
        self.moments[0] = value

    @property
    def dipole(self):
        """ Site dipole.

        Returns
        -------
        array
            The dipole at this site i.e. l = 1 multipole
            as components [Uz, Ux, Uy]. 
        """
        if self.rank < 1:
            return None
        return self.moments[1:4]

    @dipole.setter
    def dipole(self, values):
        self.rank = max(self.rank, 1)
        self.moments[1:4] = values

    @property
    def quadrupole(self):
        """ Site quadrupole.

        Returns
        -------
        array
            The traceless quadrupole at this site i.e. l = 2 multipole
            as components [Q20, Q21c, Q21s, Q22c, Q22s].
        """
        if self.rank < 2:
            return None
        return self.moments[4:9]

    @quadrupole.setter
    def quadrupole(self, values):
        self.rank = max(self.rank, 2)
        self.moments[4:9] = values

    @property
    def octapole(self):
        if self.rank < 3:
            return None
        return self.moments[9:16]

    @octapole.setter
    def octapole(self, values):
        """ Site octapole.

        Returns
        -------
        array
            The traceless octapole at this site i.e. l = 3 multipole
            as components [Q30, Q31c, Q31s, Q32c, Q32s, Q33c, Q33s].
        """
        self.rank = max(self.rank, 3)
        self.moments[9:16] = values

    @property
    def hexadecapole(self):
        """ Site hexadecapole.

        Returns
        -------
        array
            The traceless hexadecapole at this site i.e. l = 4 multipole
            as components [Q40, Q41c, Q41s, Q42c, Q42s, Q43c, Q43s, Q44c, Q44s].
        """
        if self.rank < 4:
            return None
        return self.moments[16:25]

    @hexadecapole.setter
    def hexadecapole(self, values):
        self.rank = max(self.rank, 4)
        self.moments[16:25] = values

    def ranks(self):
        """ Iterator of the available ranks in this Multipole.

        Returns
        -------
        iterator
            Yields charge, dipole, quadrupole etc. arrays
        """
        for i in range(self.rank + 1):
            if i == 0:
                yield [self.charge]
            else:
                yield (getattr(self, RANK_NAMES[i]))

    def moment_magnitudes(self):
        """ Dictionary of the magnitudes of each multipole moment
        Not necessarily implemented correctly for anything other
        than dipole.

        Returns
        -------
        dict
            Magnitudes of charge, dipole, quadrupole etc. moments
        """
        magnitudes = {}
        for i in range(self.rank + 1):
            if i == 0:
                magnitudes["charge"] = self.charge
            else:
                n = RANK_NAMES[i]
                magnitudes[n] = np.linalg.norm(getattr(self, n))
        return magnitudes

    def __eq__(self, other):
        if self.rank != other.rank:
            return False
        return np.allclose(self.moments, other.moments, atol=1e-4, rtol=1e-5)

    @classmethod
    def from_string(cls, contents):
        r""" Create a multipole site from a string

        Parameters
        ----------
        contents : str
            A single or multiple line string in the format
            output from GDMA.

        Returns
        -------
        Multipole
            A new multipole object

        >>> m = Multipole.from_string('0.22\n  0.0 -0.1 0.0\n')
        >>> print(m.charge)
        0.22
        >>> print(m.dipole)
        [ 0.  -0.1  0. ]
        """
        try:
            moments = np.asarray([float(x) for x in contents.split()])
        except ValueError as exc:
            LOG.debug("Trying regex as split failed")
            moments = np.asarray([float(x) for x in re.findall(NUMERIC, contents)])

        return cls(moments=moments, rank=int(np.sqrt(len(moments))) - 1)

    @classmethod
    def from_moment_dict(cls, moments):
        idxs = []
        vals = []
        for k, v in moments.items():
            idxs.append(_GDMA_MULT_IDX[k])
            vals.append(v)
        moments = np.empty(len(idxs))
        moments[idxs] = vals
        return cls(moments=moments, rank=int(np.sqrt(len(moments))) - 1)

    def as_moment_dict(self):
        return {k: v for k, v in zip(_GDMA_MULTIPOLE_COMPONENTS, self.moments)}

    def as_dict(self):
        """ a dictionary representation of these multipole moments"""
        return {"label": self.label, "moments": self.moments}

    @staticmethod
    def from_dict(d):
        """ Create a multipole site from its dictionary representation.

        Parameters
        ----------
        d: dict
            Dictionary containing "label" and "moments" members.

        Returns
        -------
        Multipole
            A new multipole object

        >>> m = Multipole.from_dict({'label': 'C1', 'moments': np.array([0.1])})
        >>> print(m.charge)
        0.1
        """
        m = Multipole()
        m.moments = d["moments"]
        m.label = d["label"]
        return m
