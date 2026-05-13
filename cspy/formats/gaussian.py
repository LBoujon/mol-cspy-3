import logging
from cspy.templates import get_template
from cspy.configuration import DEFAULTS as CSPY_DEFAULTS
import numpy as np

LOG = logging.getLogger(__name__)
DEFAULTS = CSPY_DEFAULTS["gaussian"]


class GaussianScfInput(object):
    r"""
    >>> scf = GaussianScfInput(
    ... 'H 0.0 0.0 0.0\nH 1.4 0.0 0.0',
    ... 0, 1,
    ... link0={'chk': 'job'}
    ... )
    >>> s = scf.as_string()
    >>> print(s)
    %chk=job
    # B3LYP/6-311G** 
    <BLANKLINE>
    job
    <BLANKLINE>
    0 1
    H 0.0 0.0 0.0
    H 1.4 0.0 0.0
    <BLANKLINE>
    <BLANKLINE>
    <BLANKLINE>
    """
    template = get_template("gaussian_scf")

    def __init__(
        self,
        geometry,
        charge,
        multiplicity,
        title="job",
        additional_sections=(),
        link0=None,
        **kwargs
    ):
        if not link0:
            link0 = {}
        else:
            assert isinstance(link0, dict), "link0 must be a dictionary"

        self.parameters = DEFAULTS.copy()
        LOG.debug("self.parameters: \n%s", self.parameters)
        self.parameters.update(locals())
        self.parameters.pop("self")
        self.parameters.update(kwargs)

    def as_string(self):
        return self.template.render(**self.parameters)


class GaussianLogFile:
    optimized_tag = "Optimized Parameters"
    std_xyz_tag = "Standard orientation"
    inp_xyz_tag = "Input orientation"

    def __init__(self, filename=None):
        self.filename = filename
        if self.filename is not None:
            self.read_contents()

    def read_contents(self):
        with open(self.filename) as f:
            self.content_lines = f.readlines()

    @property
    def geometries(self):
        if hasattr(self, "_geometries"):
            return self._geometries
        lines_iter = iter(self.content_lines)
        parsing_xyz = None
        geometries = {"inp": [], "std": []}
        for line in lines_iter:
            # if optimized.. not found and we are not looking for original,
            # loop here.
            if self.inp_xyz_tag in line:
                # skip 4 lines
                for _ in range(4):
                    next(lines_iter)
                    parsing_xyz = "inp"
            elif self.std_xyz_tag in line:
                # skip 4 lines
                for _ in range(4):
                    next(lines_iter)
                    parsing_xyz = "std"
            if parsing_xyz:
                atomic_numbers = []
                positions = []
                while True:
                    next_line = next(lines_iter)
                    if "----" in next_line:
                        break
                    tokens = next_line.strip().split()
                    atomic_numbers.append(int(tokens[1]))
                    positions.append(tuple(map(float, tokens[3:6])))
                geometries[parsing_xyz].append(
                    {
                        "elements": np.asarray(atomic_numbers),
                        "positions": np.asarray(positions),
                    }
                )
                parsing_xyz = None
        self._geometries = geometries
        return geometries

    @property
    def final_geometry(self, kind="inp"):
        return self.geometries[kind][-1]

    def parse_archive_block(self):
        block_lines = []
        adding_lines = False
        for line in self.content_lines:
            # can't end raw string with backslash
            if adding_lines:
                block_lines.append(line.strip())
            if r"1\1" in line:
                adding_lines = True
                block_lines.append(line.strip())
            if r"\\@" in line:
                break
        self._archive_block_sections = "".join(block_lines).split("\\")
        for section in self._archive_block_sections:
            if "=" in section:
                tokens = section.split("=")
                if len(tokens) == 2:
                    l, r = tokens
                    setattr(self, l, r)

    @property
    def final_energy(self):
        if not hasattr(self, "_archive_block_sections"):
            self.parse_archive_block()
        return float(getattr(self, "HF", "nan"))

    @property
    def dipole(self):
        if not hasattr(self, "_archive_block_sections"):
            self.parse_archive_block()
        return np.array(
            [float(x) for x in getattr(self, "Dipole", "nan,nan,nan").split(",")]
        )

    @property
    def quadrupole(self):
        if not hasattr(self, "_archive_block_sections"):
            self.parse_archive_block()
        return np.array(
            [
                float(x)
                for x in getattr(self, "Quadrupole", "nan,nan,nan,nan,nan,nan").split(
                    ","
                )
            ]
        )

    @property
    def normal_termination(self):
        return "Normal termination" in self.content_lines[-1]

    @classmethod
    def from_string(cls, contents):
        res = cls()
        res.content_lines = contents.splitlines()
        return res


class GaussianFchkFile:

    UNIT_CONVERSIONS = {"kj/mol": 2625.499638, "hartree": 1.0}
    CONVERSIONS = {"R": float, "I": int, "C": str}

    def __init__(self, filename, parse=False):
        self.filename = filename
        if parse:
            self._parse()

    def _parse(self):
        energy = 0.0
        contents = {}
        with open(self.filename) as f:
            contents["header"] = (f.readline(), f.readline())
            line = f.readline()

            while line:
                name = line[:43].strip()
                kind = line[43]
                num = 1
                if line[47:49] == "N=":
                    num = int(line[49:].strip())
                    value = []
                    line = f.readline()
                    valid = True
                    while valid:
                        tokens = line.strip().split()
                        convert = self.CONVERSIONS[kind]
                        value += [convert(x) for x in tokens]
                        line = f.readline()
                        valid = line and line[0] == " "
                else:
                    value = self.CONVERSIONS[kind](line[48:].strip())
                    line = f.readline()
                contents[name] = value
        self.contents = contents

    def __getitem__(self, key):
        return self.contents[key]

    def _parse_energy_only(self):
        with open(self.filename) as f:
            for line in f:
                if line.startswith("Total Energy"):
                    return float(line[49:].strip())

    @classmethod
    def scf_energy(cls, fname, units="hartree"):
        fchk = cls(fname, parse=False)
        return cls.UNIT_CONVERSIONS[units] * fchk._parse_energy_only()
