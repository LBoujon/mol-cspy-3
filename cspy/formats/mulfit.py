from cspy.templates import get_template
from cspy.chem.multipole import Multipole, DistributedMultipoles
from pathlib import Path
import numpy as np
import logging

MULFIT_TEMPLATE = get_template("mulfit")
LOG = logging.getLogger(__name__)


class MulfitInput:
    """class to generate input files for MULFIT for fitting lower order
    multipoles for a given :obj:`DistributedMultipoles` instance.

    Parameters
    ----------
    mults: :obj:`DistributedMultipoles`
        DMA consisting of a single molecule

    Keyword Args
    ------------
    default_rank : int, optional
        default maximum rank to fit new moments. All ranks below this will also be used
    limit : int, optional
        maximum rank of DMA to use in fitting (i.e. rank of input multipoles).
    type_radii : dict, optional
        dict of type radii (str, float) to alter the default (vdw radius) as the atomic radius per atom type
    type_ranks : dict, optional
        dict of space separated ranks (str, str) e.g. "0 1" for given types where we do not wish to use the default rank
    dma_method : str, optional
        for future possible use, thought currently only GDMA is supported
    dma_file : str, optional
        path to write the input dma file (default tmp_mulfit_input.dma)
    title : str, optional
        title for the mulfit job
    verbose : int, optional
        verbosity for mulfit output
    inner_shell : float, optional
        minimum distance for range where fitting is performed
    outer_shell : float, optional
        maximum distance for range where fitting is performed
    tensors : tuple, optional
        tuple of int, floats for which derivatives to use in the fit (see manual)
    method : str, optional
        either 'cumulative' or 'overall', detailing which type of fit to use
    """

    def __init__(self, mults, **kwargs):
        self.mults = mults
        molecules = mults.molecules
        assert len(molecules) == 1, "Mulfit can only be run on one molecule at a time"
        self.type_radii = kwargs.get("type_radii", {})
        self.type_ranks = kwargs.get("type_ranks", {})
        # by default we fit charges for all sites
        self.default_rank = kwargs.get("default_rank", 0)
        self.dma_method = kwargs.get("dma_method", "GDMA")
        self.dma_file = kwargs.get("dma_file", "tmp_mulfit_input.dma")
        self.title = kwargs.get("title", "refit for cspy2")
        self.verbose = kwargs.get("verbose", 0)
        self.inner_shell = kwargs.get("inner_shell", 1.2)
        self.outer_shell = kwargs.get("outer_shell", 3.0)
        self.limit = kwargs.get("limit", 6)
        self.tensors = kwargs.get("tensors", (0, 1.0))
        self.method = kwargs.get("mulfit_method", "CUMULATIVE").upper()

        for mol in molecules:
            mults = mol.properties.get("multipoles", None)
            if mults is None:
                raise ValueError("Molecule must have multipoles in order to use mulfit")
            # populate type radii
            for t, r, m in zip(mol.properties.get("types", []), mol.vdw_radii, mults):
                if t not in self.type_radii:
                    self.type_radii[t] = r
                if t not in self.type_ranks:
                    self.type_ranks[t] = " ".join(
                        str(i) for i in range(self.default_rank + 1)
                    )

    def as_string(self):
        "Render the input contents to a string"
        return MULFIT_TEMPLATE.render(
            title=self.title,
            verbose=self.verbose,
            type_radii=self.type_radii,
            dma_file=self.dma_file,
            dma_method=self.dma_method,
            shell_inner=self.inner_shell,
            shell_outer=self.outer_shell,
            tensors=" ".join(str(t) for t in self.tensors),
            type_ranks=self.type_ranks,
            limit=self.limit,
            method=self.method,
        )

    def __str__(self):
        return self.as_string()

    def write_dma_file(self):
        """Write the DMA string corresponding to self.mults to self.dma_file in
        preparation for running MULFIT
        """
        LOG.debug("Writing dma file to %s", self.dma_file)
        Path(self.dma_file).write_text(self.mults.to_dma_string(include_types=True))


class MulfitOutput:
    """class to parse output files from MULFIT for extracting
    fitted lower order multipoles.

    Parameters
    ----------
    contents : str
        Output file contents
    parse : bool, optional
        Whether or not to parse the file upon construction (default True)
    """

    def __init__(self, contents, parse=True):
        self.lines = contents.splitlines()
        self.groups = []
        self.sites = []
        self.distributed_multipoles = None
        if parse:
            self._parse()

    def _parse_multipole_section(self, start, end):
        l = None
        xyz = None
        t = None
        moments = None
        rank = None
        i = 0
        sites = []
        for line in self.lines[start : end + 1]:
            i += 1
            if "Rank" in line:
                if moments is not None:
                    mults = Multipole.from_moment_dict(moments)
                    sites.append((l, t, xyz, mults))
                tokens = line.split()
                l = tokens[0]
                xyz = np.array([float(x) for x in tokens[1:4]])
                for k, v in zip(*(iter(tokens[4:]),) * 2):
                    k = k.lower()
                    if k == "type":
                        t = v
                    elif k == "rank":
                        rank = int(v)
                moments = {}
            elif "Q" in line:
                tokens = line.split()
                moments[tokens[0]] = float(tokens[2])
        if moments is not None:
            mults = Multipole.from_moment_dict(moments)
            sites.append((l, t, xyz, mults))

        self.groups.append(sites)

    def _parse(self):
        multipoles_start = -1
        multipoles_end = -1
        for i, line in enumerate(self.lines):
            if line.startswith("Fitted multipoles"):
                multipoles_start = i + 1
            elif "End" in line:
                if multipoles_start > -1:
                    multipoles_end = i

        self._parse_multipole_section(multipoles_start, multipoles_end)
        self.distributed_multipoles = DistributedMultipoles(self.groups)

    @classmethod
    def from_mulfit_output_file(cls, filename):
        "Convenience constructor from a filename"
        return cls(Path(filename).read_text())

    @classmethod
    def parse_mulfit_output_file(cls, filename):
        "Convenience method to immediately get parsed :obj:`DistributedMultipoles` instance"
        return cls.from_mulfit_output_file(filename).distributed_multipoles
