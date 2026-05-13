import numpy as np
import copy
import logging
import re
from cspy.chem import Molecule, Element
from cspy.templates import get_template
from .multipole import Multipole

LOG = logging.getLogger(__name__)
NUMERIC_CONST_PATTERN = r"[-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?"
TR_PATTERN = r"[a-zA-Z]+\s+\w+"
SITE_PATTERN = r"[a-zA-Z]\w*"
SMULT_SECTION_REGEX = re.compile(
    r"\s*({s})\s+({f})\s+({f})\s+({f})\s+(.*)?\n([^A-CF-Za-cf-z]+)".format(
        s=SITE_PATTERN, f=NUMERIC_CONST_PATTERN, t=TR_PATTERN
    )
)


class DistributedMultipoles:

    template = get_template("smult")

    def __init__(self, groups):
        """A set of distributed multipoles, e.g. from GDMA"""
        self.molecules = []
        for group in groups:
            labels = []
            positions = []
            mults = []
            types = []
            for label, typeid, pos, mult in group:
                positions.append(pos)
                labels.append(label)
                mults.append(mult)
                if typeid is None:
                    typeid = Element.from_string(label).symbol
                types.append(typeid)
            self.molecules.append(
                Molecule.from_arrays(
                    elements=labels,
                    positions=positions,
                    labels=labels,
                    multipoles=mults,
                    types=types,
                )
            )

    def sort_atoms_for_neighcrys(self):
        Molecule.group_atoms_by_element(self.molecules)
        for m in self.molecules:
            m.assign_default_labels()

    def reorient_onto_molecular_axes(self, **kwargs):
        for m in self.molecules:
            m.transform(
                m.axes(**kwargs), translation=-m.center_of_mass, rotate_multipoles=True
            )

    def matching_multipoles(
        self, mol, rmsd_threshold=1e-4, foreshorten_hydrogens=False
    ):
        for i, x in enumerate(self.molecules):
            rms, reflection_plane = x.rmsd_with_reflection(
                mol, method="fixed", plane="xy"
            )
            LOG.debug(
                "RMSD: (%.3g) < (%.3g): %s", rms, rmsd_threshold, rms < rmsd_threshold
            )
            if rms < rmsd_threshold:
                LOG.debug("Match RMSD (%.3g), reflection: %s", rms, reflection_plane)
                match = copy.deepcopy(self.molecules[i])
                if reflection_plane:
                    match = match.reflect(reflection_plane)
                return match, rms
        s = (
            "No RMSD within threshold ({:.3g}) even with "
            "possible reflections".format(rmsd_threshold)
        )
        raise ValueError(s)

    def map_charges_to_molecule(self, mol):
        mapped, rms = self.matching_multipoles(mol)
        for atom, m in zip(mol.atoms, mapped.multipoles):
            atom.charge = m.charge

    def extract_mults_for(self, mols):
        multipoles = []
        for mol in mols:
            mults, rms = self.matching_multipoles(mol)
            LOG.debug("Adding %s to multipoles, rmsd = %.4f", mults, rms)
            multipoles.append(mults)
        return DistributedMultipoles.from_molecules(multipoles)

    @staticmethod
    def parse_types_ranks(string):
        typeid, rank = None, None
        for k, v in zip(*(iter(string.split()),) * 2):
            k = k.lower()
            if k == "type":
                typeid = v
            elif k == "rank":
                rank = int(v)
        return typeid, rank

    @staticmethod
    def parse_molecule_gdma(string):
        multipole_sites = []
        for label, x, y, z, tr, m in re.findall(SMULT_SECTION_REGEX, string):
            m = Multipole.from_string(m)
            typeid, rank = DistributedMultipoles.parse_types_ranks(tr)
            if rank:
                m.rank = int(rank)
            multipole_sites.append(
                (label, typeid, np.asarray((x, y, z), dtype=np.float64), m)
            )
        return multipole_sites

    @staticmethod
    def from_molecules(mols):
        d = DistributedMultipoles([])
        d.molecules = mols
        return d

    @staticmethod
    def from_dma_file(filename):
        with open(filename) as f:
            contents = f.read()
        return DistributedMultipoles.from_dma_string(contents)

    @staticmethod
    def from_dma_string(contents):
        mols = [
            DistributedMultipoles.parse_molecule_gdma(x)
            for x in contents.split("#ENDMOL")
            if x.strip()
        ]
        return DistributedMultipoles(mols)

    def to_dma_string(self, include_types=False):
        """
        >>> pos1 = np.asarray([1.0, 0.0, 0.0])
        >>> mult1 = Multipole([0.1, 0.0, 1.0, 0.0], rank=1)
        >>> groups = [[('H1', "H", pos1, mult1), ('H2', "H", pos1 + 1.0, mult1)]]
        >>> dms = DistributedMultipoles(groups)
        >>> s = dms.to_dma_string()
        """
        molecules = []
        for mol in self.molecules:
            molecular_sites = []
            labels = mol.properties.get("neighcrys_label", mol.labels)
            types = mol.properties.get("types", [x.symbol for x in mol.elements])
            for l, t, p, m in zip(labels, types, mol.positions, mol.multipoles):
                molecular_sites.append((l, t, p, m))
            molecules.append(molecular_sites)
        return self.template.render(molecules=molecules, include_types=include_types)

    def to_dma_file(self, filename, **kwargs):
        with open(filename, "w") as f:
            f.write(self.to_dma_string(**kwargs))

    @property
    def max_rank(self):
        return max(x.rank for mol in self.molecules for x in mol.multipoles)

    @max_rank.setter
    def max_rank(self, rank):
        for mol in self.molecules:
            for mult in mol.multipoles:
                mult.rank = rank

    def __repr__(self):
        return f"<{self.__class__.__name__} n={len(self.molecules)}, max_rank={self.max_rank}>"
