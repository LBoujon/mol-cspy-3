#!/usr/bin/env python
import argparse
from collections import defaultdict
import logging
import copy
import numpy as np
import re
from collections import namedtuple
from cspy.templates import get_template
from .neighcrys_potential_codes import NC_POT_CODE_TO_LABEL
from .crystal import Crystal

LOG = logging.getLogger(__name__)

Line = namedtuple("Line", "axis atom_a atom_b sep_ab xyz_a xyz_b")
Plane = namedtuple("Plane", "axis atom_a atom_b sep_ab atom_c sep_ac xyz_a xyz_b xyz_c")
HALOGENS = {9, 17, 35, 53}
AXIS_TEMPLATE = get_template("neighcrys_axis")
AVAILABLE_POTS = NC_POT_CODE_TO_LABEL.keys()
MOL_AXIS_REGEX = re.compile(
    r"X\s+LINE\s+(\w+)\s+(\w+)\s+(\d+)\s*\n"
    r"Y\s+PLANE?\s+(\w+)\s+(\w+)\s+(\d+)\s+(\w+)\s+(\d+)"
)


class NeighcrysLabeller:
    def __init__(self, potential_type):
        if potential_type not in AVAILABLE_POTS:
            raise ValueError(
                "potential_type should be one of {}".format(
                    tuple(x for x in AVAILABLE_POTS)
                )
            )
        self.potential_type = potential_type
        self.pot_codes = NC_POT_CODE_TO_LABEL[self.potential_type]

    def label(self, crystal, alter_crystal=False, include_uc_index=False):
        """ Get the string neighcrys label for all atoms in the asymmetric_unit of
        a given crystal

        Parameters
        ----------
        self : NeighcrysAxis
            This NeighcrysAxis object
        crystal: Atom
            The atom in question, assumes that an atomic
            label has been set, `atom.potential_label`,
            `atom.element_index` and `atom.uc_index` have
            been set

        Returns
        -------
        label : str
            the neighcrys label for the atom
        """

        fields = "{atom:_<2s}{pot:_>2s}{inv}{num:_<5s}"
        molecules = crystal.symmetry_unique_molecules()

        if sum(len(x) for x in molecules) != len(crystal.asymmetric_unit):
            error_msg = (
                f"AsymmetricUnit in crystal does not consist of whole molecules: "
                f"{len(crystal.asymmetric_unit)} in asymmetric unit vs. "
                f"{sum(len(x) for x in molecules)} atoms in symmetry unique molecules"
            )
            raise ValueError(error_msg)
        counts = defaultdict(int)
        asym_element_index = []
        for element in crystal.asymmetric_unit.elements:
            counts[element] += 1
            asym_element_index.append(counts[element])

        inversion = False
        crystal_labels = []
        for mol in molecules:
            potential_labels, potential_codes = zip(
                *self.atom_potential_labels(mol, with_code=True)
            )
            labels = [
                fields.format(
                    atom=el.symbol,
                    pot=pl[-2:],
                    inv="I" if inversion else "_",
                    num=str(asym_element_index[asym]),
                )
                for el, pl, asym in zip(
                    mol.elements,
                    potential_labels,
                    mol.properties["asymmetric_unit_atoms"],
                )
            ]
            if alter_crystal:
                mol.properties["neighcrys_label"] = labels
                mol.properties["neighcrys_potential_label"] = potential_labels
                mol.properties["neighcrys_potential_code"] = potential_codes
            crystal_labels.append(labels)
        return crystal_labels

    def atom_potential_labels(self, mol, with_code=False):
        """ Set neighcrys potential labels for each
        atom in `mol`, based on the connectivity in `mol`.

        The following table illustrates the rules:

        Element     Bonding          FIT    Williams
        --------------------------------------------
        H           C-H group       H_F1        H_W1
        H           alcoholic       H_F2        H_W2
        H           carboxyl        H_F2        H_W3
        H           N-H             H_F2        H_W4
        H           in water        H_Wa        H_Wa

        C           4 bonds         C_F1        C_W4
        C           3 bonds         C_F1        C_W3
        C           2 bonds         C_F1        C_W2

        N           triple bond     N_F1        N_W1
        N           ~H (otherwise)  N_F1        N_W2
        N           1H              N_F1        N_W3
        N           >2H             N_F1        N_W4

        O           1 bond          O_F1        O_W1
        O           2 bonds         O_F1        O_W2
        O           water           O_Wa        O_Wa

        Parameters
        ----------
        mol : Molecule
            The molecule in question, assumed to be
            connected, though it may work regardless.
            Will have each atom modified
        """
        atomic_numbers = np.array([a.atomic_number for a in mol.elements])
        # convert connectivity to adjacency list
        neighbours = []
        for i in range(len(mol)):
            neighbours.append(list(mol.bonds[i].nonzero()[1]))

        potential_labels = []
        for i in range(len(mol)):
            potential_label = self.nc_potential_label(
                i, neighbours, atomic_numbers, with_code=with_code
            )
            potential_labels.append(potential_label)
        return potential_labels

    def nc_potential_label(self, idx, neighbours, atomic_numbers, with_code=False):
        """ Get the neighcrys potential label for the atom at the
        provided `idx` within the molecule defined by `neighbours`
        and `atomic_numbers`

        Modified from the neighcrys subroutine to set the atom type.
        in utilities.f90. It supports the FIT and Williams
        (JCC 22, 1154-1166, 2001) potentials.

        Parameters
        ----------
        self : NeighcrysAxis
            This NeighcrysAxis object
        idx : int
            The index for the given atom
        neighbours : list(tuple)
            Adjacency list for all atoms in the molecule
        atomic_numbers : array(int)
            atomic numbers for each atom in the molecule

        Returns
        -------
        code : str
            Neighcrys potential type (e.g. W4, F1, 01)
        """

        nc_potential_code = 0

        # HYDROGEN
        num_neighbours = len(neighbours[idx])
        atomic_number = atomic_numbers[idx]
        if atomic_number == 1:
            if num_neighbours != 1:
                raise ValueError(
                    f"Neighcrys requires Hydrogen atoms to be bonded to only 1 atom, "
                    f"atom {idx} is bonded to {num_neighbours}"
                )
            neigh1 = neighbours[idx][0]
            neigh1_el = atomic_numbers[neigh1]

            # CONNECTED TO CARBON
            if neigh1_el == 6:
                nc_potential_code = 501
            # CONNECTED TO NITROGEN
            elif neigh1_el == 7:
                nc_potential_code = 504
            # CONNECTED TO OXYGEN
            elif neigh1_el == 8:
                nc_potential_code = 502  # assume alcoholic group
                for neigh2 in neighbours[neigh1]:
                    if neigh2 == idx:
                        continue
                    neigh2_el = atomic_numbers[neigh2]
                    if neigh2_el == 6:
                        # CHECK IF IT IS A CARBOXYLIC HYDROGEN
                        # loop over the atoms X, where X(KC)-C(LC)-O(MC)-H(idx)
                        for neigh3 in neighbours[neigh2]:
                            if neigh3 == neigh1:
                                continue

                            neigh3_el = atomic_numbers[neigh3]
                            if neigh3_el != 8:
                                continue

                            if len(neighbours[neigh3]) == 1:
                                nc_potential_code = 503
                                break
                    # CHECK IF IT IS IN WATER
                    if len(neighbours[neigh1]) == 2:
                        neigh_elements = [atomic_numbers[n] for n in neighbours[neigh1]]
                        if all(n == 1 for n in neigh_elements):
                            nc_potential_code = 505
                            break
        # CARBON
        elif atomic_number == 6:
            if num_neighbours == 4:
                nc_potential_code = 511
            elif num_neighbours == 3:
                nc_potential_code = 512
            elif num_neighbours == 2:
                nc_potential_code = 513
        # NITROGEN
        elif atomic_number == 7:
            if num_neighbours == 1:
                nc_potential_code = 521
            else:
                # COUNT CONNECTED HYDROGENS
                num_hydrogens = 0
                for neigh1 in neighbours[idx]:
                    if atomic_numbers[neigh1] == 1:
                        num_hydrogens += 1
                if num_hydrogens == 0:
                    nc_potential_code = 522
                elif num_hydrogens == 1:
                    nc_potential_code = 523
                else:
                    nc_potential_code = 524
        # OXYGEN
        elif atomic_number == 8:
            if num_neighbours == 1:
                nc_potential_code = 531
            elif num_neighbours == 2:
                nc_potential_code = 532

                # CHECK IF IT IS AN OXYGEN IN WATER
                num_hydrogens = 0
                for neigh1 in neighbours[idx]:
                    if atomic_numbers[neigh1] == 1:
                        num_hydrogens += 1
                if num_hydrogens == 2:
                    nc_potential_code = 533
        # FLUORINE
        elif atomic_number == 9:
            nc_potential_code = 540
        # CHLORINE
        elif atomic_number == 17:
            nc_potential_code = 541
        # SULFUR
        elif atomic_number == 16:
            nc_potential_code = 542
        # POTASSIUM
        elif atomic_number == 19:
            nc_potential_code = 543
        # BROMINE
        elif atomic_number == 35:
            nc_potential_code = 544
        # ATOMS NOT EXPLICITLY INCLUDED IN THE POTENTIALS
        else:
            nc_potential_code = 0
            if with_code:
                return "01", nc_potential_code
        code = self.pot_codes[nc_potential_code]
        if with_code:
            return code[-2:], nc_potential_code
        return code[-2:]


class NeighcrysAxis:

    """Utility class to enable the generation of valid
    neighcrys axis files from a given molecule.

    This assumes all atoms in the crystal object have
    been given uc_index and element_index parameters upon
    reading in the res file (should be set in
    Molecule.fragement_geometry)

    This supports:
        Water, anisotropic halogens in molecules

    Does not support:
        Lone atoms (need to use dummy atoms)
        diatomic molecules
    """

    def __init__(
        self,
        crystal,
        potential_type="F",
        file_content=None,
        alter_crystal=False,
        gen=True,
    ):
        self.potential_type = potential_type
        self.labeller = NeighcrysLabeller(potential_type=self.potential_type)
        self.crystal = crystal
        self.file_content = file_content
        self.alter_crystal = alter_crystal
        self.molecular_axis_atom_labels = None
        if gen:
            self.generate_molecular_axes()

    def generate_molecular_axes(self):
        """Generate the lists `self.molecular_axes`,
        `self.anisotropic_axes` by iterating through
        the molecules in `self.crysta`

        Parameters
        ----------
        self : NeighcrysAxis
            The axis object containing the crystal
        """
        self.molecular_axes = []
        self.anisotropic_axes = []
        if self.file_content is not None:
            self._parse_axis_file_contents()

        labels = self.labeller.label(self.crystal, alter_crystal=self.alter_crystal)

        for mol, mol_labels in zip(self.crystal.symmetry_unique_molecules(), labels):
            # no axis for single atom molecules
            if len(mol) == 1:
                continue
            l, p, anis = self.get_axis(
                mol, mol_labels, alter_molecule=self.alter_crystal
            )
            self.molecular_axes.append((l, p))
            self.anisotropic_axes += anis

    def _parse_axis_file_contents(self):
        assert self.file_content, "Cannot parse empty contents"

        if "_W1" in self.file_content and self.potential_type != "W":
            LOG.warn(
                "altering potential_type %s -> W, as per axis file", self.potential_type
            )
            self.potential_type = "W"
        molx = int(re.findall(r"MOLX\s+(\d+)", self.file_content)[0])
        mol_axes = re.findall(MOL_AXIS_REGEX, self.file_content)
        assert len(mol_axes) == molx, "Number of axes in axis file must match MOLX"

        self.molecular_axis_atom_labels = mol_axes

    def get_axis(self, mol, mol_labels, alter_molecule=False):
        """ Generate a molecular axis frame for the provided
        molecule (`mol`). By default uses the first 3 atoms found.

        Parameters
        ----------
        self : NeighcrysAxis
            This NeighcrysAxis object
        mol : Molecule
            The molecule in question

        Returns
        -------
        axis : tuple(Line, Plane)
            Axis defined by a line and a plane in `mol`
        """

        LOG.debug("Basis atoms:\n%s", "\n".join(mol_labels))
        if self.molecular_axis_atom_labels:
            for ax in self.molecular_axis_atom_labels:
                ax_labels = ax[0], ax[1], ax[-2]
                try:
                    atom_idxs = [mol_labels.index(ax_label) for ax_label in ax_labels]
                    break
                except ValueError as e:
                    continue
            else:
                LOG.warning(
                    "Could not find matching labels for axis, "
                    "defaulting to first 3 atoms"
                )
                LOG.warning(
                    "Potential type: %s, axis contents:\n%s",
                    self.potential_type,
                    self.file_content,
                )
                LOG.warning("Basis atoms:\n%s", "\n".join(mol_labels))
                atom_idxs = [0, 1, 2]
        else:
            atom_idxs = [0, 1, 2]

        a, b, c = atom_idxs
        if alter_molecule:
            mol.properties["neighcrys_axis_atom"] = (a, b, c)
        anisotropic_atoms = self.get_anisotropic_axes(mol, mol_labels)
        separations = mol.bond_path_separation()
        sep_ab = separations[a, b]
        sep_ac = separations[a, c]
        return (
            Line(
                axis="X",
                atom_a=mol_labels[a],
                atom_b=mol_labels[b],
                sep_ab=sep_ab,
                xyz_a=mol.positions[a],
                xyz_b=mol.positions[b],
            ),
            Plane(
                axis="Y",
                atom_a=mol_labels[a],
                atom_b=mol_labels[b],
                sep_ab=sep_ab,
                atom_c=mol_labels[c],
                sep_ac=sep_ac,
                xyz_a=mol.positions[a],
                xyz_b=mol.positions[b],
                xyz_c=mol.positions[c],
            ),
            anisotropic_atoms,
        )

    def get_anisotropic_axes(self, mol, mol_labels):
        """ Generate atomic anisotropic axes for the provided
        molecule (`mol`). Default atoms to look for are the halogens
        as these are defined in Williams (W) potentials. FIT (F)
        potentials will make this a null operation

        Parameters
        ----------
        self : NeighcrysAxis
            This NeighcrysAxis object
        mol : Molecule
            The molecule in question

        Returns
        -------
        axes : list(label, Line, Plane)
            list of anisotropic axes for the atoms in `mol`
            represented by label
        """
        axes = []
        inv_offset = len(mol_labels)
        inv_labels = []
        for l in mol_labels:
            idx = str(int(l[5:].replace("_", "")) + inv_offset).ljust(5, "_")
            inv_labels.append(l[:4] + "I" + idx)

        if self.potential_type == "F":
            # No anisotropic potentials in FIT
            return axes

        atomic_numbers = np.array([a.atomic_number for a in mol.elements])
        if not any(n in HALOGENS for n in atomic_numbers):
            return axes

        neighbours = []
        for i in range(len(mol)):
            neighbours.append(list(mol.bonds[i].nonzero()[1]))

        separations = mol.bond_path_separation()
        for i, n in enumerate(atomic_numbers):
            if n in HALOGENS:
                num_neighbours = len(neighbours[i])
                if num_neighbours != 1:
                    sym = mol.elements[i].symbol
                    error_msg = (
                        f"{sym} atom idx={i} is bonded to {num_neighbours} atoms: "
                        "only halogens with 1 neighbour are supported"
                    )
                    raise RuntimeError(error_msg)
                neigh1 = neighbours[i][0]
                for neigh2 in neighbours[neigh1]:
                    if neigh2 != i:
                        break
                else:
                    LOG.error("Case of molecule with <3 atoms not supported")
                axes.append((i, neigh1, neigh2))
        anisotropic_axes = [
            (
                mol_labels[i],
                Line(
                    axis="Z",
                    atom_a=mol_labels[n1],
                    atom_b=mol_labels[i],
                    sep_ab=separations[n1, i],
                    xyz_a=mol.positions[n1],
                    xyz_b=mol.positions[i],
                ),
                Plane(
                    axis="X",
                    atom_a=mol_labels[n2],
                    atom_b=mol_labels[n1],
                    sep_ab=separations[n2, n1],
                    atom_c=mol_labels[i],
                    sep_ac=separations[n2, i],
                    xyz_a=mol.positions[n2],
                    xyz_b=mol.positions[n1],
                    xyz_c=mol.positions[i],
                ),
            )
            for i, n1, n2 in axes
        ]
        # include inverted sites for anisotropic repulsion
        anisotropic_axes += [
            (
                inv_labels[i],
                Line(
                    axis="Z",
                    atom_a=inv_labels[n1],
                    atom_b=inv_labels[i],
                    sep_ab=separations[n1, i],
                    xyz_a=mol.positions[n1],
                    xyz_b=mol.positions[i],
                ),
                Plane(
                    axis="X",
                    atom_a=inv_labels[n2],
                    atom_b=inv_labels[n1],
                    sep_ab=separations[n2, n1],
                    atom_c=inv_labels[i],
                    sep_ac=separations[n2, i],
                    xyz_a=mol.positions[n2],
                    xyz_b=mol.positions[n1],
                    xyz_c=mol.positions[i],
                ),
            )
            for i, n1, n2 in axes
        ]
        return anisotropic_axes

    def to_string(self):
        """ Generate the string reperesentation of the given
        neighcrys axis object (for input in neighcrys/dmacrys)

        Parameters
        ----------
        self : NeighcrysAxis
            This axis object, assumes that molecular_axes and
            anisotropic_axes have been generated

        Returns
        -------
        axis_string : str
            The textual representation of the axes (see
            neighcrys_axis.template)
        """
        return AXIS_TEMPLATE.render(
            nmols=len(self.molecular_axes),
            molecular_axes=self.molecular_axes,
            anisotropic_axes=self.anisotropic_axes,
        )

    def __str__(self):
        return self.to_string()

    @classmethod
    def from_axis_file_and_crystal(cls, axis_filename, crystal):
        with open(axis_filename) as f:
            return NeighcrysAxis.from_axis_string_and_crystal(f.read(), crystal)


def main():
    from cspy.util.path import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("res", type=str, help="Res file")
    parser.add_argument(
        "-p",
        "--potential-type",
        choices=("C", "F", "W"),
        default="F",
        help="Custom (C), FIT (F) or Williams(W) " "potential form",
    )
    parser.add_argument(
        "-e", "--extension", default=".mols", help="File extension for output"
    )
    parser.add_argument(
        "--log-level", default="INFO", help="level of logging information"
    )
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)
    crystal = Crystal.load(args.res)
    LOG.info(
        "%d molecules in asymmetric unit", len(crystal.symmetry_unique_molecules())
    )

    axis_def = NeighcrysAxis(crystal, potential_type=args.potential_type)
    ax_str = axis_def.to_string()

    LOG.info("Axis file contents:\n%s", ax_str)
    p = Path(args.res).with_suffix(args.extension)
    LOG.debug("writing axis to file: %s", p.name)
    p.write_text(ax_str)


if __name__ == "__main__":
    main()
