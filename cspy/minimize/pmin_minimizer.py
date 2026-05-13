import logging
from cspy.executable import Pmin, ReturnCodeError, TimeoutExpired
from cspy.potentials import PotentialData, available_potentials
from cspy.crystal import Crystal
import copy
import shutil
from cspy.util.path import Path
from os.path import exists, basename, join
from cspy.templates import get_template
from collections import namedtuple
import sys
import re
from tempfile import TemporaryDirectory
import os
import time


LOG = logging.getLogger("PminMinimizer")
PMIN_TEMPLATE = get_template("pmin")
PminAtom = namedtuple("PminAtomRow", "label code a b c charge potential mass group")

LATTICE_REFINEMENT_CODES = {
    "triclinic": "111000",
    "monoclinic": "111000",
    "orthorhombic": "111000",
    "tetragonal": "101000",
    "hexagonal": "101000",
    "trigonal": "111000",
    "cubic": "100000",
    "rhombohedral": "100000",
}

ISYSTEM = {
    "monoclinic": 0,
    "orthorhombic": 0,
    "triclinic": 0,
    "hexagonal": 1,
    "tetragonal": 1,
    "cubic": 2,
    "rhombohedral": 3,
    "trigonal": 3,
}


class PminMinimizer:
    def __init__(self,
                 multipoles,
                 axis,
                 reorder_atoms_if_high_rmsd=False,
                 reorder_method='molecular_axis',
                 name="Unknown",
                 potential="fit",
                 **kwargs):
        from cspy.configuration import CONFIG

        LOG.debug("Initializing PminMinimizer: %s", multipoles)
        self.multipoles = multipoles
        self.axis = axis
        self.reorder_atoms_if_high_rmsd = reorder_atoms_if_high_rmsd
        self.reorder_method = reorder_method
        self.name = name
        self.args = copy.deepcopy(CONFIG.get("pmin"))
        self.args.update(**kwargs)
        self.potential = potential
        self.potentials, self.cross_terms = PotentialData(potential).pmin_form()
        self.potential_type = available_potentials[potential][0]
        if self.potential_type == "W":
            self.args["foreshorten_hydrogens"] = self.args.get(
                "foreshorten_hydrogens", True
            )

    def setup_minimization(self, crystal):
        args = self.args.copy()
        args["title"] = self.name
        args["irosen"] = 1
        args["nmols"] = len(crystal.symmetry_unique_molecules())
        args["refcode"] = crystal.titl
        args["lattice_parameters"] = crystal.unit_cell.parameters
        args["cross_terms"] = {}
        atoms = []

        crystal.neighcrys_setup(
            potential_type=self.potential_type, file_content=self.axis
        )
        reduction = 0.1 if self.args.get("foreshorten_hydrogens") else None
        crystal.map_multipoles(self.multipoles,
                               foreshorten_hydrogens=reduction,
                               reorder_atoms_if_high_rmsd=self.reorder_atoms_if_high_rmsd,
                               reorder_method=self.reorder_method)
        for i, mol in enumerate(crystal.symmetry_unique_molecules(), start=1):
            labels = mol.properties["asymmetric_unit_labels"]
            potential_labels = mol.properties["neighcrys_potential_label"]
            potential_codes = mol.properties.get(
                "neighcrys_potential_code", [500] * len(mol)
            )
            frac_pos = crystal.to_fractional(mol.positions)
            charges = [x.charge for x in mol.multipoles]
            for l, el, pot_label, pot_code, frac, charge in zip(
                labels,
                mol.elements,
                potential_labels,
                potential_codes,
                frac_pos,
                charges,
            ):
                label = el.symbol.ljust(2, "_") + pot_label.rjust(2, "_")
                code = pot_code
                if label not in self.potentials:
                    LOG.error("Available keys: %s", self.potentials.keys())
                    raise ValueError(f"Unknown potential {label} in {self.potential}")
                potential = self.potentials[label]
                a, b, c = frac
                atoms.append(PminAtom(l, code, a, b, c, charge, potential, el.mass, i))
            for i in range(len(labels)):
                l1 = labels[i]
                el1 = mol.elements[i].symbol
                p1 = f"{el1}_{potential_labels[i]}"
                for j in range(i + 1, len(labels)):
                    l2 = labels[j]
                    el2 = mol.elements[j]
                    p2 = f"{el2}_{potential_labels[j]}"
                    if (p1, p2) in self.cross_terms:
                        args["cross_terms"][(l1, l2)] = self.cross_terms[(p1, p2)]
                    elif (p2, p1) in self.cross_terms:
                        args["cross_terms"][(l2, l1)] = self.cross_terms[(p2, p1)]

        args["icross"] = len(args["cross_terms"])
        args["atoms"] = atoms
        args["natom"] = len(atoms)
        # symmetry operations must have identity symop first in PMIN
        args["symops"] = [
            (x.rotation, x.translation)
            for x in crystal.space_group.ordered_symmetry_operations()
        ]
        args["nsym"] = len(args["symops"])
        args["molwt"] = sum(x.mass for x in atoms)
        args["isystem"] = ISYSTEM[crystal.space_group.lattice_type]
        args["refinement_code"] = int(
            LATTICE_REFINEMENT_CODES[crystal.space_group.lattice_type]
            + "111111" * args["nmol"]
        )
        args["group_identifiers"] = ["{:3d}".format(x.group) for x in atoms]
        self.input_contents = PMIN_TEMPLATE.render(**args)

    def run_pmin(self, working_directory):
        exe = Pmin(
            self.input_contents, name=self.name, working_directory=working_directory
        )
        t1 = time.time()
        exe.run()
        t2 = time.time()
        success = exe.summary_contents is not None
        LOG.debug(
            "PMIN minimization %s in %.2fs",
            "succeeded" if success else "failed",
            t2 - t1,
        )
        if not success:
            LOG.error(
                "Last line in PMIN output: %s", exe.output_contents.splitlines()[-1]
            )
        return exe

    def minimize(self, crystal):
        LOG.debug("Minimizing crystal %s", crystal.titl)
        with TemporaryDirectory(prefix="/dev/shm/") as tmpdirname:
            self.setup_minimization(crystal)
            try:
                pm = self.run_pmin(tmpdirname)
            except (ReturnCodeError, TimeoutExpired) as exc:
                LOG.exception("Error in pmin minimization: %s", exc)
                return None
            output_lines = [
                x for x in pm.output_contents.splitlines() if x.startswith("PMIN")
            ]
            try:
                for method, line in zip(["RSS", "LSQ"], output_lines):
                    tokens = line.split(";")
                    output_contents = {}
                    for token in tokens:
                        k, v = token.split("=")
                        output_contents[k.strip()] = v.strip()
                    try:
                        energy = 4.184 * float(output_contents["Lat E"])
                    except:
                        LOG.exception("Error in Pmin: Invalid energy at end of optimisation. Probable Buckingham catastrophe.", stack_info=False, exc_info=False)
                        return None
                    density = float(output_contents["density"])
                    LOG.debug(
                        "%s Energy: %.2f, density: %.2f ", method, energy, density
                    )
            except ValueError as exc:
                LOG.exception("Error in pmin minimization: %s", exc)
                return None
            result = Crystal.from_pmin_save_string(
                pm.save_contents, space_group=crystal.space_group
            )
            result.properties["titl"] = crystal.titl + " {} {}".format(energy, density)
            result.properties["lattice_energy"] = energy
            result.properties["density"] = density
            result.properties["cell_volume"] = result.unit_cell.volume()
            result.properties["pmin_save"] = pm.save_contents
        return result

    def __repr__(self):
        return "PminMinimizer(for='{}')".format(self.name)

    def __call__(self, crystal):
        return self.minimize(crystal)

    @classmethod
    def from_axis_and_dma_files(cls, axis_filename, dma_filename, **kwargs):
        from cspy.chem.multipole import DistributedMultipoles

        axis = Path(axis_filename).read_text()
        mults = DistributedMultipoles.from_dma_file(dma_filename)
        return cls(mults, axis, **kwargs)
