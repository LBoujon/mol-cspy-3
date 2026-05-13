import re
import logging
import math
import numpy as np
from typing import Tuple

LOG = logging.getLogger(__name__)

class DmacrysOutputException(Exception):
    """Generic exception for Dmacrys output failures.

    """


class DmacrysOutput:

    def __init__(self, contents):
        """Dmacrys output class for parsing the dmacrys standard output
        file and converting data into a more useful form.

        Parameters
        ----------
        contents : string
            string containing the contents of the dmacrys standard out

        """
        self.contents = contents

    def dmaout_eigenvectors(self, crystal):
        """Parse the dmacrys output file, obtain phonon eigenvalues
        straight from the dmacrys output.

        Parameters
        ----------
        crystal : cspy.Crystal
            crystal structure obtained from the dmacrys phonon
            calculation

        Returns
        -------
        eigenvectors : numpy.array
            matrix of phonon eigenvectors

        """
        molecules = crystal.unit_cell_molecules()
        phonon_section = re.findall(
            "Zone Centre phonon eigenvectors(.*)Zone Centre Phonon Frequencies",
            self.contents, flags=re.DOTALL
        )[0]
        phonon_section = re.findall(
            " +[0-9]*" + (" +(-?[0-9]+[.][0-9]+)" * 6 + " *\n") * len(molecules),
            phonon_section
        )

        eigenvectors = []
        for eigenvector in phonon_section:
            eigenvectors.append([float(i) for i in eigenvector])

        return np.array(eigenvectors)

    def atom_forces(self, crystal: "Crystal") -> np.ndarray:
        """Parse dmaout file atomic forces and reorder to original atom ordering

        Parameters
        ----------
        crystal : cspy.Crystal
            input crystal structure
        """
        from collections import defaultdict
        
        # get all forces sections in dmacrys output (seperated by molecule)
        forces_sections = re.findall(
            "Centralisation of forces \(global axes\)\\n"
            "and torques \(global axes\) for molecule:\s*(.*?)Central force",
            self.contents, 
            flags=re.DOTALL
        )

        # attribute each section to the corresponding molecule number
        # which is the first char in the section
        # this overwrites earlier sections, hence will take the last section 
        # for each molecule (i.e. works for single points and optimizations)
        forces_sections = {
            int(x.split('\n')[0]): '\n'.join(x.split('\n')[1:])
            for x in forces_sections
        }

        # parse atom forces in each section/molecule
        all_atom_forces = []
        labels = []
        for mol_idx, forces_content in forces_sections.items():
            atom_forces = re.findall(
                "([0-9]  .{10}" + "\s*-?[0-9]+[.][0-9]+E[-+][0-9]*" * 4 + ")\n",
                forces_content,
            )

            for atom_forces_line in atom_forces:
                contents = atom_forces_line.split()
                label = contents[1]
                # symmetry equiv molecules will have the same label
                # need to also handle inverted molecules, which have I in the label
                # NOTE: dmacrys "forces" are gradients hence taking negative
                if label not in labels and label[4] != "I": 
                    labels.append(label)
                    all_atom_forces.append([ -float(x) for x in contents[2:-1]])

        # dmacrys reorders atoms by element type, with elements in order occur in input
        # change this back by finding the index of each atom in its element type in the 
        # original input 
        elem_ordering = defaultdict(list)
        for i, elem in enumerate(crystal.asymmetric_unit.elements):
            elem_ordering[elem.symbol].append(i)
        
        ordering_to_cspy = {}
        for i, label in enumerate(labels):
            elem = label[:2].rstrip('_')
            idx = int(label[5:].rstrip('_')) - 1
            ordering_to_cspy[i] = elem_ordering[elem][idx]
        
        ordering_to_cspy = [ k for k, _ in sorted(ordering_to_cspy.items(), key=lambda x: x[1]) ]
        
        all_atom_forces = np.array(all_atom_forces)
        all_atom_forces = all_atom_forces[ordering_to_cspy]

        rotated_back_all_atom_forces = []
        for v in all_atom_forces:
            rotated_back_all_atom_forces.append(crystal.dmacry_vector_to_cspy_vector(v))
        return np.array(rotated_back_all_atom_forces)

    def stress_tensor(
        self, fixed_cell: bool = False, scaling_factor: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get stress tensor from dmacrys stdout.

        Parameters
        ----------
        fixed_cell : bool
            If fixed cell (CONV) calculation.
        scaling_factor : float
            Factor to scale stress tensor components.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Returns stress tensor in eV and kJ/mol in Voigt ordering (XX YY ZZ YZ XZ XY).
        """

        if not fixed_cell:
            # if variable cell (i.e. CONP) get strain matrix from properties section
            strain_section = re.findall(
                "Strain matrix to be applied:\\n"
                "(.*?)"
                "\\n Final lattice vectors are",
                self.contents,
                flags=re.DOTALL,
            )

            strain_section = strain_section[-1] # get last strain section in case of opt

            stress_tensor = re.findall(
                #"(" + "\s*-?[0-9]+[.][0-9]*" * 6 + ")\n",
                "\s*(-?[0-9]+[.][0-9]*)",
                strain_section,
            )

            stress_tensor_eV = np.array([ float(x) for x in stress_tensor[:6] ])
            stress_tensor_kjmol = stress_tensor_eV * 96.48530749925793

        else:
            # if fixed cell (i.e. CONV) get first strain matrix (the properties 
            # strain matrix will be all 0 for fixed cell)
            LOG.warning("fixed cell (CONV) strain matrix has dubious validity")
            
            strain_section = re.findall(
                "Strain derivatives for matrix deformation of form: \\n"
                "(.*?)"
                "\\n Units for strain derivatives",
                self.contents,
                flags=re.DOTALL,
            )

            strain_section = strain_section[-1] # get last strain section in case of opt

            stress_tensor = re.findall(
                #"(" + "\s*-?[0-9]+[.][0-9]*" * 6 + ")\n",
                "\s*(-?[0-9]+[.][0-9]*)",
                strain_section,
            )

            # first 6 floats are the eV stress tensor, next 6 are kjmol
            stress_tensor_eV = scaling_factor * np.array(
                [ float(x) for x in stress_tensor[:6] ]
            ) 
            stress_tensor_kjmol = scaling_factor * np.array(
                [ float(x) for x in stress_tensor[6:] ]
            )
        
        return (stress_tensor_eV, stress_tensor_kjmol)
        

    def phonon_eigenvectors(self, crystal):
        """Returns eigenvectors with reordered coordinates of the
        molecules so that it is the same order as the molecule
        orderings in cspy.Crystal.

        Parameters
        ----------
        crystal : cspy.Crystal
            crystal structure obtained from the dmacrys phonon
            calculation

        Returns
        -------
        eigenvectors : numpy.array
            matrix of phonon eigenvectors

        """
        return self.dmaout_eigenvectors(crystal) @ self.reordering_matrix(crystal)

    def mass_and_moments(self):
        """Returns the masses and moments of inertia for each molecule.
        Masses are are duplicated three times for each xyz coordinate.

        Returns
        -------
        mass_and_moments : list
            an ordered list of masses and moments of inertia

        """
        mass_section = re.findall(
            " *Molecule No[.] +Centred at +Total Mass(.*)Molecule No[.] "
            "+Principal axes of inertia",
            self.contents, flags=re.DOTALL
        )[0]
        mass_section = re.findall(
            " +[0-9]+" + " +-?[0-9]+[.][0-9]+" * 3 + " +([0-9]+[.][0-9]+) *",
            mass_section
        )
        inertia_section = re.findall(
            "Molecule No[.] +Principal axes of inertia(.*)Maximum "
            "intramolecular distance",
            self.contents, flags=re.DOTALL
        )[0]
        inertia_section = re.findall(
            " +Ix = +([0-9]+[.][0-9]+E?[-+]?[0-9]*), "
            "Iy = +([0-9]+[.][0-9]+E?[-+]?[0-9]*), "
            "Iz = +([0-9]+[.][0-9]+E?[-+]?[0-9]*) *",
            inertia_section
        )

        mass_and_moments = []
        for i, moments in enumerate(inertia_section):
            mass_and_moments += [float(mass_section[i])] * 3
            mass_and_moments += [float(i) for i in moments]
        return np.array(mass_and_moments)

    def transform_cspy_global(self, crystal):
        """Generates a matrix that transforms the translation and
        infinitesimal rotations coordinates of the phonon ordered
        eigenvectors from dmacrys global axis for translation and
        local principle axes for rotations to cspys global axes frame.

        Parameters
        ----------
        crystal : cspy.Crystal
            crystal structure obtained from the dmacrys phonon
            calculation

        Returns
        -------
        numpy.array
            numpy array of the transformation matrix that transforms
            the infinitesimal rotation of the phonon eigenvectors to
            the global axes frame

        """
        axis_section = re.findall(
            " +Final lattice vectors are(.*) +Final basis positions are",
            self.contents, flags=re.DOTALL
        )[0]
        axis = np.array(re.findall(
            (" +(-?[0-9]+[.][0-9]+)" * 3 + " *\n") * 3, axis_section
        )[0]).reshape(3, 3).astype(float)
        dmacrys_direct = crystal.unit_cell.c * axis
        dmacrys_global_to_cspy_global \
            = np.linalg.inv(dmacrys_direct) @ crystal.unit_cell.direct

        dmacrys_local_to_cspy_global = []
        inertia_section = re.findall(
            "Molecule No[.] +Principal axes of inertia(.*)Maximum "
            "intramolecular distance",
            self.contents, flags=re.DOTALL
        )[0]
        inertia_section = re.findall(
            " +[0-9]+" + (" +(-?[0-9]+[.][0-9]+E?[-+]?[0-9]*)" * 3 + " *\n") * 3,
            inertia_section
        )

        molecules = crystal.unit_cell_molecules()
        for i, inertia_tensor in enumerate(inertia_section):
            dmacrys_local_to_cspy_global.append(
                np.array(inertia_tensor).reshape(3, 3).astype(float)
                @ molecules[i].axes()
            )

        transform_bmat = []
        for i, transform_mat in enumerate(dmacrys_local_to_cspy_global):
            bmat_row = []
            for j in range(len(dmacrys_local_to_cspy_global)):
                if i == j:
                    bmat_row.append(np.block([
                        [dmacrys_global_to_cspy_global, np.zeros((3, 3))],
                        [np.zeros((3, 3)), transform_mat]
                    ]))
                else:
                    bmat_row.append(np.zeros((6, 6)))
            transform_bmat.append(bmat_row)

        return np.block(transform_bmat) @ self.reordering_matrix(crystal)

    def reordering_matrix(self, crystal: "Crystal") -> np.ndarray:
        """Match molecule orderings between dmacrys and CSPy to ensure
        each part of the eigenvector refer to the same molecule in
        both ordering systems.

        Parameters
        ----------
        crystal : cspy.Crystal
            crystal structure obtained from the dmacrys phonon
            calculation

        Returns
        -------
        reorder_matrix : numpy.array
            numpy array of a matrix that reorders the eigenvectors so
            that each part refers to molecules in the same order that
            CSPy molecules are listed in

        """
        com_section = re.findall(
            " *Final crystalographic positions of molecule centres of "
            "mass(.*)Direction cosines ",
            self.contents, flags=re.DOTALL
        )[0]
        com_section = re.findall(
            " +[0-9]+" + " +(-?[0-9]+[.][0-9]+)" * 3 + " *",
            com_section
        )

        molecules = crystal.unit_cell_molecules()
        reorder_matrix = []
        for com_i in com_section:
            com_i = np.array([float(i) for i in com_i])

            reorder_row = []
            for mol_j in molecules:

                com_ij = crystal.to_fractional(mol_j.center_of_mass) - com_i
                smallest_diff = crystal.to_cartesian(np.array([
                    k - math.floor(k)
                    if abs(k - math.floor(k)) < abs(k - math.ceil(k))
                    else k - math.ceil(k) for k in com_ij
                ]))

                if all([abs(k) < 1e-2 for k in smallest_diff]):
                    reorder_row.append(np.identity(6))
                else:
                    reorder_row.append(np.zeros((6, 6)))
            reorder_matrix.append(reorder_row)

        reorder_matrix = np.block(reorder_matrix)

        if np.sum(reorder_matrix) != 6 * len(molecules):
            raise DmacrysOutputException(
                "Failure to match molecules from dmacrys to CSPy could "
                "be that more than one molecules from dmacrys was "
                "matched to CSPy or that a molecules failed to match "
                "with any from dmacrys for some reason."
            )

        if np.trace(reorder_matrix) != 6 * len(molecules):
            LOG.warning(
                "Different order in molecules from dmacrys and CSPy "
                "was found, phonon eigenvectors will be reordered."
            )

        return reorder_matrix
