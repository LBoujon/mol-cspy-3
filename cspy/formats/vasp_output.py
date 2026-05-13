import logging
from cspy.crystal import AsymmetricUnit
from cspy import Crystal
import numpy as np
from cspy.formats.vasp_input import make_poscar
from typing import List, Union, Dict, Literal

LOG = logging.getLogger(__name__)


def recover_ordering_from_poscar(crystal: Crystal, poscar: str) -> List[int]:
    """
    Recover ordering from POSCAR file corresponding to crystal.

    Args:
        crystal: Crystal object
        poscar: str. POSCAR file contents

    Returns:
        List[int]: List of ordering indices. The length of the list is equal to the length of
        crystal.asymmetric_unit. For P1 cells, it returns ordering for all atoms.
    """
    new_crystal = Crystal.from_CONTCAR(poscar)
    new_crystal_ordering = [
        np.where(new_crystal.site_labels == x)[0][0] for x in crystal.site_labels
    ]
    return new_crystal_ordering


def recover_ordering_from_poscar_str(
    crystal: Crystal, poscar_string: str
) -> List[int]:
    """
    Recover ordering from POSCAR file corresponding to crystal.

    Args:
        crystal: Crystal object
        poscar_string: str. Contents of POSCAR file.

    Returns:
        List[int]: List of ordering indices. The length of the list is equal to the
        length of crystal.asymmetric_unit. For P1 cells, it returns ordering for all atoms.
    """
    new_crystal = Crystal.from_CONTCAR_string(poscar_string)
    new_crystal_ordering = [
        np.where(new_crystal.site_labels == x)[0][0]
        for x in crystal.site_labels
    ]
    return new_crystal_ordering


def contcar_to_ordered_crystal(
    crystal: Crystal,
    CONTCAR: str,
    is_fractional: bool = True,
    enforce_p1: bool = True
) -> Crystal:
    """
    Takes a Crystal object, generates a POSCAR corresponding to it, recovers the ordering based on the POSCAR, and uses
    the ordering to convert CONTCAR (relaxed POSCAR) to Crystal object with correct ordering.

    Args:
        crystal: Crystal object. Starting geometry.
        CONTCAR: str. CONTCAR file corresponding to geometry optimised POSCAR.
        is_fractional: bool. Whether the CONTCAR is in fractional coordinates or not.
        enforce_p1: Boolean. CONTCAR is in P1, but it is believed that it has the same symmetry of crystal, set this
            value to False to recover the same symmetry as Crystal.

    Returns:
        Crystal object. Crystal object of the relaxed geometry with correct ordering.
    """
    if enforce_p1 and crystal.space_group.international_tables_number != 1:
        LOG.info(
            f'While converting {CONTCAR} to Crystal object, {crystal.titl} was not P1. Converting it to P1.'
        )
        crystal = crystal.as_P1()
    poscar_string = "\n".join(
        make_poscar(
            crystal,
            elem_set=set(crystal.asymmetric_unit.elements),
            comment=f'{crystal.titl}'
        )
    )
    new_crystal_ordering = recover_ordering_from_poscar_str(crystal, poscar_string)
    new_crystal = Crystal.from_CONTCAR(CONTCAR)

    if not is_fractional:
        new_crystal.asymmetric_unit.positions = new_crystal.to_fractional(
            new_crystal.asymmetric_unit.positions
        )

    new_asymmetric_unit = AsymmetricUnit(
        elements=[new_crystal.asymmetric_unit.elements[i] for i in new_crystal_ordering],
        positions=[new_crystal.asymmetric_unit.positions[i] for i in new_crystal_ordering]
    )
    new_crystal.asymmetric_unit = new_asymmetric_unit
    return new_crystal


class VaspOutput():
    def __init__(self, contents: str, **kwargs):
        self.file_contents = contents.split("\n")

    def _valid(self, job_type: str) -> bool:
        """
        Check if the VASP output file is valid for the given job type.

        Args:
            job_type (str): The type of job, either "minimize" or "single_point".

        Returns:
            bool: True if the file is valid for the job type, False otherwise.
        """
        if job_type not in ["minimize", "single_point"]:
            raise ValueError(
                "Invalid job type: {}. Expected 'minimize' or 'single_point'.".format(job_type)
            )
        valid_check_map = {
            "minimize": self._valid_minimization,
            "single_point": self._valid_singlepoint,
        }
        return valid_check_map[job_type]()

    def _valid_singlepoint(self) -> bool:
        return (
            "------------------------ aborting loop because EDIFF is reached ----------------------------------------" 
            in self.file_contents
        )

    def _valid_minimization(self) -> bool:
        return (
            "reached required accuracy - stopping structural energy minimisation"
            in self.file_contents
        )

    def _first_match(self, pattern: str, contents: List, split_index: int, start: Literal["start", "bottom"]) -> Union[str, None]:
        """
        Find the first occurrence of a pattern in the contents and return the specified part.

        Args:
            pattern (str): The pattern to search for.
            contents (List): The list of strings to search in.
            split_index (int): The index of the part to return after splitting the matched string.
            start (str): Where to start searching from, either 'top' or 'bottom'.
        
        Returns:
            Union[str, None]: The specified part of the matched string or None if not found.
        """
        if start == 'top':
            match = next((item for item in contents if pattern in item), None)
            if match:
                match = match.split()[split_index]
        elif start == 'bottom':
            match = next((item for item in reversed(contents) if pattern in item), None)
            if match:
                match = match.split()[split_index]
        else:
            raise ValueError("Invalid start value: {}. Expected 'top' or 'bottom'.".format(start))
        return match
    
    def _parse_forces(self) -> List[List[float]]:
        """
        reads forces section from OUTCAR. The atoms the forces apply to will be the 
        same order as the CONTCAR/POSCAR
        
        Returns:
            List[List[float]]: A list of forces for each atom in the system.
            Forces are in eV/Angst (in cartesian)
        """
        forces = []
        contents = iter(self.file_contents)
        for line in contents:
            # read the force section
            if "POSITION" in line and "TOTAL-FORCE" in line:
                next(contents)
                # write data into first sample in the data set (assuming having only one sample)
                while True:
                    line = next(contents)
                    parts = line.split()
                    if len(parts) != 6:
                        break
                    forces.append([float(frc) for frc in parts[3:]])
        
        return forces

    def _parse_ion_forces(self) -> Dict[str, List[List[float]]]:
        """
        Reads "forces acting on ions" section from OUTCAR.
        The atoms the forces apply to will be in the same order as the CONTCAR/POSCAR.

        Forces are in eV/Angstrom (in cartesian).

        Returns:
            Dict[str, List[List[float]]]: A dictionary with force contribution keys:
                "electron-ion", "ewald-force", "non-local-force", "convergence-correction".
        """
        forces = {
            "electron-ion": [],
            "ewald-force": [],
            "non-local-force": [],
            "convergence-correction": [],
        }
        contents = iter(self.file_contents)
        for line in contents:
            # read the forces section
            if "ewald-force" in line:
                next(contents)
                # write data into first sample in the data set (assuming having only one sample)
                while True:
                    line = next(contents)
                    parts = line.split()
                    if len(parts) != 12:
                        break
                    forces["electron-ion"].append([float(frc) for frc in parts[:3]])
                    forces["ewald-force"].append([float(frc) for frc in parts[3:6]])
                    forces["non-local-force"].append([float(frc) for frc in parts[6:9]])
                    forces["convergence-correction"].append([float(frc) for frc in parts[9:]])
        return forces

    def _parse_file(self) -> Dict[str, Union[str, List[List[float]]]]:
        expressions = {
            'initial_energy': ['free  energy   TOTEN', 4, 'top'],
            'final_energy': ['free  energy   TOTEN', 4, 'bottom'],
        }
        return {
            key: self._first_match(
                value[0],
                self.file_contents,
                value[1],
                value[2]
            )
            for key, value in expressions.items()
        }




