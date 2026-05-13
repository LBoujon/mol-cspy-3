from ase.calculators.calculator import Calculator
from ase import Atoms
from typing import List
import logging
import numpy as np

LOG = logging.getLogger(__name__)

calculator_units = {
    'energy': 'eV',
    'length': 'Ang',
    'normalisation': 'atom',
    'energy_corr': 'lattice_energy'
}


def setup_custom_calculator(**kwargs) -> Calculator:
    """
    Pass kwargs to set up calculator.

    The development calculator doesn't need any.
    """
    calculator = dev_calculator()
    return calculator
    

class dev_calculator(Calculator):
    """
    Minimalist calculator for development purposes.
    Sets energy of system to be 1 eV per atom.
    """

    implemented_properties = ("energy")

    def __init__(self):
        """
        Initialize the development calculator.
        """
        LOG.info("Initialising development calculator.")
        self.results = {}

    def calculate(self, atoms: Atoms, properties: List[str]) -> float:
        """
        Calculate energy and atomic forces of atoms object.

        Args:
            atoms (Atoms): ASE Atoms object containing structure information.
            properties (list): List of properties to calculate.

        Returns:
            float: Total energy.
        """
        LOG.debug("Executing calculate function in development calculator.")
        return self.get_potential_energy(atoms)

    def get_potential_energy(self, atoms: Atoms) -> float:
        """
        Calculate (potential) energy of atoms object.

        Args:
            atoms (Atoms): ASE Atoms object containing structure information.

        Returns:
            float: Total energy.
        """
        self.atoms = atoms
        LOG.debug(
            "Development (dummy) calculator setting energy to the sum of the distances between atoms (ignoring PBC)."
        )
        energy_per_cell = np.sum(atoms.get_all_distances())
        energy = energy_per_cell / len(atoms)
        self.results['energy'] = energy
        return energy
