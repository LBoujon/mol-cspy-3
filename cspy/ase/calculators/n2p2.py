import logging
from copy import deepcopy
from typing import Optional, Union, List, Literal

import numpy as np
import pynnp
from ase import Atoms
from ase.calculators.calculator import Calculator

from cspy.ml.n2p2_ase_calculator.n2p2_tools import (
    CNN,
    stress_tensor,
    n2p2_committee_calculator,
)
from cspy.ml.nnp.snapshot import Snapshot
from cspy.ml.nnp.atom import Atom, atoms_dict
from cspy.potentials.NNPots import available_n2p2_potentials
from copy import deepcopy
from typing import Union, Optional

LOG = logging.getLogger(__name__)

calculator_units = {'energy' : 'eV',
                    'length' : 'Ang',
                    'normalisation' : 'cell',
                    'energy_corr' : 'lattice_energy'}


def setup_custom_calculator(**kwargs) -> Calculator:
    """
    Pass kwargs to set up calculator.

    Returns:
        Calculator: An instance of N2P2ASECalculator.
    """
    from cspy.ml.n2p2_ase_calculator.n2p2_tools import CNN

    if kwargs.get("n2p2_model"):
        model_name = kwargs.get("n2p2_model")
        input_nn = available_n2p2_potentials[model_name]["n2p2_inputnn_filepath"]
        scaling_data = available_n2p2_potentials[model_name]["n2p2_scaling_filepath"]
        top_folder = available_n2p2_potentials[model_name]["n2p2_committee_dir"]
    else:
        input_nn = kwargs.get('n2p2_inputnn_filepath')
        scaling_data = kwargs.get('n2p2_scaling_filepath')
        top_folder = kwargs.get('n2p2_committee_dir')

    cnn = CNN(
        input_nn=input_nn,
        scaling_data=scaling_data,
        top_folder=top_folder,
        silent=kwargs.get("silent", False),
        )

    damp_forces = kwargs.get('damp_forces', False)

    calculator = N2P2ASECalculator(
        cnn,
        restoring_forces=None,
        damp_forces=damp_forces
    )

    return calculator


def ase_atoms_to_n2p2(atoms: Atoms, 
                      output_format: Literal["pynnp", "snapshot"] = "pynnp", 
                      elementmap: Union[None, pynnp.ElementMap] = None
                     ) -> Union[pynnp.Structure, Snapshot]:
    """
    A tool to convert ASE Atoms object to either pynnp.Structure or Snapshot objects.
    Both formats are n2p2 data formats.

    Args:
        atoms: ASE Atoms object
            input coordinates.
        output_format: str
            The structure can be converted to pynnp.Structure format ('pynnp') or
            custom Snapshop object ('snapshot')
        elementmap: pynnp.ElementMap object
            When output_format=='pynnp', an elementmap is required to add atoms to the structure.
            elementmap can be setup of a model = pynnp.Mode() with model.setupElementMap()

    Returns:
        strucutre: pynnp.Structure or Snapshot object

    """
    if atoms.pbc.any():
        lattice_vectors = np.array([atoms.cell[i] for i in range(3)])
    else:
        lattice_vectors = None
    atoms_pos = atoms.positions
    try:
        elements = [atoms_dict[atomic_number] for atomic_number in atoms.numbers]
    except KeyError as e:
        LOG.error(f"Element not found in NNP. Cannot convert from ase to n2p2")
        LOG.error(f"Input atoms.numbers: {atoms.numbers}")

    # converting to Custom n2p2 format
    if output_format == 'pynnp':
        structure = pynnp.Structure()
        structure.setElementMap(elementmap)
        for i, pos in enumerate(atoms_pos):
            x, y, z = pos
            temp_atom = pynnp.Atom()
            temp_atom.r = pynnp.Vec3D(x=x, y=y, z=z)
            structure.addAtom(temp_atom, elements[i])
        for i, lv in enumerate(lattice_vectors):
            x, y, z = lv
            structure.box[i].r = pynnp.Vec3D(x=x, y=y, z=z)

        structure.isPeriodic = atoms.pbc.any()
        structure.pbc = [int(p) for p in atoms.pbc]
        return structure
    elif output_format == 'snapshot':
        atom_list = []
        for i, pos in enumerate(atoms_pos):
            atom_list.append(
                Atom(
                    element=elements[i],
                    position=pos,
                    force=[0, 0, 0],
                    charge=0
                )
            )
        structure = Snapshot(
            label=atoms.info.get('name', 'from ASE.atoms'),
            atoms=atom_list,
            energy=0,
            lattice_vectors=lattice_vectors
        )
        return structure
    else:
        raise ValueError(
            f"Unknown output_format: {output_format}. "
            "Supported formats are 'pynnp' and 'snapshot'."
        )


class N2P2ASECalculator(Calculator):
    """
    Basic ASE Calculator based on the N2P2 package for training and using of Behler Parrinello neural network potentials
    """

    implemented_properties = ("energy", "forces", "stress")
    excluded_properties = ("dipole", "charges", "magmom", "magmoms")

    def __init__(self, cnn: CNN, restoring_forces: Optional[Calculator] = None, damp_forces: bool = False):
        """
        Initialize N2P2ASECalculator.

        Args:
            cnn (CNN): Committee neural network model.
            restoring_forces (Optional[Calculator]): Calculator used for adding restoring forces
                to cNNP forces to avoid unphysical structures.
            damp_forces (bool): If True, damp forces when standard deviation of predictions is high.
                Useful for geometry optimization to keep the system within the model's applicability domain.
                The forces are damped by a factor of 1 / (1 + exp(max_std_f / 0.1)), where max_std_f is
                the maximum standard deviation of forces predicted by committee members.
                It is quite arbitrary, but it might be useful for some systems.
        """
        super().__init__()
        self.parameters = {
            'version': 'v2.2.0'
        }
        self._directory = None
        self.cnn = cnn
        self.results = {}
        self.restoring_forces = restoring_forces
        self.damp_forces = damp_forces
        LOG.info("Initialising N2P2ASECalculator with CNN model: %s", cnn.top_folder)

    def calculate(self, atoms: Atoms, properties: List):
        """
        Calculates energy and atomic forces of atoms object.

        Args:
            atoms (ase.Atoms): Structure information in the form of ASE atoms object.

        Returns:
            None: Sets self.results dictionary with keys:
                energy (float): Total energy.
                forces (np.ndarray): Array of atomic forces.
                std_of_energies (float): Standard deviation of energies predicted by committee members.
                std_of_forces (float): Average standard deviation of forces predicted by committee members.
        """
        if any(prop in properties for prop in ("energy", "forces")):
            self.get_forces(atoms)
        if "stress" in properties:
            self.get_stress(atoms)
        LOG.debug("Results of N2P2ASECalculator: %s", self.results)

    def get_forces(self, atoms: Atoms) -> np.ndarray:
        """
        Calculates atomic forces of atoms object.
        Args:
            atoms: ASE Atoms object
                Structure information in the form of ASE atoms object

        Returns:
            forces: np.Array
                An array of atomic forces

        """
        structure = ase_atoms_to_n2p2(
            atoms,
            elementmap=self.cnn.models[0].elementMap
        )
        energy, forces, std_of_energies, std_of_forces = n2p2_committee_calculator(
            self.cnn,
            structure,
            calculate_forces=True
        )
        if self.cnn.stats_file:
            with open(self.cnn.stats_file, 'a') as f:
                f.write(
                    f'{energy:>8.4f}  '
                    f'{np.max(np.linalg.norm(forces, axis=1)):>8.4f}  '
                    f'{std_of_energies:>8.4f}  '
                    f'{np.max(std_of_forces):>8.4f}\n'
                )

        self.results['energy'] = energy

        if self.damp_forces:
            max_std_f = np.max(std_of_forces)
            forces *= 1 / (1 + np.exp(max_std_f / 0.1))

        if self.restoring_forces:
            # Since pynnp is not pure python, when atoms.calc is set to N2P2ASECalculator(), deepcopy
            # cannot be used. For this reason, I have to backup the calculator and set atoms.calc to
            # None before deepcopying the atoms object. In the last line, I will restore the calculator.
            current_calc = atoms.calc
            atoms.calc = None

            atoms_temp = deepcopy(atoms)

            atoms_temp.calc = self.restoring_forces
            restoring_forces = atoms_temp.get_forces()
            restoring_forces[np.where(restoring_forces > 1)] = 0
            forces += restoring_forces

            # restoring the calculator
            atoms.calc = current_calc

        self.results['forces'] = forces
        self.results['std_of_energies'] = std_of_energies
        self.results['std_of_forces'] = std_of_forces
        LOG.debug("Forces calculated: %s, standard deviation: %s", forces, std_of_forces)
        if self.restoring_forces:
            LOG.debug("Restoring forces added: %s", restoring_forces)
        return forces

    def get_potential_energy(self, atoms: Atoms) -> float:
        """
        Calculates (potential) energy of atoms object.

        Args:
            atoms (ase.Atoms): Structure information in the form of ASE atoms object.

        Returns:
            float: Total energy.

        Also stores the results in self.results dictionary with keys:
            energy (float): Total energy.
            forces (np.ndarray): Array of atomic forces.
            std_of_energies (float): Standard deviation of energies predicted by committee members.
            std_of_forces (float): Average standard deviation of forces predicted by committee members.
        """
        structure = ase_atoms_to_n2p2(
            atoms,
            elementmap=self.cnn.models[0].elementMap
        )
        energy, forces, std_of_energies, std_of_forces = n2p2_committee_calculator(
            self.cnn,
            structure,
            calculate_forces=False
        )
        self.results['energy'] = energy
        self.results['forces'] = forces
        self.results['std_of_energies'] = std_of_energies
        self.results['std_of_forces'] = std_of_forces
        LOG.debug("Energy calculated: %s, standard deviation: %s", energy, std_of_energies)
        return energy

    def get_stress(
        self,
        atoms: Atoms,
        dh: float = 0.01,
        conversion_factor: float = 1,  # Set to 1 because Calculator.calculate_numerical_stress() does not include unit conversion.
        voigt: bool = True
    ) -> np.ndarray:
        structure = ase_atoms_to_n2p2(
            atoms,
            elementmap=self.cnn.models[0].elementMap
        )
        stress_ = stress_tensor(
            self.cnn,
            structure,
            dh=dh,
            conversion_factor=conversion_factor
        )
        if voigt:
            stress_ = stress_.flat[[0, 4, 8, 5, 2, 1]]
        self.results['stress'] = stress_
        LOG.debug("Stress calculated: %s", stress_)
        return stress_
