import numpy as np
import pynnp
from pathlib import Path
from cspy.ml.nnp.snapshot import Snapshot
from cspy.ml.nnp.atom import Atom
from copy import copy
from os import remove
from cspy.util.constants import EV_ANGSTROM3_TO_GPA
import logging
from typing import List, Tuple, Union

LOG = logging.getLogger(__name__)


class CNN:
    """
    Committee Neural Network configuration class.
    """
    def __init__(
        self,
        input_nn: str,
        scaling_data: str,
        top_folder: str,
        n_members: int = 1,
        stats_file: Union[str, None] = None,
        **kwargs
    ):
        """
        Initialize the Committee Neural Network configuration.

        Args:
            input_nn (str): Path to input.nn file containing settings of NN, shared by all committee members.
            scaling_data (str): Path to scaling.data file, shared by all members.
            top_folder (str): Path to the top folder containing members (usually NN??) sub-folders which contain weights.?????.data.
            n_members (int, optional): Number of members in the committee. Default is 1.
            stats_file (str, optional): Path to the stats file for collecting statistics about the committee members.
                If None, no statistics will be collected. Default is None.
            **kwargs: Additional keyword arguments to pass to the setup_models method.
                For example, `silent=True` to suppress warnings about extrapolation.
        """
        self.input_nn = input_nn
        self.scaling_data = scaling_data
        self.top_folder = top_folder
        self.n_members = n_members
        self.models = None
        self.setup_models(**kwargs)
        self.stats_file = stats_file

    def setup_models(self, **kwargs) -> None:
        """
        Sets up the models for the committee neural network using pynnp.

        Args:
            **kwargs: Additional keyword arguments, such as `silent` to suppress warnings.
        """
        import os
        LOG.debug(f'Loading models from {self.top_folder}')
        self.get_member_dirs()
        cwd = Path.cwd()
        self.models = []
        for member_dir in self.member_dirs:
            os.chdir(member_dir)
            LOG.debug(f'Loading model from {member_dir} = {Path.cwd()}')
            m_temp = pynnp.Mode()
            m_temp.log.writeToStdout = False
            m_temp.initialize()
            m_temp.loadSettingsFile(self.input_nn)
            m_temp.setupNormalization()
            m_temp.setupElementMap()
            m_temp.setupElements()
            m_temp.setupCutoff()
            m_temp.setupSymmetryFunctions()
            m_temp.setupSymmetryFunctionMemory()  # comment out when compiled with N2P2_FULL_SFD_MEMORY
            m_temp.setupSymmetryFunctionCache()   # comment out when compiled with N2P2_NO_SF_CACHE
            m_temp.setupSymmetryFunctionGroups()
            m_temp.setupNeuralNetwork()
            m_temp.setupSymmetryFunctionScaling(self.scaling_data)
            m_temp.setupSymmetryFunctionStatistics(
                collectStatistics=False,
                collectExtrapolationWarnings=True,
                writeExtrapolationWarnings=False if kwargs.get('silent', False) else True,
                stopOnExtrapolationWarnings=False
            )
            m_temp.setupNeuralNetworkWeights()
            # Older versions of n2p2 (before Nov 2024) work with the following line
            # m_temp.setupNeuralNetworkWeights(f'{member_dir}/weights.%03zu.data')
            self.models.append(m_temp)
            os.chdir(cwd)
        LOG.info(f'Successfully loaded {len(self.models)} models')

    def get_member_dirs(self) -> List[Path]:
        """
        Finds all member directories in the top folder that contain weights files.

        Returns:
            List[Path]: A list of unique member directories.
        """
        all_weights_paths = Path(self.top_folder).glob('*/weights.*.data')
        member_dirs = set()
        for weight_path in all_weights_paths:
            member_dirs.add(weight_path.parent)
        self.member_dirs = list(member_dirs)
        self.n_members = len(self.member_dirs)
        LOG.debug(f'Found {self.n_members} members: {self.member_dirs}')
        return self.member_dirs


def input_data_reader(fname: str) -> Snapshot:
    """
    Reads a coordinates file in n2p2 (input.data) format and returns a Snapshot object.

    Args:
        fname (str): Path to the coordinates file in n2p2 (input.data) format.

    Returns:
        Snapshot: Snapshot object containing the parsed data.
    """
    with open(fname, 'r') as f:
        lattice_vectors = []
        atoms = []
        label = 'no_label'
        energy = None
        for line in f:
            parts = line.split()
            if not parts:
                continue

            if parts[0] == 'atom':
                atoms.append(
                    Atom(
                        parts[4],  # element
                        np.array([float(parts[1]), float(parts[2]), float(parts[3])]),  # pos
                        np.array([float(parts[7]), float(parts[8]), float(parts[9])]),  # force
                        float(parts[5]),  # charge
                    )
                )
            elif parts[0] == 'lattice':
                lattice_vectors.append(
                    np.array([
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3])
                    ])
                )
            elif parts[0] == 'comment':
                label = parts[1] if len(parts) > 1 else 'no_label'
            elif parts[0] == 'energy':
                energy = float(parts[1])
            elif parts[0] == 'end':
                # Save snapshot and reset the lists
                snapshot = Snapshot(
                    label=label,
                    atoms=atoms,
                    energy=energy,
                    lattice_vectors=lattice_vectors
                )
                return snapshot
    return None


# def n2p2_single_nnp_old(
#     model: pynnp.Model,
#     structure: pynnp.Structure,
#     calculate_forces: bool = False
# ) -> Tuple[float, List[np.array]]:
#     """
#     Calculates energies and forces of Structure using a single NN model.

#     Args:
#         model: pynnp.Mode object
#             Contains a single NNP.
#         structure: pynnp.Structure object
#             Contains structural information, typically from input.data file.
#         calculate_forces: bool
#             Is the calculation of forces required?

#     Returns:
#         Tuple[float, List[np.array]]: energy, np.array(forces)
#     """

#     # Calculate atomic neural networks.
#     model.calculateAtomicNeuralNetworks(structure, True)

#     # Sum up potential energy.
#     model.calculateEnergy(structure)

#     # Collect force contributions.
#     forces = []
#     if calculate_forces:
#         model.calculateForces(structure)
#         for atom in structure.atoms:
#             forces.append(atom.f.r)

#     # If normalization is used, convert structure data back to physical units.
#     if model.useNormalization():
#         structure.toPhysicalUnits(
#             model.getMeanEnergy(),
#             model.getConvEnergy(),
#             model.getConvLength(),
#             0,  # TODO: add convCharge
#         )
#     model.addEnergyOffset(structure, False)
#     model.addEnergyOffset(structure, True)

#     return structure.energy, forces


def n2p2_committee_calculator_file(
    cnn: CNN,
    input_data: str,
    calculate_forces: bool = False
) -> Tuple[float, List[np.array], float, List[np.array]]:
    """
    Takes a list of models (committee) and returns average energy, average forces,
    standard deviation of energy, standard deviation of forces.

    Args:
        cnn (CNN): Committee neural network configuration class.
        input_data (str): Path to the input.data file.
        calculate_forces (bool): Whether or not forces are needed.

    NOTE:
        Currently will return NaN for forces if data normalization is used.
    """
    committee_energies = []
    committee_forces = []

    structure = pynnp.Structure()
    structure.setElementMap(cnn.models[0].elementMap)
    structure.readFromFile(input_data)

    for i_m, m in enumerate(cnn.models):
        normalized = m.useNormalization()
        # If normalization is used, convert structure data.
        if (calculate_forces and normalized) and i_m > 0:
            structure.reset()
            structure.readFromFile(input_data)
        if normalized:
            structure.toNormalizedUnits(
                m.getMeanEnergy(),
                m.getConvEnergy(),
                m.getConvLength(),
                0,  # TODO: add convCharge
            )

        if i_m == 0 or (calculate_forces and normalized):
            # Retrieve cutoff radius from NNP setup.
            cutoff_radius = m.getMaxCutoffRadius()
            # Calculate neighbor list.
            structure.calculateNeighborList(cutoff_radius)
            # Calculate symmetry functions for all atoms (use groups).
            m.calculateSymmetryFunctionGroups(structure, derivatives=calculate_forces)

        m.calculateAtomicNeuralNetworks(structure, derivatives=calculate_forces)

        # Sum up potential energy.
        m.calculateEnergy(structure)
        if calculate_forces:
            m.calculateForces(structure)

        # If normalization is used, convert structure data back to physical units.
        if normalized:
            structure.toPhysicalUnits(
                m.getMeanEnergy(),
                m.getConvEnergy(),
                m.getConvLength(),
                0,  # TODO: add convCharge
            )
        # m.addEnergyOffset(structure, ref=False) # these lines don't seem to do anything
        # m.addEnergyOffset(structure, ref=True)

        committee_energies.append(structure.energy)

        # Collect force contributions.
        if calculate_forces:
            member_forces = []
            for atom in structure.atoms:
                member_forces.append(atom.f.r)
            committee_forces.append(member_forces)
    energy = np.mean(committee_energies)
    std_of_energies = np.std(committee_energies)

    forces = None
    std_of_forces = None
    if calculate_forces:
        forces = np.mean(committee_forces, axis=0)
        std_of_forces = np.std(committee_forces, axis=0)

    return energy, forces, std_of_energies, std_of_forces

def n2p2_committee_calculator_string(
    cnn: CNN,
    input_string: str,
    calculate_forces: bool
) -> Tuple[float, List[np.array], float, List[np.array]]:
    """
    Takes a list of models (committee) and return average energy, average forces, standard deviation of energy,
    standard deviation of forces.

    Args:
        cnn: CNN object
            Committee neural network configuration class.
        input_string: str
            String of n2p2 input.data.
        calculate_forces: bool
            Whether or not forces are needed.

    NOTE:
        If use normalization in training, will be unable to calculate the symm funcs
        once and instead will have to calculate for each model, which takes longer. 
        For some reason, only affects forces not energy. Also, does not occur if do not
        normalize.

        See n2p2/examples/pynnp/ though it is unclear which way is intended behaviour.
    """
    committee_energies = []
    committee_forces = []
    input_lines = input_string.split('\n')

    structure = pynnp.Structure()
    structure.setElementMap(cnn.models[0].elementMap)
    structure.readFromLines(input_lines)

    for i_m, m in enumerate(cnn.models):
        normalized = m.useNormalization()
        # If normalization is used, convert structure data.
        if (calculate_forces and normalized) and i_m > 0:
            structure.reset()
            structure.readFromLines(input_lines)
        if normalized:
            structure.toNormalizedUnits(
                m.getMeanEnergy(),
                m.getConvEnergy(),
                m.getConvLength(),
                0,  # TODO: add convCharge
            )

        if i_m == 0 or (calculate_forces and normalized):
            # Retrieve cutoff radius from NNP setup.
            cutoff_radius = m.getMaxCutoffRadius()
            # Calculate neighbor list.
            structure.calculateNeighborList(cutoff_radius)
            # Calculate symmetry functions for all atoms (use groups).
            m.calculateSymmetryFunctionGroups(structure, derivatives=calculate_forces)

        m.calculateAtomicNeuralNetworks(structure, derivatives=calculate_forces)

        # Sum up potential energy.
        m.calculateEnergy(structure)
        if calculate_forces:
            m.calculateForces(structure)

        # If normalization is used, convert structure data back to physical units.
        if normalized:
            structure.toPhysicalUnits(
                m.getMeanEnergy(),
                m.getConvEnergy(),
                m.getConvLength(),
                0,  # TODO: add convCharge
            )
        # m.addEnergyOffset(structure, ref=False) # these lines don't seem to do anything
        # m.addEnergyOffset(structure, ref=True)

        committee_energies.append(structure.energy)

        # Collect force contributions.
        if calculate_forces:
            member_forces = []
            for atom in structure.atoms:
                member_forces.append(atom.f.r)
            committee_forces.append(member_forces)
    energy = np.mean(committee_energies)
    energy_std_dev = np.std(committee_energies)

    forces = None
    forces_std_dev = None
    if calculate_forces:
        forces = np.mean(committee_forces, axis=0)
        forces_std_dev = np.std(committee_forces, axis=0)

    structure.reset()
    del structure

    return energy, forces, energy_std_dev, forces_std_dev


def n2p2_committee_calculator(
    cnn: CNN,
    structure: pynnp.Structure,
    calculate_forces: bool = False
) -> Tuple[float, List[np.array], float, List[np.array]]:
    """
    Takes a list of models (committee) and returns average energy, average forces,
    standard deviation of energy, standard deviation of forces.

    Args:
        cnn: CNN object
            Committee neural network configuration class.
        structure: pynnp.Structure
            Structure object with atomic configuration.
        calculate_forces: bool
            Whether or not forces are needed.

    NOTE:
        Currently will return NaN for forces if data normalization is used.
    """
    committee_energies = []
    committee_forces = []
    structure_orig = structure
    structures = [structure_orig for _ in range(len(cnn.models))]

    for i_m, m in enumerate(cnn.models):
        structure = structures[i_m]
        m.removeEnergyOffset(structure)
        # If normalization is used, convert structure data.
        if i_m == 0:
            if m.useNormalization():
                structure.toNormalizedUnits(
                    m.getMeanEnergy(),
                    m.getConvEnergy(),
                    m.getConvLength(),
                    0,  # TODO: add convCharge
                )
            # Retrieve cutoff radius from NNP setup.
            cutoff_radius = m.getMaxCutoffRadius()
            # Calculate neighbor list.
            structure.calculateNeighborList(cutoff_radius)
            # Calculate symmetry functions for all atoms (use groups).
            m.calculateSymmetryFunctionGroups(structure, derivatives=calculate_forces)

        m.calculateAtomicNeuralNetworks(structure, derivatives=calculate_forces)

        # Sum up potential energy.
        m.calculateEnergy(structure)
        if calculate_forces:
            m.calculateForces(structure)

        # If normalization is used, convert structure data back to physical units.
        if m.useNormalization():
            structure.toPhysicalUnits(
                m.getMeanEnergy(),
                m.getConvEnergy(),
                m.getConvLength(),
                0,  # TODO: add convCharge
            )
        # m.addEnergyOffset(structure, ref=False)
        # m.addEnergyOffset(structure, ref=True)

        committee_energies.append(structure.energy)

        # Collect force contributions.
        if calculate_forces:
            member_forces = []
            for atom in structure.atoms:
                member_forces.append(atom.f.r)
            committee_forces.append(member_forces)

        # Redo normalization for next iteration
        if m.useNormalization():
            structure.toNormalizedUnits(
                m.getMeanEnergy(),
                m.getConvEnergy(),
                m.getConvLength(),
                0,  # TODO: add convCharge
            )
        # structure.reset()
        # structure.readFromFile('input.data')

    energy = np.mean(committee_energies)
    std_of_energies = np.std(committee_energies)

    forces = None
    std_of_forces = None
    if calculate_forces:
        forces = np.mean(committee_forces, axis=0)
        std_of_forces = np.std(committee_forces, axis=0)

    return energy, forces, std_of_energies, std_of_forces


def stress_tensor_file(
    cnn: CNN,
    input_data: str,
    dh: float = 0.01,
    conversion_factor: float = EV_ANGSTROM3_TO_GPA
) -> np.array:
    """
    Calculates the numerical stress tensor in GPa.

    Args:
        cnn (CNN): Committee neural network configuration class.
        input_data (str): Lattice and atoms coordinate in input.data format.
        dh (float): Displacement for calculation of numerical derivatives.
        conversion_factor (float): Conversion factor from calculator units to GPa. Default: EV_ANGSTROM3_TO_GPA

    Returns:
        np.array: A 3x3 array of numerical stress tensor in GPa.
    """
    directions = ['x', 'y', 'z']

    structure = input_data_reader(input_data)
    volume = np.linalg.det(structure.lattice_vectors)

    _stress_tensor = np.zeros((3, 3))
    for i in range(3):
        for j in range(i, 3):

            # calculating unit lattice displacement tensor
            lattice_dh = np.zeros((3, 3))
            lattice_dh[i, i] = 1
            lattice_dh[j, j] = 1
            lattice_dh[j, i] = 1
            lattice_dh[i, j] = 1
            lattice_dh /= np.linalg.norm(lattice_dh)

            # calculating unit atoms displacement vector
            atoms_dh = np.zeros(3)
            atoms_dh[i] = 1
            atoms_dh[j] = 1
            atoms_dh /= np.linalg.norm(atoms_dh)

            diff = 0
            for sign in [-1, 1]:
                structure_ = copy(structure)
                dh_ = sign * dh
                lattice_dh_ = lattice_dh * dh_
                atoms_dh_ = atoms_dh * dh_

                structure_.lattice_vectors += lattice_dh_

                for atom in structure_.atoms:
                    atom.position += atoms_dh_
                # m.calculateSymmetryFunctions(structure, False)
                fname = f'input_{directions[i]}{directions[j]}_{dh_:+8.6f}.data'
                structure_.write(fname)
                e_, f_, std_e_, std_f_ = n2p2_committee_calculator_file(
                    cnn,
                    input_data=fname,
                    calculate_forces=False
                )
                diff += sign * e_
                remove(fname)
            _stress_tensor[i, j] = diff / (2 * dh * volume)
            if i != j:
                _stress_tensor[j, i] = _stress_tensor[i, j]
    return _stress_tensor * conversion_factor


def stress_tensor(
    cnn: CNN,
    structure: pynnp.Structure,
    dh: float = 0.001,
    conversion_factor: float = EV_ANGSTROM3_TO_GPA
) -> np.ndarray:
    """
    Calculates the numerical stress tensor in GPa.

    Args:
        cnn: CNN object
            Committee neural network configuration class.
        structure: pynnp.Structure object
            Lattice and atoms coordinate in pynnp friendly format.
        dh: float
            Displacement for calculation of numerical derivatives.
        conversion_factor: float
            Conversion factor from calculator units to GPa. Default: EV_ANGSTROM3_TO_GPA

    Returns:
        stress_tensor: np.ndarray
            A 3x3 array of numerical stress tensor in GPa.
    """
    direct = np.array([v.r for v in structure.box])
    inverse = np.linalg.inv(direct)
    volume = np.linalg.det(direct)

    original_atoms_positions_cartesian = [atom.r.r for atom in structure.atoms]
    original_atoms_positions_fractional = np.dot(
        original_atoms_positions_cartesian, inverse
    )

    _stress_tensor = np.zeros((3, 3))
    for i in range(3):
        for j in range(i, 3):

            # calculating unit lattice displacement tensor
            lattice_dh = np.zeros((3, 3))
            if i == j:
                lattice_dh[i, i] = dh
            else:
                lattice_dh[j, i] = dh / np.sqrt(2)
                lattice_dh[i, j] = dh / np.sqrt(2)

            diff = 0
            for sign in [-1, 1]:
                structure_ = pynnp.Structure()
                structure_.setElementMap(cnn.models[0].elementMap)

                # updating atoms positions based on the new lattice vectors
                new_direct = direct + sign * lattice_dh
                new_atoms_positions_cartesian = np.dot(
                    original_atoms_positions_fractional, new_direct
                )
                for ilat, lv in enumerate(new_direct):
                    x, y, z = lv
                    structure_.box[ilat].r = pynnp.Vec3D(x=x, y=y, z=z)

                for iatom, pos in enumerate(new_atoms_positions_cartesian):
                    x, y, z = pos
                    temp_atom = pynnp.Atom()
                    temp_atom.r = pynnp.Vec3D(x=x, y=y, z=z)
                    structure_.addAtom(
                        temp_atom,
                        cnn.models[0].elementMap[structure.atoms[iatom].element]
                    )

                structure_.isPeriodic = structure.isPeriodic
                structure_.pbc = structure.pbc

                e_, f_, std_e_, std_f_ = n2p2_committee_calculator(
                    cnn,
                    structure_,
                    calculate_forces=False
                )
                diff += sign * e_
            _stress_tensor[i, j] = diff / (2 * dh * volume)
            _stress_tensor[j, i] = _stress_tensor[i, j]
    return _stress_tensor * conversion_factor
