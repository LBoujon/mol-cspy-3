import numpy as np
from typing import Optional, Union, List, Tuple
from ase import Atoms, neighborlist
import logging
from scipy.spatial import distance_matrix
from scipy import sparse
import networkx as nx

LOG = logging.getLogger(__name__)


def extended_xyz_str(
    atomic_numbers: Union[List[int], np.ndarray],
    atom_symbols: Union[List[str], np.ndarray],
    positions: np.ndarray,
    lattice_vectors: np.ndarray,
    energy: Optional[float] = None,
    forces: Optional[np.ndarray] = None
) -> str:
    """
    Generate the string for an extended XYZ file.

    Parameters:
        atomic_numbers: list or numpy array of atomic numbers
        atom_symbols: list or numpy array of atom symbols
        positions: numpy array of shape (N, 3) where N is the number of atoms
        lattice_vectors: numpy array of shape (3, 3)
        energy: Total energy
        forces: numpy array of shape (N, 3), optional. If not specified, adds zero for forces.

    Returns:
        Extended xyz string
    """
    num_atoms = len(atomic_numbers)

    if energy is None:
        energy = 0

    if forces is None:
        forces = np.zeros_like(positions)

    output_str = f"{num_atoms}\n"
    output_str += (
        f"total_energy={energy} pbc=\"T T T\" "
        f"Lattice=\"{' '.join(f'{vec:12.8f}' for vec in lattice_vectors.flatten())}\" "
        "Properties=species:S:1:pos:R:3:forces:R:3:Z:I:1\n"
    )

    for i in range(num_atoms):
        atom_symbol = atom_symbols[i]
        atomic_num = atomic_numbers[i]
        pos = positions[i]
        force = forces[i]
        pos_str = ' '.join(f"{p:12.8f}" for p in pos)
        force_str = ' '.join(f"{f:12.8f}" for f in force)
        output_str += f"{atom_symbol:<3} {pos_str} {force_str} {atomic_num}\n"

    return output_str


def connect_molecule(
    positions: np.ndarray,
    numbers: np.ndarray,
    cell_vectors: np.ndarray,
    cut_off: Optional[float] = 1.6
) -> tuple[np.ndarray, np.ndarray]:
    """
    Takes the position of atoms in a molecule (possibly connected by periodic images),
    and lattice vectors, and generates a connected molecule in the central unit cell.

    Args:
        positions: np.ndarray
            Position of atoms of the detected connected molecule.
        numbers: np.ndarray
            Atomic numbers of atoms in the molecule.
        cell_vectors: np.ndarray
            Lattice vectors.
        cut_off: float
            Cutoff for distances below which considered bonded.

    Returns:
        connected_atoms: np.ndarray
            Positions of atoms of the connected molecule in the unit cell.
        connected_numbers: np.ndarray
            Atomic numbers of the connected molecule.
    """
    supercell = np.empty((0, 3))
    supercell_numbers = np.empty((0,), dtype=numbers.dtype)
    hkl = [0, 1, -1]

    for i in hkl:
        for j in hkl:
            for k in hkl:
                shift = np.dot([i, j, k], cell_vectors)
                pos_temp = positions + shift
                supercell = np.vstack([supercell, pos_temp])
                supercell_numbers = np.concatenate([supercell_numbers, numbers])

    distance_mat = distance_matrix(supercell, supercell)
    distance_mat[distance_mat > cut_off] = 0
    g = nx.from_numpy_matrix(distance_mat)

    connected_atoms_ids = nx.node_connected_component(g, 0)
    connected_atoms = supercell[list(connected_atoms_ids)]
    connected_numbers = supercell_numbers[list(connected_atoms_ids)]

    return connected_atoms, connected_numbers


def get_molecules(atoms: Atoms) -> tuple[int, list]:
    """
    Takes an ase.Atoms object and returns a list of IDs of molecules to which atoms belong.

    Args:
        atoms: ase.Atoms object

    Returns:
        n_molecules: int
            Number of detected molecules.
        molecules_list: list
            The list of molecule IDs that each atom belongs to.
    """
    cut_off = neighborlist.natural_cutoffs(atoms)
    neighbor_list = neighborlist.NeighborList(
        cut_off,
        self_interaction=False,
        bothways=True
    )
    neighbor_list.update(atoms)
    matrix = neighbor_list.get_connectivity_matrix()
    n_molecules, molecules_list = sparse.csgraph.connected_components(matrix)
    return n_molecules, molecules_list


def fix_unitcell_molecules(atoms: Atoms, cut_off: Optional[float] = 1.6):
    """
    Takes an ase.Atoms object and connects the molecules in it, in a way that molecules won't need
    periodic images to be full.

    Args:
        atoms: ase.Atoms object
        cut_off: float
            Cutoff for distances below which considered bonded.

    Returns:
        atoms: ase.Atoms object

    """
    n_components, component_list = get_molecules(atoms)

    # Finding IDs of atoms of molecules
    molecules = []
    for mol_id in np.unique(component_list):
        mol_atoms = np.where(component_list == mol_id)[0]
        molecules.append(mol_atoms)

    # Correct the atoms positions in molecules, in a way that a molecule is already connected in the unit
    # cell without need to consider periodic images.
    corrected_positions = np.array([])
    corrected_numbers = np.array([])
    for molecule in molecules:
        mol_atoms_positions = atoms.positions[molecule]
        mol_atoms_numbers = atoms.numbers[molecule]
        updated_positions, updated_numbers = connect_molecule(
            mol_atoms_positions,
            mol_atoms_numbers,
            atoms.cell,
            cut_off=cut_off
        )
        corrected_positions = np.append(corrected_positions, updated_positions).reshape(-1, 3)
        corrected_numbers = np.append(corrected_numbers, updated_numbers)
    atoms.positions = corrected_positions
    atoms.numbers = corrected_numbers

    return atoms


def extract_bond(
    atoms: Atoms,
    cut_off: Optional[float] = 1.6
) -> Tuple[Atoms, list]:
    """
    Takes an ase.Atoms object, fixes the coordinates of molecules so each molecule is
    already connected, without need for periodic images. Returns the 'fixed' cell and the
    list of all the bonds.

    Args:
        atoms: ase.Atoms object
            Structure for which bonds should be listed.
        cut_off: float
            Cutoff for distances below which considered bonded.

    Returns:
        atoms: ase.Atoms object
            The fixed atoms object.
        bonds: list
            List of tuples representing bonded atom indices.
    """
    atoms = fix_unitcell_molecules(atoms, cut_off=cut_off)
    distance_mat = distance_matrix(atoms.positions, atoms.positions)
    distance_mat[distance_mat > cut_off] = 0
    g = nx.from_numpy_matrix(distance_mat)
    bonds = list(g.edges)
    return atoms, bonds


def fix_unitcell_and_detect_molecules(
    atoms: Atoms,
    cut_off: Optional[float] = 1.6
) -> Tuple[Atoms, list]:
    """
    Takes an ase.Atoms object, connects the molecules in the central unit cell,
    and returns the IDs of molecules.

    Args:
        atoms: ase.Atoms object
        cut_off: float
            Cutoff for detecting bonds.

    Returns:
        atoms: ase.Atoms object
            "Fixed" unit cell with connected molecules.
        molecule_ids: list
            A list of integers specifying which molecule the corresponding atom is connected to.
    """
    atoms = fix_unitcell_molecules(atoms, cut_off=cut_off)
    n_molecules, molecule_ids = get_molecules(atoms)
    return atoms, molecule_ids
