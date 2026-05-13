import logging
import os
import sys
from copy import deepcopy
from collections import defaultdict
from typing import List, Dict, Tuple, Literal, Union, Optional, TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree as KDTree
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix, dok_matrix

from networkx import minimum_cycle_basis
from networkx.algorithms import isomorphism

from cspy.linalg.kabsch import reorient_points, rmsd_points
from cspy.util.constants import BOHR2ANGSTROM
from .element import Element

from rdkit.Chem import RWMol

if TYPE_CHECKING:
    from cspy.crystal import Crystal

# networkx.from_scipy_sparse_matrix is deprecated in new version
if sys.version_info.major > 2 and sys.version_info.minor >= 8:
    from networkx import from_scipy_sparse_array as from_scipy_sparse
else:
    from networkx import from_scipy_sparse_matrix as from_scipy_sparse


LOG = logging.getLogger(__name__)


def guess_hybridization_state(average_angle):
    if average_angle >= 155.0:
        return "sp"
    elif average_angle >= 115.0:
        return "sp3"
    else:
        return "sp2"
    
def amd_molecules(positions: np.ndarray) -> np.ndarray:
    """Calculate the average mean distance (AMD) descriptor of a molecule.

    Parameters
    ----------
    positions : np.ndarray
        The positions of the atoms in the molecule.

    Returns
    -------
    np.ndarray
        The AMD descriptor.
    """
    distance_mat = distance_matrix(positions, positions)
    return np.mean(np.sort(distance_mat, axis=1), axis=0)


def find_exact_mapping(input_molecule: "Molecule", target_molecule: "Molecule"):
    """
    Find the exact mapping between two molecules.
    The two molecules should have the same geometry and NO SYMMETRY.

    Parameters
    ----------
    input_molecule : cspy.chem.Molecule
        The molecule to reorder the atoms.
    target_molecule : cspy.chem.Molecule
        The molecule to match the atoms to.

    Returns
    -------
    list
        The exact mapping between the two molecules.
    """
    pos1 = input_molecule.positions
    pos2 = target_molecule.positions
    amd1 = amd_molecules(pos1)
    amd2 = amd_molecules(pos2)
    if np.linalg.norm(amd1 - amd2) > 1e-3:
        raise ValueError("The two molecules have different geometries.")

    distances1 = cdist(pos1, pos1)
    descriptor1 = np.sort(distances1, axis=1)
    distances2 = cdist(pos2, pos2)
    descriptor2 = np.sort(distances2, axis=1)

    order = []
    mapping = []
    for i in range(descriptor2.shape[0]):
        j = np.argmin(np.linalg.norm(descriptor1 - descriptor2[i], axis=1))
        order.append(j)
        mapping.append((j, i))

    if logging.getLogger().isEnabledFor(logging.DEBUG):
        LOG.debug("Testing the exact mapping between the two molecules.")
        LOG.debug("before reordering:")
        LOG.debug(
            "rmsd: %s",
            rmsd_points(input_molecule.positions, target_molecule.positions),
        )
        LOG.debug(
            "%s",
            input_molecule.positions_in_molecular_axis_frame()
            - target_molecule.positions_in_molecular_axis_frame(),
        )

        overlay_output = deepcopy(input_molecule)
        overlay_output.reorder_atoms(order)
        LOG.debug("after reordering:")
        LOG.debug(
            "rmsd: %s",
            rmsd_points(overlay_output.positions, target_molecule.positions),
        )
        LOG.debug(
            "%s",
            overlay_output.positions_in_molecular_axis_frame()
            - target_molecule.positions_in_molecular_axis_frame(),
        )

        LOG.debug("Exact mapping found (input index, target index): %s", mapping)
        LOG.debug(
            "Saving input_mol.xyz, target_mol.xyz, overlay_output_mol.xyz to the current directory."
        )
        overlay_output.to_xyz_file("overlay_output_mol.xyz")
        target_molecule.to_xyz_file("target_mol.xyz")
        input_molecule.to_xyz_file("input_mol.xyz")

    return order

def AMD_molecules(positions):
    """Calculate the average mean distance of a molecule.
    
    Parameters
    ----------
    positions : np.ndarray
        The positions of the atoms in the molecule.
    """
    distance_mat_ = distance_matrix(positions, positions)
    return np.mean(np.sort(distance_mat_, axis=1), axis=0)

def find_exact_mapping(input_molecule, target_molecule):
    """
    Find the exact mapping between two molecules.
    The two molecules should have the same geometry and NO SYMMETRY.

    Parameters
    ----------
    input_molecule : cspy.chem.Molecule
        The molecule to reorder the atoms.
    target_molecule : cspy.chem.Molecule
        The molecule to match the atoms to.

    Returns
    -------
    list
        The exact mapping between the two molecules.
    """
    pos1 = input_molecule.positions
    pos2 = target_molecule.positions
    amd1 = AMD_molecules(pos1)
    amd2 = AMD_molecules(pos2)
    if np.linalg.norm(amd1 - amd2) > 1e-3:
        raise ValueError("The two molecules have different geometries.")
    
    distances1 = cdist(pos1, pos1)
    descriptor1 = np.sort(distances1, axis=1)
    distances2 = cdist(pos2, pos2)
    descriptor2 = np.sort(distances2, axis=1)

    order = []
    mapping = []
    for i in range(descriptor2.shape[0]):
        j = np.argmin(np.linalg.norm(descriptor1 - descriptor2[i], axis=1))
        order.append(j)
        mapping.append((j, i))

    if logging.getLogger().isEnabledFor(logging.DEBUG):
        LOG.debug(f"Testing the exact mapping between the two molecules.")
        LOG.debug("before reordering:")
        LOG.debug(f"rmsd: {rmsd_points(input_molecule.positions, target_molecule.positions)}")
        LOG.debug(input_molecule.positions_in_molecular_axis_frame() - target_molecule.positions_in_molecular_axis_frame())

        overlay_output = deepcopy(input_molecule)
        overlay_output.reorder_atoms(order)
        LOG.debug("after reordering:")
        LOG.debug(f"rmsd: {rmsd_points(overlay_output.positions, target_molecule.positions)}")
        LOG.debug(overlay_output.positions_in_molecular_axis_frame() - target_molecule.positions_in_molecular_axis_frame())

        LOG.debug(f"Exact mapping found (input index, target index): {mapping}")
        LOG.debug("Saving input_mol.xyz, target_mol.xyz, overlay_output_mol.xyz to the current directory.")
        overlay_output.to_xyz_file("overlay_output_mol.xyz")
        target_molecule.to_xyz_file("target_mol.xyz")
        input_molecule.to_xyz_file("input_mol.xyz")

    return order


class Molecule:
    positions: np.ndarray
    elements: list
    labels: np.ndarray
    properties: dict
    components: list

    def __init__(
        self,
        elements: list[Element],
        positions: np.ndarray,
        bonds: np.ndarray = None,
        labels: np.ndarray = None,
        components: list = ["Molecule"],
        **kwargs,
    ):
        self.positions = positions
        self.elements = elements
        self.properties = {}
        self.properties.update(kwargs)
        self.bonds = None
        self.components = components
        self.equivalent_atoms = None

        if bonds is None:
            if kwargs.get("guess_bonds", False):
                self.guess_bonds()
        else:
            self.bonds = dok_matrix(bonds)

        if labels is None:
            self.assign_default_labels()
        else:
            self.labels = labels

    def match_atom_ordering_to(self, 
                               other: "Molecule", 
                               method: Literal["position_match", "molecular_axis", "rdkit"] = "position_match"
                              ) -> None:
        """
        Reorder the atoms in this molecule to match the atom order of another molecule
        with the same geometry but possibly different atom order.

        Parameters
        ----------
        other : Molecule
            The molecule whose atom order to match.
        method : str, optional
            The method to use for reordering the atoms. Options:
            - 'position_match': Match atoms by position in molecular axis frame.
            - 'molecular_axis': Match atoms by minimum distance in molecular axis frame.
            - 'rdkit': Use RDKit substructure matching.

        Returns
        -------
        None
            Updates the order of atoms in the molecule to match the other molecule.
        """
        if method not in ["position_match", "molecular_axis", "rdkit"]:
            raise NotImplementedError(f"Method '{method}' is not implemented.")

        if method not in ["position_match", "molecular_axis", "rdkit"]:
            raise NotImplementedError("Method {} is not implemented.".format(method))

        pos1 = self.positions_in_molecular_axis_frame()
        pos2 = other.positions_in_molecular_axis_frame()

        if pos1.shape != pos2.shape:
            raise ValueError("The two molecules must have the same number of atoms.")

        if np.linalg.norm(amd_molecules(pos1) - amd_molecules(pos2)) > 1e-3:
            raise ValueError(
                "AMD descriptor of molecules is different. Ensure the geometries are identical."
            )

        if method == "position_match":
            new_order = [
                np.argmin(np.linalg.norm(pos1 - pos, axis=1)) for pos in pos2
            ]
        elif method == "molecular_axis":
            new_order = [
                np.argmin(np.linalg.norm(pos1 - pos, axis=1)) for pos in pos2
            ]
        elif method == "rdkit":
            from rdkit import Chem
            mol1 = Chem.MolFromXYZBlock(self.to_xyz_string())
            mol2 = Chem.MolFromXYZBlock(other.to_xyz_string())
            new_order = list(mol1.GetSubstructMatch(mol2))

        self.reorder_atoms(new_order)

    def reorder_atoms(self, new_order: List[int]) -> None:
        """Reorder atoms in the molecule according to new_order indices.

        Parameters
        ----------
        new_order : List[int]
            The new order of atom indices.

        Returns
        -------
        None
        """
        self.positions = self.positions[new_order]
        if not isinstance(self.labels, np.ndarray):
            self.labels = np.array(self.labels)
        self.labels = self.labels[new_order]
        self.elements = [self.elements[i] for i in new_order]

    def __iter__(self):
        for atom in zip(self.elements, self.positions):
            yield atom

    def __len__(self):
        return len(self.elements)

    @property
    def covalent_radii(self):
        return np.array([x.cov for x in self.elements])

    @property
    def vdw_radii(self):
        return np.array([x.vdw for x in self.elements])

    def neighcrys_bond_cutoffs(self, tolerance=0.1001):
        if self.bonds is None:
            self.guess_bonds()
        max_bond = {}
        for (i, j), v in self.bonds.items():
            el_i = self.elements[i].symbol
            el_j = self.elements[j].symbol
            key = tuple(sorted((el_i, el_j)))
            max_bond[key] = max(max_bond.get(key, 0.0), v)

        return {k: round(v + tolerance, 4) for k, v in max_bond.items()}

    def def_components(self, component_atom_ids : list[list[int]], reset: bool = False) -> None:
        """ Set new property called 'components' which is a list of
        Molecule objects. Each of these components must contain only
        atoms which are found in the top-level Molecule object.
        This allows for us to define a top-level Molecule object 
        which contains multiple molecules (i.e. a cluster) and then
        define each of the constituent molecules as components.
        Because the components are Molecule objects, they can be
        operated on individually with any of our Molecule methods.
        Alternatively, we can operate on the entire cluster as a single
        unit by calling methods from the top-level Molecule object.

        Parameters
        ----------
        component_atom_ids : list[list[int]]
            List of lists where each sublist is one component.
            Each component list atoms the atom ids from the Molecule.
            If reset, wipe existing list of components.

        reset : bool
            If true, delete the existing list of components

        Returns
        -------
        None
        """

        if reset:
            self.components = []

        for component in component_atom_ids:
            comp_elements = []
            comp_positions = []
            for atom in component:
                comp_elements.append(self.elements[atom])
                comp_positions.append(self.positions[atom])

            comp_positions = np.asarray(comp_positions)

            compMol = Molecule(comp_elements, comp_positions)
            self.components.append(compMol)

    def add_component(self, component : "Molecule"):
        """ Add new component to Molecule. 
        The new component should already be a Molecule.
        Atoms are added to existing list of positions and elements.

        Parameters
        ----------
        component : Molecule
            A mol-CSPy Molecule object that will be added to the 
            existing Molecule object and designated as a component.

        Returns
        -------
        None
        """

        num_old_atoms = len(self.elements)

        self.elements += component.elements
        self.positions = np.concatenate((self.positions, component.positions), axis=0)

        component_atom_ids = [[atom for atom in range(num_old_atoms, len(self.elements))]]

        self.def_components(component_atom_ids, reset=False)


    def sort_components(self, new_order : list[int]) -> None:
        """ Change order of components.
        Used so that an output .xyz or .res has the right
        ordering for DMACRYS.

        Parameters
        ----------
        new_order : list[int]
            List of integers where each integer is the index
            of a component.
            The order of the new_order list determines the 
            new order of the components.

        Returns
        -------
        None
        """

        old_components = self.components
        new_components = []
        new_elements = []
        new_positions = np.asarray([])
        for ind in new_order:
            component = old_components[ind]
            new_components.append(component)
            new_elements += component.elements
            if len(new_positions) == 0:
                new_positions = component.positions
            else:
                new_positions = np.concatenate((new_positions, component.positions), axis=0)

        self.components = new_components
        self.elements = new_elements
        self.positions = new_positions


    def guess_bonds(self, tolerance=0.40):
        """Use geometric distances and covalent radii
        to determine bonding information for this molecule.
        Will set the bonds member.

        tolerance is default 0.4 angstroms, which is recommended
        by the CCDC
        """
        tree = KDTree(self.positions)
        covalent_radii = np.array([x.cov for x in self.elements])
        max_cov = np.max(covalent_radii)
        thresholds = (
            covalent_radii[:, np.newaxis] + covalent_radii[np.newaxis, :] + tolerance
        )
        max_distance = max_cov * 2 + tolerance
        dist = tree.sparse_distance_matrix(tree, max_distance=max_distance).toarray()
        mask = (dist > 0) & (dist < thresholds)
        self.bonds = np.zeros(dist.shape)
        self.bonds[mask] = dist[mask]
        self.bonds = dok_matrix(self.bonds)

    def bfs_order(self, start=0, directed=True, **kwargs):
        from scipy.sparse.csgraph import breadth_first_order

        if self.bonds is None:
            self.guess_bonds()
        return breadth_first_order(self.bonds, start, directed=directed, **kwargs)

    def dfs_order(self, start=0, directed=True, **kwargs):
        from scipy.sparse.csgraph import depth_first_order

        if self.bonds is None:
            self.guess_bonds()
        return depth_first_order(self.bonds, start, directed=directed, **kwargs)

    def bond_angle(self, idxs):
        """idxs a, b, c -> the bond angle between bonds ab and bc"""
        origin = self.positions[idxs[1]]
        neighbours = [idxs[0], idxs[2]]
        vecs = self.positions[neighbours, :] - origin
        vecs = vecs / np.linalg.norm(vecs, axis=1)[:, np.newaxis]
        dots = np.clip(np.dot(vecs, vecs.T), -1.0, 1.0)
        return np.degrees(np.arccos(dots[np.triu_indices(2, k=1)]))[0]

    def plane_normal(self, idxs):
        """idxs a, b, c -> the normal to the plane containg atoms a, b, c"""
        origin = self.positions[idxs[0]]
        neighbours = [idxs[1], idxs[2]]
        vecs = self.positions[neighbours, :] - origin
        return np.cross(vecs[0, :], vecs[1, :])

    def dihedral_angle(self, idxs):
        """a, b, c, d -> dihedral angle of planes through a, b, c and b, c, d"""
        n1 = self.plane_normal([idxs[0], idxs[1], idxs[2]])
        n2 = self.plane_normal([idxs[1], idxs[2], idxs[3]])
        n1n2 = np.linalg.norm(n1) * np.linalg.norm(n2)
        return np.degrees(np.arccos(np.dot(n1, n2) / n1n2))

    @property
    def rings(self):
        if not hasattr(self, "_rings"):
            if self.bonds is None:
                self.guess_bonds()
            self._rings = minimum_cycle_basis(
                from_scipy_sparse(self.bonds)
            )
        return self._rings

    def neighbouring_atoms(self, idx):
        if self.bonds is None:
            self.guess_bonds()
        return self.bonds[idx].nonzero()[1]

    def foreshortened_hydrogen_positions(self, reduction=0.1):
        positions = self.positions.copy()
        nums = self.atomic_numbers
        for i in range(positions.shape[0]):
            if nums[i] != 1:
                continue
            neighbours = self.neighbouring_atoms(i)
            j = neighbours[0]
            v_ij = self.positions[j] - self.positions[i]
            d = reduction * (v_ij / np.linalg.norm(v_ij))
            positions[i, :] += d
        return positions

    def ring_dihedral_angles(self):
        from cspy.linalg import plane_of_best_fit

        rings = self.rings
        angles = {}
        for a in range(len(rings)):
            n1, d1, rmsd1 = plane_of_best_fit(self.positions[rings[a]])
            for b in range(a + 1, len(rings)):
                n2, d2, rmsd2 = plane_of_best_fit(self.positions[rings[b]])
                angles[(a, b)] = np.degrees(np.arccos(np.vdot(n1, n2)))
        return angles

    def guess_hybridization_states(self):
        states = {}
        for i in range(len(self)):
            name = self.labels[i]
            origin = self.positions[i]
            neighbours = self.bonds[i].nonzero()[1]
            num_neighbours = len(neighbours)
            if num_neighbours < 2:
                continue
            vecs = self.positions[neighbours, :] - origin
            vecs = vecs / np.linalg.norm(vecs, axis=1)[:, np.newaxis]
            dots = np.dot(vecs, vecs.T)
            angles = np.degrees(np.arccos(dots[np.triu_indices(num_neighbours, k=1)]))
            average_angle = np.mean(angles)
            state = guess_hybridization_state(average_angle)
            states[name] = state
        return states

    def axes(self, method="nc", homogeneous=False):
        """ If `neighcrys_axis_atom` is defined in self.properties,
        will use these indices to generate the axis."""
        if method == "nc":
            if len(self) < 3:
                LOG.error("Invalid length of molecular positions: %d", len(self))
                raise NotImplementedError(
                    "molecular axes (method=nc) only implemented for n >= 3"
                )
            atom_a, atom_b, atom_c = self.properties.get(
                "neighcrys_axis_atom", (0, 1, 2)
            )
            LOG.debug("Neighcrys axis atom idxs: %s", (atom_a, atom_b, atom_c))
            x0 = self.positions[atom_b] - self.positions[atom_a]
            x0 /= np.linalg.norm(x0)
            x1 = self.positions[atom_a] - self.positions[atom_c]
            x1 /= np.linalg.norm(x1)
            x1 = x1 - np.vdot(x1, x0) * x0
            x1 /= np.linalg.norm(x1)
            x2 = np.cross(x0, x1)
            axes = np.array([x0, x1, x2])
            assert np.linalg.det(axes) > 0, "Coordinate system must be right-handed"
        elif method == "pca":
            axes, s, vh = np.linalg.svd((self.positions - self.center_of_mass).T)
        elif method == "moi":
            moi = np.zeros((3, 3))
            masses = np.asarray([x.mass for x in self.elements])
            positions = self.positions - self.center_of_mass
            for mass, pos in zip(masses, positions):
                moi[0][0] += mass * (pos[1]**2 + pos[2]**2)
                moi[1][1] += mass * (pos[0]**2 + pos[2]**2)
                moi[2][2] += mass * (pos[0]**2 + pos[1]**2)
                moi[0][1] -= mass * pos[0] * pos[1]
                moi[0][2] -= mass * pos[0] * pos[2]
                moi[1][2] -= mass * pos[1] * pos[2]
            moi = moi + moi.T - np.diag(np.diag(moi))
            axes = np.linalg.eigh(moi)[1].T
        else:
            raise ValueError(f"Unknown molecular axis method '{method}'")

        if homogeneous:
            transform = np.eye(4)
            transform[:3, :3] = axes
            translation = -np.dot(axes, self.center_of_mass)
            transform[:3, 3] = translation
            transform[np.abs(transform) < 1e-15] = 0
            return transform

        return axes

    def distance_to(self, other, method="centroid"):
        method = method.lower()
        if method == "centroid":
            return np.linalg.norm(self.centroid - other.centroid)
        elif method == "center_of_mass":
            return np.linalg.norm(self.center_of_mass - other.center_of_mass)
        elif method == "nearest_atom":
            return np.min(cdist(self.positions, other.positions))
        else:
            raise ValueError(f"Unknown method={method}")

    @property
    def homogeneous_positions(self):
        return np.c_[self.positions, np.ones(len(self))].T

    @property
    def atomic_numbers(self):
        return np.array([e.atomic_number for e in self.elements])

    def transform(
        self, transformation_matrix, translation=np.zeros(3), rotate_multipoles=False
    ):
        if transformation_matrix.shape[1] == 4:
            h = self.homogeneous_positions
            t = np.dot(transformation_matrix, h)
            self.positions = t.T[:, :3]
            if rotate_multipoles:
                raise NotImplementedError(
                    "Haven't implemented multipole rotation for homogeneous transforms"
                )
        else:
            self.positions = np.dot(
                self.positions + translation, transformation_matrix.T
            )
            if rotate_multipoles:
                self.properties["multipoles"] = [
                    m.transformed(transformation_matrix) for m in self.multipoles
                ]

    def translate(self, translation):
        self.positions += translation

    def translated(self, translation):
        import copy

        result = copy.deepcopy(self)
        result.positions += translation
        return result

    @property
    def is_connected(self):
        from scipy.sparse.csgraph import connected_components

        nfrag, _ = connected_components(self.bonds)
        return nfrag == 1

    def connected_fragments(self):
        from cspy.linalg import cartesian_product
        from scipy.sparse.csgraph import connected_components

        if self.bonds is None:
            self.guess_bonds()

        nfrag, labels = connected_components(self.bonds)
        molecules = []
        for frag in range(nfrag):
            atoms = np.where(labels == frag)[0]
            na = len(atoms)
            sqidx = cartesian_product(atoms, atoms)
            molecules.append(
                Molecule(
                    [self.elements[i] for i in atoms],
                    self.positions[atoms],
                    labels=self.labels[atoms],
                    bonds=self.bonds[sqidx[:, 0], sqidx[:, 1]].reshape(na, na),
                )
            )
        return molecules

    def assign_default_labels(self):
        counts = defaultdict(int)
        labels = []
        for el, _ in self:
            counts[el] += 1
            labels.append("{}{}".format(el.symbol, counts[el]))
        self.labels = np.asarray(labels)

    @property
    def centroid(self):
        return np.mean(self.positions, axis=0)

    @property
    def center_of_mass(self):
        masses = np.asarray([x.mass for x in self.elements])
        return np.sum(self.positions * masses[:, np.newaxis] / np.sum(masses), axis=0)

    @property
    def molecular_formula(self):
        from cspy.chem.element import chemical_formula

        return chemical_formula(self.elements, subscript=False)

    @property
    def multipoles(self):
        return self.properties.get("multipoles", None)

    def __repr__(self):
        return "<{}: {}({:.2f},{:.2f},{:.2f})>".format(
            self.__class__.__name__, self.molecular_formula, *self.center_of_mass
        )

    def reflect(self, plane):
        if plane == "xy":
            self.positions[:, 2] *= -1
            if "multipoles" in self.properties:
                self.properties["multipoles"] = [
                    x.reflected(plane) for x in self.multipoles
                ]
        else:
            raise NotImplementedError

    def positions_in_molecular_axis_frame(
        self, method="nc", foreshorten_hydrogens=None
    ):
        if method not in ("nc", "pca", "moi"):
            raise NotImplementedError("Only nc, pca, moi implemented")
        if len(self) == 1:
            return np.array([[0.0, 0.0, 0.0]])
        axis = self.axes(method=method)
        if foreshorten_hydrogens:
            positions = self.foreshortened_hydrogen_positions(
                reduction=foreshorten_hydrogens
            )
            return np.dot(positions - self.center_of_mass, axis.T)
        else:
            return np.dot(self.positions - self.center_of_mass, axis.T)

    @classmethod
    def group_atoms_by_element(cls, molecules):
        """Sort all arrays, grouped by element"""
        from cspy.util import grouped_ordering

        symbols = []
        atom_nums = [0]
        for mol in molecules:
            symbols += [i.symbol for i in mol.elements]
            atom_nums.append(atom_nums[-1] + len(mol))

        new_order = grouped_ordering(symbols)
        for n, mol in enumerate(molecules):

            temp = []
            for i in range(atom_nums[n], atom_nums[n + 1]):
                temp.append(new_order.index(i))
            sub_order = [None for _ in range(len(temp))]
            for i, j in enumerate(sorted(temp)):
                sub_order[temp.index(j)] = i
            sub_order = [sub_order.index(i) for i in range(len(sub_order))]

            mol.elements = [mol.elements[i] for i in sub_order]
            mol.positions = mol.positions[sub_order]
            mol.labels = np.array(mol.labels)[sub_order]
            for item in mol.properties.items():
                if isinstance(item, np.ndarray):
                    item = item[sub_order]
                elif isinstance(item, list) and len(item) == len(mol):
                    item = [item[i] for i in sub_order]
            if mol.bonds:
                mol.guess_bonds()

    def oriented(self, method="nc"):
        from copy import deepcopy

        result = deepcopy(self)
        result.positions = self.positions_in_molecular_axis_frame(method=method)
        return result

    def minimize_with_dftb(self, **kwargs):
        """Optimizes the structure using dftb

        Returns a minimized version of this molecule
        """
        from cspy.minimize import dftb_calculator
        return dftb_calculator(self, **kwargs)

    def minimize_with_xtb(self, **kwargs):
        """Optimizes the structure using xtb

        Returns a minimized version of this molecule
        """
        from cspy.minimize import xtb_calculator
        return xtb_calculator(self, **kwargs)

    @classmethod
    def from_xyz_file(cls, filename, **kwargs):
        from cspy.formats.xyz import parse_xyz_file

        xyz_dict = parse_xyz_file(filename)
        return cls.from_xyz_dict(xyz_dict, **kwargs)

    @classmethod
    def from_xyz_string(cls, contents, **kwargs):
        from cspy.formats.xyz import parse_xyz_string

        xyz_dict = parse_xyz_string(contents)
        return cls.from_xyz_dict(xyz_dict, **kwargs)

    @classmethod
    def from_xyz_dict(cls, xyz_dict, **kwargs):
        elements = []
        positions = []
        for label, position in xyz_dict["atoms"]:
            elements.append(Element[label])
            positions.append(position)
        return cls(
            elements, np.asarray(positions), components = [], comment=xyz_dict["comment"], **kwargs
        )

    @classmethod
    def from_gaussian_optimization(cls, filename, **kwargs):
        from cspy.formats.gaussian import GaussianLogFile

        log_file = GaussianLogFile(filename)
        geometries = log_file.geometries
        if len(geometries["inp"]) > 0:
            geom = geometries["inp"][-1]
        else:
            LOG.info(
                "Could not find input orientation in %s, loading standard orientation instead",
                filename,
            )
            geom = geometries["std"][-1]
        return cls.from_arrays(
            elements=geom["elements"],
            positions=geom["positions"],
            scf_energy=log_file.final_energy,
            **kwargs,
        )

    @classmethod
    def from_fchk_file(cls, filename, **kwargs):
        from cspy.formats.gaussian import GaussianFchkFile

        fchk = GaussianFchkFile(filename, parse=True)
        elements = np.array(fchk["Atomic numbers"])
        positions = np.array(fchk["Current cartesian coordinates"]).reshape(
            elements.shape[0], 3
        )
        positions *= BOHR2ANGSTROM
        return cls.from_arrays(elements=elements, positions=positions, **kwargs)

    @classmethod
    def from_gaussian_optimization_string(cls, contents, **kwargs):
        from cspy.formats.gaussian import GaussianLogFile

        log_file = GaussianLogFile.from_string(contents)
        geometries = log_file.geometries
        if len(geometries["inp"]) > 0:
            geom = geometries["inp"][-1]
        else:
            LOG.info(
                "Could not find input orientation in %s, loading standard orientation instead",
                log_file,
            )
            geom = geometries["std"][-1]
        return cls.from_arrays(
            elements=geom["elements"],
            positions=geom["positions"],
            scf_energy=log_file.final_energy,
            **kwargs,
        )

    @property
    def energy(self):
        return self.properties.get("scf_energy", np.nan)

    @classmethod
    def from_mol2_file(cls, filename, **kwargs):
        from cspy.formats.mol2 import parse_mol2_file

        atoms, bonds = parse_mol2_file(filename)
        positions = np.c_[atoms["x"], atoms["y"], atoms["z"]]
        elements = [Element[x].atomic_number for x in atoms["name"]]
        return cls.from_arrays(elements=elements, positions=positions, **kwargs)

    def to_gen_string(self):
        gen_template = (
                "{num_atoms} C\n"
                "{elements}\n"
                "{coords}\n"
        )
        pos = self.positions
        ele = self.elements
        unique_ele = set(sorted(ele))
        index = np.linspace(1, len(ele), len(ele))
        ele_order_dict = {item:i+1 for i, item in enumerate(unique_ele)}
        temp_coords = np.column_stack(([ele_order_dict[item] for item in ele], pos))
        sorted_temp_coords = temp_coords[temp_coords[:,0].argsort()]
        genfile_coords ="\n".join("{:.10g} {:.10g} {:.10f} {:.10f} {:.10f}".format(*item)
                                  for item in np.column_stack((index, sorted_temp_coords)))
        input_contents = gen_template.format(
            num_atoms=len(ele),
            elements=" ".join(item.symbol for item in unique_ele),
            coords=genfile_coords,
            )
        return input_contents

    def to_gen_file(self, filename):
        with open(filename, "w") as f:
            f.write(self.to_gen_string())

    @classmethod
    def from_gen_file(cls, filename, **kwargs):
        """Initialize a molecule from a GEN file"""
        from cspy.util.path import Path
        p = Path(filename)
        return cls.from_gen_string(p.read_text(), **kwargs)
    
    @classmethod
    def from_gen_string(cls, file_content, **kwargs):
        from cspy.formats.gen import parse_gen_file_content
        gen_dict = parse_gen_file_content(file_content)
        elements = gen_dict['elements']
        positions = gen_dict['positions']
        return cls(elements, np.asarray(positions), **kwargs)

    def save(self, filename):
        extension_map = {
            ".xyz": self.to_xyz_file,
            ".gen": self.to_gen_file,
        }
        extension = os.path.splitext(filename)[-1].lower()
        return extension_map[extension](filename)

    @classmethod
    def load(cls, filename, **kwargs):
        extension_map = {
            ".xyz": cls.from_xyz_file,
            ".log": cls.from_gaussian_optimization,
            ".mol2": cls.from_mol2_file,
            ".zmat": cls.from_zmatrix_file,
            ".fchk": cls.from_fchk_file,
            ".gen": cls.from_gen_file,
        }
        extension = os.path.splitext(filename)[-1].lower()
        return extension_map[extension](filename, **kwargs)

    def rmsd(self, other):
        return self.overlay(other)[2]

    def rmsd_with_reflection(self, other: "Molecule", method: Literal["fixed"] ="fixed", plane="xy"):
        """
        Calculate the RMSD between self and other, optionally reflecting self about a plane.

        Parameters
        ----------
        other : Molecule
            The other molecule to compare with.
        method : str, optional
            The method to use for RMSD calculation. Only 'fixed' is implemented.
        plane : str, optional
            The plane to reflect about ('xy', 'xz', 'yz'). Default is 'xy'.

        Returns
        -------
        tuple
            RMSD value and the plane used for reflection (if any).
        """
        from cspy.linalg import rmsd_points

        if method != "fixed":
            raise NotImplementedError("Only 'fixed' method is implemented.")

        coords_self = self.positions.copy()
        coords_self -= np.mean(coords_self, axis=0)
        coords_other = other.positions.copy()
        coords_other -= np.mean(coords_other, axis=0)
        rms = rmsd_points(coords_self, coords_other)
        LOG.debug("RMSD without flip: %.5g", rms)

        planes = {"xy": 2, "xz": 1, "yz": 0}
        best_reflection = None

        if plane in planes:
            reflection_axis = planes.get(plane)
            if reflection_axis is not None:
                coords_flipped = coords_self.copy()
                coords_flipped[:, reflection_axis] *= -1
                rms_flipped = rmsd_points(coords_flipped, coords_other)
                LOG.debug("Flipping about '%s' plane, RMSD: %.5g", plane, rms_flipped)
                if rms_flipped < rms:
                    rms = rms_flipped
                    best_reflection = plane
        else:
            raise ValueError(f"Invalid plane '{plane}'. Valid options are 'xy', 'xz', 'yz'.")
        return rms, best_reflection

    def rmsd_by_rdkit(self, other: "Molecule") -> float:
        """Calculate the RMSD between self and other molecules using RDKit.

        Parameters
        ----------
        other : cspy.chem.Molecule
            The other molecule to compare with.

        Returns
        -------
        float
            The RMSD between the two molecules.
        """
        try:
            from rdkit.Chem.AllChem import GetBestRMS
            from rdkit.Chem import MolFromMol2Block
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "rdkit was not found. Try to install it by 'pip install rdkit'"
            )
        mol1 = MolFromMol2Block(self.to_mol2_string())
        mol2 = MolFromMol2Block(other.to_mol2_string())
        rmsd = GetBestRMS(mol1, mol2)
        return rmsd

    def __eq__(self, other: "Molecule") -> bool:
        """Check if two molecules are equal based on their positions
        and average minimum distance (AMD) descriptor.

        Parameters
        ----------
        other : Molecule
            The other molecule to compare with.

        Returns
        -------
        bool
            True if the molecules are considered equal, False otherwise.
        """
        if not isinstance(other, Molecule):
            raise TypeError(
                f"Cannot compare Molecule with {type(other)}"
            )

        if len(self) != len(other):
            return False

        return np.allclose(self.amd_descriptor, other.amd_descriptor, atol=1e-3)

    def overlay_by_rdkit(
        self,
        other: "Molecule",
        reorder_atoms_to: Literal["self", "other"] = "self",
        check_connectivity: bool = True,
        **kwargs,
    ) -> Tuple:
        """Overlay and reorder the other molecule onto this one using rdkit.

        Parameters
        ----------
        other : cspy.chem.Molecule
            The other molecule to overlay onto this one.
        reorder_atoms_to : str, optional
            Reorder the atoms of the overlap to match 'self' or 'other'.
            Defaults to 'self'. If 'self', the other molecule will be reordered
            to match the atom order of this molecule. If 'other', the other molecule
            will be reordered to match the atom order of the other molecule.
        check_connectivity : bool, optional
            Check if the molecules have the same connectivity using graph matching.
            Defaults to True.

        kwargs : dict, optional
        -----------------------
            Additional keyword arguments to pass to the GetBestAlignmentTransform function
          - prbMol      molecule that is to be aligned
          - refMol      molecule used as the reference for the alignment
          - prbCid      ID of the conformation in the probe to be used 
                        for the alignment (defaults to first conformation)
          - refCid      ID of the conformation in the ref molecule to which 
                        the alignment is computed (defaults to first conformation)
          - map:        (optional) a list of lists of (probeAtomId, refAtomId)
                        tuples with the atom-atom mappings of the two
                        molecules. If not provided, these will be generated
                        using a substructure search.
          - maxMatches  (optional) if atomMap is empty, this will be the max number of
                        matches found in a SubstructMatch().
          - symmetrizeConjugatedTerminalGroups (optional) if set, conjugated
                        terminal functional groups (like nitro or carboxylate)
                        will be considered symmetrically.
          - weights     Optionally specify weights for each of the atom pairs
          - reflect     if true reflect the conformation of the probe molecule
          - maxIters    maximum number of iterations used in minimizing the RMSD
          - numThreads  (optional) number of threads to use

        Returns
        -------
            A copy with other molecule transformed to overlay with
            this molecule, a list of reordering indices used to
            reorder the other molecule and an rmsd of the overlay.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem.rdMolAlign import GetBestAlignmentTransform
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "rdkit was not found. Try to install it by 'pip install rdkit'"
            )

        if reorder_atoms_to not in ["self", "other"]:
            raise ValueError(
                "reorder_atoms_to parameter should be 'self' or 'other'"
            )

        if check_connectivity:
            if self.has_similar_molecular_graph_to(other):
                LOG.info("The molecules have the same connectivity.")
            else:
                raise NotImplementedError(
                    "Error! Overlaying two different molecules is not implemented."
                )

        mol1 = Chem.MolFromXYZBlock(self.to_xyz_string())
        mol2 = Chem.MolFromXYZBlock(other.to_xyz_string())
        try:
            rmsd, transform_matrix, mapping = GetBestAlignmentTransform(
                mol2, mol1, **kwargs
            )
            LOG.debug("RMSD: %.5g", rmsd)
            LOG.debug("Transform matrix: %s", transform_matrix)
            LOG.debug("Mapping: %s", mapping)
        except Exception as e:
            LOG.error("Failed to overlay molecules using rdkit: %s", e)
            raise ValueError("Failed to overlay molecules using rdkit.")

        if reorder_atoms_to == "self":
            # sort mapping by self labels
            mapping = sorted(mapping, key=lambda x: x[0])
            best_order = [mapped for ref, mapped in mapping]
        else:
            # sort mappings by other labels
            mapping = sorted(mapping, key=lambda x: x[1])
            best_order = [ref for ref, mapped in mapping]
        LOG.debug("Best order: %s", best_order)
        overlayed = deepcopy(other)

        new_pos = []
        for pos in other.positions:
            new_pos.append(np.dot(transform_matrix, np.append(pos, 1.0))[:3])
        new_pos = np.array(new_pos)

        overlayed.positions = new_pos[best_order]
        if not isinstance(overlayed.labels, np.ndarray):
            overlayed.labels = np.array(overlayed.labels)
        overlayed.labels = overlayed.labels[best_order]
        overlayed.elements = [overlayed.elements[i] for i in best_order]

        # Double check atom ordering is correct
        if reorder_atoms_to == "other":
            overlayed.match_atom_ordering_to(other, method="position_match")
            # TODO best_order should be updated to reflect the new ordering
        if overlayed.positions.shape != other.positions.shape:
            raise IOError("The number of atoms in overlayed molecules has changed.")

        return overlayed, best_order, rmsd

    def shake_and_overlay(
        self,
        other: "Molecule",
        check_connectivity: bool = True,
        noise_level: float = 0.02,
        **kwargs,
    ) -> Tuple["Molecule", List, float]:
        """Shake the other molecule and overlay it onto this one.

        Parameters
        ----------
        other : cspy.chem.Molecule
            The other molecule to overlay onto this one.
        check_connectivity : bool, optional
            Check if the molecules have the same connectivity using graph matching.
        noise_level : float, optional
            The standard deviation of the Gaussian noise to add to the positions of the other molecule.
        kwargs : dict, optional
            Additional keyword arguments to pass to the overlay_by_rdkit function.

        Returns
        -------
        tuple
            A copy with other molecule transformed to overlay with
            this molecule, a list of reordering indices used to
            reorder the other molecule and an rmsd of the overlay.
        """
        from rdkit.Chem import AllChem

        noise = np.random.normal(0, noise_level, self.positions.shape)
        shaken = deepcopy(other)
        shaken.positions += noise

        # The overlayed molecule is reordered to match the other molecule
        overlayed_shaken, _, rmsd = self.overlay_by_rdkit(
            shaken,
            reorder_atoms_to="self",
            check_connectivity=check_connectivity,
            **kwargs,
        )
        order = find_exact_mapping(overlayed_shaken, shaken)
        overlayed_shaken.reorder_atoms(order)

        final_overlayed = deepcopy(other)
        overlayed_shaken_rdkit = overlayed_shaken.to_rdkit_mol()
        final_overlayed_rdkit = final_overlayed.to_rdkit_mol()

        # The overlayed molecule below is NOT reordered
        rmsd = AllChem.AlignMol(final_overlayed_rdkit, overlayed_shaken_rdkit)
        final_overlayed.positions = final_overlayed_rdkit.GetConformer().GetPositions()

        return final_overlayed, order, rmsd


    def has_similar_molecular_graph_to(self, other: "Molecule") -> bool:
        """Check whether molecules have similar molecular graphs using networkx.

        Parameters
        ----------
        other : cspy.chem.Molecule
            The other molecule to compare with.

        Returns
        -------
        bool
            True if the molecular graphs are similar, False otherwise.
        """
        if self.bonds is None:
            self.guess_bonds()
        if other.bonds is None:
            other.guess_bonds()

        # create the graph matcher from the molecular connectivity
        graph_1 = from_scipy_sparse(csr_matrix(self.bonds))
        graph_2 = from_scipy_sparse(csr_matrix(other.bonds))
        gm = isomorphism.GraphMatcher(graph_2, graph_1)

        if (
            not np.array_equal(np.sort(self.atomic_numbers), np.sort(other.atomic_numbers))
            or not gm.is_isomorphic()
        ):
            return False
        return True

    def overlay(self, other: "Molecule", reorder_atoms_to: str = "self") -> Tuple["Molecule", List, float]:
        """Overlay and reorder the other molecule onto this one.

        Parameters
        ----------
        other : cspy.chem.Molecule
            The other molecule to overlay onto this one.
        reorder_atoms_to : str, optional
            Reorder the atoms of the overlap to match self or other.

        Returns
        -------
        overlayed : Molecule
            A copy with other molecule transformed to overlay with
            this molecule.
        best_order : list
            A list of reordering indices used to reorder the other molecule.
        best_rmsd : float
            RMSD of the overlay.
        """
        if reorder_atoms_to not in ["self", "other"]:
            raise ValueError(
                'reorder_atoms_to parameter should be "self" or "other"'
            )

        overlayed = deepcopy(other)
        if self.bonds is None:
            self.guess_bonds()
        if other.bonds is None:
            other.guess_bonds()

        graph_1 = from_scipy_sparse(csr_matrix(self.bonds))
        graph_2 = from_scipy_sparse(csr_matrix(other.bonds))
        if reorder_atoms_to == "self":
            gm = isomorphism.GraphMatcher(graph_1, graph_2)
        else:
            gm = isomorphism.GraphMatcher(graph_2, graph_1)

        if (
            not np.array_equal(np.sort(self.atomic_numbers), np.sort(other.atomic_numbers))
            or not gm.is_isomorphic()
        ):
            raise NotImplementedError(
                "Overlaying two different molecules not supported."
            )

        pos_1 = self.positions - self.center_of_mass
        pos_2 = other.positions - other.center_of_mass
        num_1 = self.atomic_numbers
        num_2 = other.atomic_numbers

        best_rmsd = np.inf
        best_order = None
        for mapping in gm.isomorphisms_iter():
            sorted_map = {i: mapping[i] for i in range(len(other))}
            order = list(sorted_map.values())
            if reorder_atoms_to == "self":
                rmsd = rmsd_points(pos_2[order], pos_1)
                equal = np.array_equal(num_1, num_2[order])
            else:
                rmsd = rmsd_points(pos_2, pos_1[order])
                equal = np.array_equal(num_1[order], num_2)
            if rmsd < best_rmsd and equal:
                best_rmsd = rmsd
                best_order = order

        if reorder_atoms_to == "self":
            overlayed.positions = reorient_points(pos_2[best_order], pos_1)
            if not isinstance(overlayed.labels, np.ndarray):
                overlayed.labels = np.array(overlayed.labels)
            overlayed.labels = overlayed.labels[best_order]
            overlayed.elements = [overlayed.elements[i] for i in best_order]
        else:
            overlayed.positions = reorient_points(pos_2, pos_1[best_order])
        overlayed.positions += self.center_of_mass
        return overlayed, best_order, best_rmsd

    def overlay_substructure(
        self,
        other: "Molecule",
        self_substructure: List[int],
        other_substructure: Union[List[int], None] = None,
    ):
        """Overlay substructure of other molecule onto substructure
        of this one.

        Parameters
        ----------
        other : cspy.chem.Molecule
            The other molecule to overlay onto this one.
        self_substructure : list of int
            The indices of the atoms of self that the substructure 
            of other will be overlayed onto.
        other_substructure : list of int, optional
            The indices of the atoms of other to be overlayed on to
            the substructure of self. If not given, assumed to be the
            same indices as self_substructure.

        Returns
        -------
        overlayed : Molecule
            A copy of other molecule transformed to overlay with the
            substructure of this molecule.
        rmsd : float
            RMSD of the substructure overlay.
        """
        from scipy.spatial.transform import Rotation

        if other_substructure is None:
            LOG.info("Assuming substructure indices are the same in both molecules.")
            other_substructure = self_substructure

        self_sub = Molecule(
            elements=[self.elements[i] for i in self_substructure],
            positions=self.positions[self_substructure],
            labels=self.labels[self_substructure],
        )
        other_sub = Molecule(
            elements=[other.elements[i] for i in other_substructure],
            positions=other.positions[other_substructure],
            labels=other.labels[other_substructure],
        )

        overlayed = deepcopy(other)
        if self_sub.bonds is None:
            self_sub.guess_bonds()
        if other_sub.bonds is None:
            other_sub.guess_bonds()

        graph_1 = from_scipy_sparse(csr_matrix(self_sub.bonds))
        graph_2 = from_scipy_sparse(csr_matrix(other_sub.bonds))
        gm = isomorphism.GraphMatcher(graph_1, graph_2)

        if (
            not np.array_equal(
                np.sort(self_sub.atomic_numbers),
                np.sort(other_sub.atomic_numbers),
            )
            or not gm.is_isomorphic()
        ):
            raise NotImplementedError(
                "Overlaying two different substructures not supported."
            )

        pos_1 = self_sub.positions - self_sub.centroid
        pos_2 = other_sub.positions - other_sub.centroid
        rotation, rmsd = Rotation.align_vectors(pos_1, pos_2)

        overlayed.positions -= other.centroid
        overlayed.positions = rotation.apply(overlayed.positions)
        overlayed.positions += other.centroid

        sub_diff = (
            np.mean(self.positions[self_substructure], axis=0)
            - np.mean(overlayed.positions[other_substructure], axis=0)
        )
        overlayed.positions += sub_diff

        return overlayed, rmsd

    def bond_path_separation(self):
        from scipy.sparse.csgraph import floyd_warshall
        from scipy.sparse import csr_matrix

        return floyd_warshall(csr_matrix(self.bonds.astype(bool))).astype(int)

    @property
    def bbox_corners(self):
        b_min = np.min(self.positions, axis=0)
        b_max = np.max(self.positions, axis=0)
        return b_min, b_max

    @property
    def bbox_size(self):
        if len(self) == 1:
            return np.array([self.elements[0].vdw] * 3)
        b_min, b_max = self.bbox_corners
        return np.abs(b_max - b_min)
    
    def to_mol2_string(self) -> str:
        """
        Convert the molecule to a mol2 string.
        """
        try:
            from openbabel import openbabel
        except ImportError:
            LOG.error("openbabel package is not installed. Try installing with 'pip install cspy[extra]'")
            sys.exit(1)

        ob_conversion = openbabel.OBConversion()
        ob_conversion.SetInAndOutFormats("xyz", "mol2")
        mol = openbabel.OBMol()
        ob_conversion.ReadString(mol, self.to_xyz_string())
        mol.AddHydrogens()
        mol2_coords = ob_conversion.WriteString(mol)
        return mol2_coords
    
    def to_mol2_file(self, filename) -> None:
        """
        Save the molecule to a mol2 file.
        
        Parameters
        ----------
        filename : str
            The filename to save the molecule to.
        """
        with open(filename, "w") as f:
            f.write(self.to_mol2_string())

    def to_rdkit_mol(self) -> RWMol:
        """
        Convert the molecule to an RDKit molecule.
        """
        from rdkit import Chem

        atoms = [x.symbol for x in self.elements]
        coords = self.positions
        mol = Chem.RWMol()
        for atom in atoms:
            mol.AddAtom(Chem.Atom(atom))
        conf = Chem.Conformer(len(atoms))
        for i, (x, y, z) in enumerate(coords):
            conf.SetAtomPosition(i, (float(x), float(y), float(z)))
        mol.AddConformer(conf)
        return mol

    def to_xyz_string(self, header=True):
        if header:
            lines = [
                f"{len(self)}",
                self.properties.get("comment", self.molecular_formula),
            ]
        else:
            lines = []
        for el, (x, y, z) in zip(self.elements, self.positions):
            lines.append(f"{el} {x: 20.12f} {y: 20.12f} {z: 20.12f}")
        return "\n".join(lines)

    def to_turbomole_string(self):
        tmol_template = "$coord angs\n" "{coords}\n"
        el = [x.symbol for x in self.elements]
        pos = self.positions
        coords = "\n".join(
            f"    {x:10.6f} {y:10.6f} {z:10.6f} {e}" for (x, y, z), e in zip(pos, el)
        )
        return tmol_template.format(coords=coords)

    @classmethod
    def from_turbomole_string(cls, coord_content):
        """Initialize from an xtb coord string resulting from optimization"""
        data = {}
        sections = coord_content.split("$")
        for section in sections:
            if not section or section.startswith("end"):
                continue
            lines = section.strip().splitlines()
            label = lines[0].strip()
            data[label] = [x.strip() for x in lines[1:]]
        elements = []
        positions = []
        key = "coord" if "coord" in data.keys() else "coord angs"
        for line in data[key]:
            x, y, z, el = line.split()
            positions.append((float(x), float(y), float(z)))
            elements.append(Element[el])
        pos = np.array(positions) * BOHR2ANGSTROM if key == 'coord' else positions
        return cls(elements, pos)

    @classmethod
    def from_zmatrix_file(cls, filename):
        from cspy.formats.zmatrix import parse_zmat_file

        return cls.from_zmatrix(parse_zmat_file(filename))

    @classmethod
    def from_zmatrix_string(cls, string, **kwargs):
        from cspy.formats.zmatrix import parse_zmat_string

        return cls.from_zmatrix(parse_zmat_string(string, **kwargs))

    @classmethod
    def from_zmatrix(cls, zmat):
        from scipy.spatial.transform import Rotation

        n = len(zmat)
        elements = []
        idxs = []
        angles = []
        dihedrals = []
        rs = []
        for x in zmat:
            elements.append(Element[x.element])
            rs.append((x.r if x.r is not None else 0.0))
            idxs.append((x.a if x.a else -1, x.b if x.b else -1, x.c if x.c else -1))
            angles.append(x.angle if x.angle is not None else 0.0)
            dihedrals.append(x.dihedral if x.dihedral is not None else 0.0)
        positions = np.zeros((n, 3))
        if n == 1:
            return cls(elements, positions)

        angles = np.radians(angles)
        dihedrals = np.radians(dihedrals)
        idxs = np.array(idxs) - 1
        rs = np.array(rs)

        # z-axis should be along vector from atoms 0 -> 1
        positions[1, 2] = rs[1]

        if n == 2:
            return cls(elements, positions)

        _, a, r, b, angle, _, _ = zmat[2]
        a, b = idxs[2, :2]
        v1 = positions[b, :] - positions[a, :]
        v1 /= np.linalg.norm(v1)
        d = r * v1
        # rotate about the y axis by angle (i.e. in the xz-plane)
        axis = np.cross(v1, (1, 0, 0))
        rot = Rotation.from_rotvec(axis * np.radians(angle))
        d = rot.apply(d)
        positions[2, :] = d + positions[a, :]

        if n == 3:
            return cls(elements, positions)

        cos_dihedrals = np.cos(dihedrals)
        sin_dihedrals = np.sin(dihedrals)
        sin_angles = np.sin(angles)
        cos_angles = np.cos(angles)
        for i in range(3, n):
            a, b, c = idxs[i]
            r = rs[i]
            v1 = positions[a] - positions[b]
            v2 = positions[a] - positions[c]
            n = np.cross(v1, v2)
            nn = np.cross(v1, n)
            n /= np.linalg.norm(n)
            nn /= np.linalg.norm(nn)
            n *= -sin_dihedrals[i]
            nn *= cos_dihedrals[i]
            v3 = n + nn
            v3 /= np.linalg.norm(v3)
            v3 *= r * sin_angles[i]
            v1 /= np.linalg.norm(v1)
            v1 *= r * cos_angles[i]
            positions[i, :] = positions[a] + v3 - v1
        return cls(elements, positions)

    @property
    def asym_symops(self):
        return self.properties.get("generator_symop", [16484] * len(self))

    def to_xyz_file(self, filename):
        with open(filename, "w") as f:
            f.write(self.to_xyz_string())

    def to_dash_data(self):
        """convert this to a dict representation useable by dash etc"""
        return [
            {"symbol": e.symbol, "x": xyz[0], "y": xyz[1], "z": xyz[2]}
            for e, xyz in zip(self.elements, self.positions)
        ]

    def to_zmatrix(self):
        from cspy.formats.zmatrix import to_zmat

        return to_zmat(self)

    def to_zmatrix_string(self):
        from cspy.formats.zmatrix import to_zmat_string

        return to_zmat_string(self)

    def to_zmatrix_file(self, filename):
        from cspy.util.path import Path

        Path(filename).write_text(self.to_zmatrix_string())

    def calculate_multipoles(self, **kwargs):
        from cspy.minimize.gaussian_minimizer import GaussianMinimizer
        from cspy.executable.gdma import Gdma
        from tempfile import TemporaryDirectory
        from cspy.util.path import Path
        from cspy.chem.multipole import DistributedMultipoles

        kwargs.pop("opt", False)
        with TemporaryDirectory() as tmpdirname:
            m = GaussianMinimizer(opt=False, **kwargs)
            fchk = Path(tmpdirname, self.molecular_formula + ".fchk").absolute()
            m.minimize_molecule(self, fchk_file=fchk)
            gdma = Gdma(fchk, **kwargs)
            gdma.timeout = 3600.0
            gdma.working_directory = tmpdirname
            gdma.run()
        mults = DistributedMultipoles.from_dma_string(gdma.punch_contents)
        self.properties["multipoles"] = mults.molecules[0].multipoles
        return mults

    @classmethod
    def from_arrays(cls, elements, positions, **kwargs):
        return cls([Element[x] for x in elements], np.array(positions), **kwargs)

    def to_crystal(self, vacuum: float = 20) -> "Crystal":
        """Convert the molecule to a crystal object with a vacuum gap.
        This is usually used to generate inputs for molecular calculations in periodic
        DFT codes.

        Parameters
        ----------
        vacuum : float, optional
            The vacuum gap to add to the molecule. Default is 20 Angstrom.

        Returns
        -------
        Crystal
            A Crystal object with the molecule in a big box.
        """
        from cspy.crystal import Crystal, UnitCell, SpaceGroup, AsymmetricUnit
        # Crystal object has to be loaded here, otherwise it will cause circular import issues

        min_pos = np.min(self.positions, axis=0)
        max_pos = np.max(self.positions, axis=0)
        lattice_dims = max_pos - min_pos + vacuum
        lattice_vectors = np.array([
            [lattice_dims[0], 0, 0],
            [0, lattice_dims[1], 0],
            [0, 0, lattice_dims[2]],
        ])
        unit_cell = UnitCell(lattice_vectors)
        space_group = SpaceGroup(international_tables_number=1)
        asymmetric_unit = AsymmetricUnit(
            elements=self.elements,
            positions=unit_cell.to_fractional(self.positions),
            labels=self.labels,
        )
        crystal = Crystal(
            unit_cell=unit_cell,
            space_group=space_group,
            asymmetric_unit=asymmetric_unit,
        )
        crystal.titl = self.molecular_formula
        return crystal

    def to_poscar_file(self, vacuum: float = 20, filename: str = "POSCAR") -> None:
        """Put the molecule in a box with vacuum in all directions, then
        write the box in POSCAR format.

        Parameters
        ----------
        vacuum : float, optional
            The vacuum gap to add to the molecule. Default is 20 Angstrom.
        filename : str, optional
            The name of the file to write the POSCAR to. Default is 'POSCAR'.

        Returns
        -------
        str
            A POSCAR string with the molecule in a big box.
        """
        crystal = self.to_crystal(vacuum=vacuum)
        return crystal.to_poscar_file(filename=filename)

    def get_symmetry_equivalent_atoms(self, tol: float = 0.3) -> List[List[int]]:
        """Uses pymatgen to get the symmetry operations and then find the equivalent atoms.

        Parameters
        ----------
        tol : float, optional
            The tolerance for the equivalent atoms. Default is 0.3.

        Returns
        -------
        List[List[int]]
            A list of lists of equivalent atoms.
        """
        from pymatgen.io.xyz import XYZ
        from pymatgen.symmetry import analyzer

        pymat_mol = XYZ.from_str(self.to_xyz_string()).molecule
        analyser = analyzer.PointGroupAnalyzer(pymat_mol)
        operators = analyser.get_symmetry_operations()

        original_positions = self.positions.copy()
        equivalent_atoms = []

        for operator in operators:
            opt_equivalent_atoms = []
            self.positions = original_positions.copy()
            rotation_matrix = operator.rotation_matrix
            translation_vector = operator.translation_vector
            self.rotate_about_com(
                alpha=None,
                beta=None,
                gamma=None,
                rotation_matrix=rotation_matrix
            )
            self.translate(translation_vector)
            for ind0, atom_pos in enumerate(original_positions):
                elem0 = self.elements[ind0]
                dif_pos = self.positions - atom_pos
                matching_coords = (dif_pos >= -tol) & (dif_pos <= tol)
                matching_atom = None
                for ind1, bool_pos in enumerate(matching_coords):
                    elem1 = self.elements[ind1]
                    if elem1 == elem0 and np.all(bool_pos):
                        matching_atom = ind1
                        break
                opt_equivalent_atoms.append(matching_atom)
            equivalent_atoms.append(opt_equivalent_atoms)

        self.equivalent_atoms = equivalent_atoms
        return equivalent_atoms

    def exchange_symmetry_equivalent_atoms(self, equivalent_atoms: Dict[int, int]) -> None:
        """Exchange the atoms in the molecule according to the equivalent atoms.

        Parameters
        ----------
        equivalent_atoms : Dict[int, int]
            A dictionary where keys are indices of atoms in the molecule and values are indices of equivalent atoms.
        """
        old_positions = self.positions
        new_positions = deepcopy(old_positions)

        for atom in equivalent_atoms.keys():
            new_positions[int(atom)] = old_positions[equivalent_atoms[atom]]

        self.positions = new_positions

    def rotate_about_point(
        self,
        point: np.ndarray,
        alpha: float,
        beta: float,
        gamma: float,
        rotation_matrix: Union[np.ndarray, None] = None,
    ) -> None:
        """Rotate the molecule about a point in space. In order
        alpha, beta, gamma.

        Parameters
        ----------
        point : np.ndarray
            The point to rotate about.
        alpha : float
            The angle to rotate about the x-axis (radians).
        beta : float
            The angle to rotate about the y-axis (radians).
        gamma : float
            The angle to rotate about the z-axis (radians).
        rotation_matrix : np.ndarray, optional
            The rotation matrix to use. Default is None.

        Returns
        -------
        None
        """
        from math import cos, sin

        if rotation_matrix is None and None in (alpha, beta, gamma):
            raise ValueError("If rotation_matrix is None, alpha, beta and gamma must be provided.")

        if rotation_matrix is None:
            cos_a = cos(alpha)
            cos_b = cos(beta)
            cos_g = cos(gamma)
            sin_a = sin(alpha)
            sin_b = sin(beta)
            sin_g = sin(gamma)

            rotation_matrix = np.array([
                [
                    cos_b * cos_g,
                    (sin_a * sin_b * cos_g) - (cos_a * sin_g),
                    (cos_a * sin_b * cos_g) + (sin_a * sin_g),
                ],
                [
                    cos_b * sin_g,
                    (sin_a * sin_b * sin_g) + (cos_a * cos_g),
                    (cos_a * sin_b * sin_g) - (sin_a * cos_g),
                ],
                [
                    -sin_b,
                    sin_a * cos_b,
                    cos_a * cos_b,
                ],
            ])

        self.translate(-point)
        self.positions = np.dot(self.positions, rotation_matrix)
        self.translate(point)

    def rotate_about_centroid(
        self,
        alpha: float,
        beta: float,
        gamma: float,
        rotation_matrix: Union[np.ndarray, None] = None,
    ) -> None:
        """Rotate the molecule about its centroid.

        Parameters
        ----------
        alpha : float
            The angle to rotate about the x-axis (radians).
        beta : float
            The angle to rotate about the y-axis (radians).
        gamma : float
            The angle to rotate about the z-axis (radians).
        rotation_matrix : np.ndarray, optional
            The rotation matrix to use. Default is None.

        Returns
        -------
        None
        """
        point = self.centroid
        self.rotate_about_point(
            point=point,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            rotation_matrix=rotation_matrix,
        )

    def rotate_about_com(
        self,
        alpha: float,
        beta: float,
        gamma: float,
        rotation_matrix: Union[np.ndarray, None] = None,
    ) -> None:
        """Rotate the molecule about its center of mass.

        Parameters
        ----------
        alpha : float
            The angle to rotate about the x-axis (radians).
        beta : float
            The angle to rotate about the y-axis (radians).
        gamma : float
            The angle to rotate about the z-axis (radians).
        rotation_matrix : np.ndarray, optional
            The rotation matrix to use. Default is None.

        Returns
        -------
        None
        """
        point = self.center_of_mass
        self.rotate_about_point(
            point=point,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            rotation_matrix=rotation_matrix,
        )

    @property
    def amd_descriptor(self) -> np.ndarray:
        """Calculate the average minimum distance (AMD) descriptor for the molecule.

        Returns
        -------
        np.ndarray
            The AMD descriptor as a numpy array.
        """
        distance_mat = distance_matrix(self.positions, self.positions)
        return np.mean(np.sort(distance_mat, axis=1), axis=0)
    
    def __hash__(self) -> int:
        """Hash the molecule based on its AMD descriptor and atomic numbers."""
        return hash((tuple(self.amd_descriptor.round(3)), tuple(sorted(self.atomic_numbers))))
    
    def get_best_plane_axes(self) -> np.ndarray:
        """Find the best fit plane for a set of 3D points.
        
        Returns
        -------
        plane_points : np.ndarray
            Three points defining the best fit plane.
        """
        best_plane_axes = np.linalg.svd(self.positions - self.centroid)[2]
        return best_plane_axes
    
    def rotate_to_best_plane(self) -> None:
        """Rotate the molecule to the best fit plane."""
        best_plane_axes = self.get_best_plane_axes()
        self.positions = np.dot(
            self.positions - self.centroid, 
            best_plane_axes.T)

    def get_vdw_volume(self, d_grid: float = 0.1, rotate_to_best_plane: bool = True) -> float:
        """
        Calculate the van der Waals volume of the molecule.

        Parameters
        ----------
        d_grid : float, optional
            Grid spacing for volume calculation. Default is 0.1.

        Returns
        -------
        vdw_volume : float
            Van der Waals volume of the molecule.
        """
        if rotate_to_best_plane:
            self.rotate_to_best_plane()
        b_min, b_max = self.bbox_corners
        b_min -= np.ones(3) * 3
        b_max += np.ones(3) * 3
        x_grid, y_grid, z_grid = np.mgrid[
            b_min[0]:b_max[0]:d_grid, 
            b_min[1]:b_max[1]:d_grid, 
            b_min[2]:b_max[2]:d_grid
        ]
        grid_points = np.vstack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]).T
        distances = cdist(grid_points, self.positions)
        within_vdw = np.any(distances < self.vdw_radii, axis=1)
        vdw_volume = np.sum(within_vdw) * d_grid**3
        return vdw_volume

