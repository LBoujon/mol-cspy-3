import logging
import os
from collections import defaultdict
from typing import List, Tuple, Union, Literal, TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree as KDTree
from scipy.sparse import dok_matrix
import scipy.sparse.csgraph as csgraph

from ase import Atoms

from cspy.util.path import Path
from cspy.formats.cif import Cif
from cspy.formats.pmin_save import PminSave
from cspy.formats.xyz import parse_xyz_file
from cspy.formats.shelx import (
    parse_shelx_file_content,
    to_res_contents,
)
from .unit_cell import UnitCell
from .space_group import SpaceGroup
from .symmetry_operation import SymmetryOperation
from cspy.linalg import cartesian_product
from cspy.chem import Element, chemical_formula, Molecule
from cspy.chem.multipole import Multipole, DistributedMultipoles
from ase import Atoms

if TYPE_CHECKING:
    from cspy.minimize import ASEMinimizer

np.set_printoptions(threshold=np.inf)

LOG = logging.getLogger(__name__)


class AsymmetricUnit:
    """Storage class for the coordinates and labels in a crystal
    asymmetric unit
    Create an asymmetric unit object from a list of Elements and 
    an array of fractional coordinates.

    Parameters
    ----------
    elements : :obj:`list` of :obj:`Element`
        N length list of elements associated with the sites in this asymmetric
        unit
    positions : array_like
        (N, 3) array of site positions in fractional coordinates
    labels : array_like
        N length array of string labels for each site
    **kwargs
        Additional properties (will populate the properties member)
        to store in this asymmetric unit
    """

    def __init__(self, elements, positions, labels=None, **kwargs):
        self.elements = elements
        self.atomic_numbers = np.asarray([x.atomic_number for x in elements])
        self.positions = np.asarray(positions)
        self.properties = {}
        self.properties.update(kwargs)
        if labels is None:
            self.labels = []
            label_index = defaultdict(int)
            for el in self.elements:
                label_index[el] += 1
                self.labels.append("{}{}".format(el, label_index[el]))
        else:
            self.labels = labels
        self.labels = np.array(self.labels)

    @classmethod
    def from_records(cls, records):
        """Initialize an AsymmetricUnit from a list of dictionary like objects
        
        Parameters
        ----------
        records : iterable
            An iterable containing dict_like objects with `label`,
            `element`, `position` and optionally `occupation` stored.
        """
        labels = []
        elements = []
        positions = []
        occupation = []
        for r in records:
            labels.append(r["label"])
            elements.append(Element[r["element"]])
            positions.append(r["position"])
            occupation.append(r.get("occupation", 1.0))
        positions = np.asarray(positions)
        return cls(elements, positions, labels=labels, occupation=occupation)

    @property
    def formula(self):
        """Molecular formula for this asymmetric unit

        Returns
        -------
        str
            The chemical formula of this asymmetric unit e.g.
            H24O12 or similar.
        """
        return chemical_formula(self.elements, subscript=False)

    def remove_duplicate_labels(self):
        new_labels = []
        counts = defaultdict(int)
        for lbl in self.labels:
                el = lbl.rstrip('0123456789')
                counts[el] += 1
                new_labels.append("{}{}".format(el, counts[el]))
        self.labels = np.asarray(new_labels)		

    def __len__(self):
        return len(self.elements)

    def __repr__(self):
        return "<{}>".format(self.formula)


class Crystal:
    """Storage class for a crystal structure, consisting of
    an asymmetric unit, a unit cell and space group information.

    Parameters
    ----------
    unit_cell : :obj:`UnitCell`
        The unit cell for this crystal i.e. the translational symmetry
        of the crystal structure.
    space_group : :obj:`SpaceGroup`
        The space group symmetry of this crystal i.e. the generators
        for populating the unit cell given the asymmetric unit.
    asymmetric_unit : :obj:`AsymmetricUnit`
        The asymmetric unit of this crystal. The sites of this
        combined with the space group will generate all translationally
        equivalent positions.
    **kwargs
        Optional properties to (will populate the properties member) store
        about the the crystal structure.
    """

    space_group: SpaceGroup
    unit_cell: UnitCell
    asymmetric_unit: AsymmetricUnit
    properties: dict

    def __init__(self, unit_cell, space_group, asymmetric_unit, **kwargs):
        self.space_group = space_group
        self.unit_cell = unit_cell
        self.asymmetric_unit = asymmetric_unit
        self.properties = {}
        self.properties.update(kwargs)

    @property
    def site_positions(self):
        """Row major array of asymmetric unit atomic positions

        Returns
        -------
        array_like
            The positions in fractional coordinates of the asymmetric unit.
        """
        return self.asymmetric_unit.positions

    @property
    def site_atoms(self):
        """Array of asymmetric unit atomic numbers

        Returns
        -------
        array_like
            The atomic numbers of the asymmetric unit.
        """
        return self.asymmetric_unit.atomic_numbers

    @property
    def nsites(self):
        """The number of sites in the asymmetric unit.

        Returns
        -------
        int
            The number of sites in the asymmetric unit.
        """
        return len(self.site_atoms)

    @property
    def symmetry_operations(self):
        """Symmetry operations that generate this crystal.

        Returns
        -------
        :obj:`list` of :obj:`SymmetryOperation`
            List of SymmetryOperation objects belonging to the space group
            symmetry of this crystal.
        """
        return self.space_group.symmetry_operations

    def to_cartesian(self, coords):
        """Convert coordinates (row major) from fractional to cartesian coordinates.

        Parameters
        ----------
        coords : array_like
            (N, 3) array of positions assumed to be in fractional coordinates

        Returns
        -------
        array_like
            (N, 3) array of positions transformed to cartesian (orthogonal) coordinates
            by the unit cell of this crystal.
        """
        return self.unit_cell.to_cartesian(coords)

    def to_fractional(self, coords):
        """Convert coordinates (row major) from cartesian to fractional coordinates.

        Parameters
        ----------
        coords : array_like
            (N, 3) array of positions assumed to be in cartesian (orthogonal) coordinates

        Returns
        -------
        array_like
            (N, 3) array of positions transformed to fractional coordinates
            by the unit cell of this crystal.
        """
        return self.unit_cell.to_fractional(coords)

    def unit_cell_atoms(self, tolerance=1e-3):
        """Generate all atoms in the unit cell (i.e. with 
        fractional coordinates in [0, 1]) along with associated
        information about symmetry operations, occupation, elements
        related asymmetric_unit atom etc.
        
        Will merge atom sites within tolerance of each other, and
        sum their occupation numbers. A warning will be logged if
        any atom site in the unit cell has > 1.0 occupancy after
        this.

        Sets the `_unit_cell_atom_dict` member as this is an expensive
        operation and is worth caching the result. Subsequent calls
        to this function will be a no-op.

        Parameters
        ----------
        tolerance : float, optional
            Minimum separation of sites in the unit cell, below which 
            atoms/sites will be merged and their (partial) occupations
            added.

        Returns
        -------
        dict
            A dictionary of arrays associated with all sites contained
            in the unit cell of this crystal, members are:

            asym_atom: corresponding asymmetric unit atom indices for all sites.

            frac_pos: (N, 3) array of fractional positions for all sites.

            cart_pos: (N, 3) array of cartesian positions for all sites.

            element: (N) array of atomic numbers for all sites.

            symop: (N) array of indices corresponding to the generator symmetry
            operation for each site.

            label: (N) array of string labels corresponding to each site
            occupation: (N) array of occupation numbers for each site. Will
            warn if any of these are greater than 1.0
        """
        if hasattr(self, "_unit_cell_atom_dict"):
            return getattr(self, "_unit_cell_atom_dict")
        pos = self.site_positions
        atoms = self.site_atoms
        natom = self.nsites
        nsymops = len(self.space_group.symmetry_operations)
        occupation = np.tile(
            self.asymmetric_unit.properties.get("occupation", np.ones(natom)), nsymops
        )
        labels = np.tile(self.asymmetric_unit.labels, nsymops)
        uc_nums = np.tile(atoms, nsymops)
        asym = np.arange(len(uc_nums)) % natom
        sym, uc_pos = self.space_group.apply_all_symops(pos)
        # translate molecules so atoms are inside center cell
        translated = np.fmod(uc_pos + 7.0, 1)
        # translated = uc_pos
        tree = KDTree(translated)
        dist = tree.sparse_distance_matrix(tree, max_distance=tolerance)
        mask = np.ones(len(uc_pos), dtype=bool)
        # because crystals may have partially occupied sites
        # on special positions, we need to merge some sites
        expected_natoms = np.sum(occupation)
        for (i, j), d in dist.items():
            if not (i < j):
                continue
            occupation[i] += occupation[j]
            mask[j] = False
        occupation = occupation[mask]
        if not np.isclose(np.sum(occupation), expected_natoms):
            LOG.warning("invalid total occupation after merging sites")
        if np.any(occupation > 1.0):
            LOG.warning("Some unit cell site occupations are > 1.0")
        setattr(
            self,
            "_unit_cell_atom_dict",
            {
                "asym_atom": asym[mask],
                "frac_pos": translated[mask],
                "element": uc_nums[mask],
                "symop": sym[mask],
                "label": labels[mask],
                "occupation": occupation,
                "cart_pos": self.to_cartesian(translated[mask]),
            },
        )
        return self._unit_cell_atom_dict

    def unit_cell_connectivity(self, tolerance=0.4, neighbouring_cells=1):
        """Periodic connectiviy for the unit cell, populates _uc_graph
        with a networkx.Graph object, where nodes are indices into the
        _unit_cell_atom_dict arrays and the edges contain the translation
        (cell) for the image of the corresponding unit cell atom with the
        higher index to be bonded to the lower

        Bonding is determined by interatomic distances being less than the
        sum of covalent radii for the sites plus the tolerance (provided 
        as a parameter)
        
        Parameters
        ----------
        tolerance : float, optional
            Bonding tolerance (bonded if d < cov_a + cov_b + tolerance)
        neighbouring_cells : int, optional
            Number of neighbouring cells in which to look for bonded atoms.
            We start at the (0, 0, 0) cell, so a value of 1 will look in the
            (0, 0, 1), (0, 1, 1), (1, 1, 1) i.e. all 26 neighbouring cells.
            1 is typically sufficient for organic systems.

        Returns
        -------
        :obj:`tuple` of (sparse_matrix in dict of keys format, dict)
            the (i, j) value in this matrix is the bond length from i,j
            the (i, j) value in the dict is the cell translation on j which
            bonds these two sites
        """
        if hasattr(self, "_uc_graph"):
            return getattr(self, "_uc_graph")

        slab = self.slab(bounds=((-1, -1, -1), (1, 1, 1)))
        n_uc = slab["n_uc"]
        uc_pos = slab["frac_pos"][:n_uc]
        uc_nums = slab["element"][:n_uc]
        neighbour_pos = slab["frac_pos"][n_uc:]
        cart_uc_pos = self.to_cartesian(uc_pos)
        unique_elements = {x: Element.from_atomic_number(x) for x in np.unique(uc_nums)}

        # first establish all connections in the unit cell
        covalent_radii = np.array([unique_elements[x].cov for x in uc_nums])
        symbols = np.array([unique_elements[x].symbol for x in uc_nums])
        max_cov = np.max(covalent_radii)

        # TODO this needs to be sped up for large cells, tends to slow for > 1000 atoms
        # and the space storage will become a problem
        tree = KDTree(cart_uc_pos)
        dist = tree.sparse_distance_matrix(tree, max_distance=2 * max_cov + tolerance)

        uc_edges = []
        for (i, j), d in dist.items():
            if not (i < j):
                continue
            if "bondlength_cutoffs" in self.properties:
                key = tuple(sorted((symbols[i], symbols[j])))
                bond_cutoff = self.properties["bondlength_cutoffs"].get(key, 0)
            else:
                bond_cutoff = covalent_radii[i] + covalent_radii[j] + tolerance
            if 1e-3 < d <= bond_cutoff:
                uc_edges.append((i, j, d, (0, 0, 0)))

        cart_neighbour_pos = self.unit_cell.to_cartesian(neighbour_pos)
        tree2 = KDTree(cart_neighbour_pos)
        dist = tree.sparse_distance_matrix(tree2, max_distance=2 * max_cov + tolerance)
        # could be sped up if done outside python
        cells = slab["cell"][n_uc:]
        for (uc_atom, neighbour_atom), d in dist.items():
            uc_idx = neighbour_atom % n_uc
            if not (uc_atom < uc_idx):
                continue
            if "bondlength_cutoffs" in self.properties:
                key = tuple(sorted((symbols[uc_atom], symbols[uc_idx])))
                bond_cutoff = self.properties["bondlength_cutoffs"].get(key, 0)
            else:
                bond_cutoff = covalent_radii[uc_atom] + covalent_radii[uc_idx] + tolerance
            if 1e-3 < d <= bond_cutoff:
                cell = cells[neighbour_atom]
                uc_edges.append((uc_atom, uc_idx, d, tuple(cell)))

        properties = {}
        uc_graph = dok_matrix((n_uc, n_uc))
        for i, j, d, cell in uc_edges:
            uc_graph[i, j] = d
            properties[(i, j)] = cell

        setattr(self, "_uc_graph", (uc_graph, properties))
        return self._uc_graph

    @property
    def titl(self):
        """The titl (i.e. name) of this crystal

        Returns
        -------
        str
            the value of "titl" in properties (if set) otherwise the chemical formula
            of the asymmetric_unit
        """
        return self.properties.get("titl", self.asymmetric_unit.formula)

    @titl.setter
    def titl(self, value):
        """Set the titl (i.e. name) of this crystal

        Parameters 
        -------
        value : str
            The name of this crystal.
        """
        self.properties["titl"] = value

    def unit_cell_molecules(self,conn_tolerance=0.4):
        """Calculate the molecules for all sites in the unit cell,
        where the number of molecules will be equal to number of
        symmetry unique molecules times number of symmetry operations.
        Parameters
        ----------
        conn_tolerance : float, optional
            Bonding tolerance (bonded if d < cov_a + cov_b + tolerance) to be used when asessing unit cell connectivity

        Returns
        -------
        :obj:`list` of :obj:`Molecule`
            List of all connected molecules in this crystal, which
            when translated by the unit cell would produce the full crystal.
            If the asymmetric is molecular, the list will be of length
            num_molecules_in_asymmetric_unit * num_symm_operations
        """
        if hasattr(self, "_unit_cell_molecules"):
            return getattr(self, "_unit_cell_molecules")
        uc_graph, edge_cells = self.unit_cell_connectivity(tolerance=conn_tolerance)
        n_uc_mols, uc_mols = csgraph.connected_components(
            csgraph=uc_graph, directed=False, return_labels=True
        )
        uc_frac = self._unit_cell_atom_dict["frac_pos"]
        uc_cartesian = self._unit_cell_atom_dict["cart_pos"]
        uc_elements = self._unit_cell_atom_dict["element"]
        uc_asym = self._unit_cell_atom_dict["asym_atom"]
        uc_symop = self._unit_cell_atom_dict["symop"]

        molecules = []

        n_uc = len(uc_frac)
        LOG.debug("%d molecules in unit cell", n_uc_mols)
        for i in range(n_uc_mols):
            nodes = np.where(uc_mols == i)[0]
            root = nodes[0]
            elements = uc_elements[nodes]
            shifts = np.zeros((n_uc, 3))
            ordered, pred = csgraph.breadth_first_order(
                csgraph=uc_graph, i_start=root, directed=False
            )
            for j in ordered[1:]:
                i = pred[j]
                if j < i:
                    shifts[j, :] = shifts[i, :] - edge_cells[(j, i)]
                else:
                    shifts[j, :] = shifts[i, :] + edge_cells[(i, j)]
            positions = self.to_cartesian((uc_frac + shifts)[nodes])
            asym_atoms = uc_asym[nodes]
            reorder = np.argsort(asym_atoms)
            asym_atoms = asym_atoms[reorder]

            mol = Molecule.from_arrays(
                elements=elements[reorder],
                positions=positions[reorder],
                guess_bonds=True,
                unit_cell_atoms=np.array(nodes)[reorder],
                asymmetric_unit_atoms=asym_atoms,
                asymmetric_unit_labels=self.asymmetric_unit.labels[asym_atoms],
                generator_symop=uc_symop[np.asarray(nodes)[reorder]],
            )
            molecules.append(mol)
        setattr(self, "_unit_cell_molecules", molecules)
        return molecules

    def asym_mols(self):
        """Alias for symmetry_unique_molecules()"""
        return self.symmetry_unique_molecules()

    def unique_components(self, rmsd_tol=0.2):
        """Gets the unqiue components of the crystal
        Returns:
        :list of cspy.chem.Molecules:
            list containing the unique components 
        :list of lists of int:
            list of lists containing indices of equivalent
            molecules in asym_mols
        """
        mols = self.asym_mols()
        equiv_mols = { i: [i] for i, _ in enumerate(mols)}

        for i, mol in enumerate(mols):
            if i not in equiv_mols.keys():
                continue
            comps = [ j for j in equiv_mols.keys() if j > i ]
            for j in comps:
                comp_mol = mols[j]
                try:
                    _, _, rmsd = mol.overlay(comp_mol)
                except:
                    continue
                if rmsd < rmsd_tol:
                    equiv_mols[i].extend(equiv_mols[j])
                    equiv_mols.pop(j)

        return (
            [ mols[i] for i in equiv_mols.keys() ],
            [ x for x in equiv_mols.values() ], 
        )

    def replace_molecules(
        self,
        molecules: List[Molecule],
        how: Literal["best_match"] = "best_match",
        reorder_atoms_to: Literal["self", "other"] = "self",
        reorder_mols: bool = False,
        method: Literal["original", "rdkit", "shake_and_overlay_by_rdkit"] = "original",
        expansion_attempt: int = 0,
        max_expansion_attempts: int = 3,
        **kwargs,
    ) -> "Crystal":
        """Attempt to replace the molecules in the asymmetric_unit with
        those provided -- akin to 'pasting' in different molecular
        geometries.

        Parameters
        ----------
        molecules : List[Molecule]
            The candidate molecules for replacing those in this crystal
        how : str, optional
            The method for determining which molecules to replace with
            those in the list. Default is 'best_match'
        reorder_atoms_to : str, optional
            Reorder the atoms of the overlap to match self or other. Default is 'self'
        reorder_mols : bool, optional
            Reorder molecules in crystal to be match order of replacement molecules. Default is 'False'
        method : str, optional
            The method to use for overlaying the molecules ('original', 'rdkit', or 
            'shake_and_overlay_by_rdkit'). Default is 'original'
        expansion_attempt : int, optional
            The number of times the unit cell has been expanded to avoid clashes. Default is 0.
        max_expansion_attempts : int, optional
            Maximum number of times to attempt to expand the unit cell to avoid clashes.
            Default is 3.

        kwargs : dict
            Additional keyword arguments to pass to the Molecule.overlay_by_rdkit method
            used if method is 'rdkit'. Please refer to the documentation of Molecule.overlay_by_rdkit
            for more information.

        Returns
        -------
        Crystal
            Modified version of this crystal with the molecules replaced.
        """

        if method not in ["original", "rdkit", "shake_and_overlay_by_rdkit"]:
            raise ValueError("method must be original or rdkit")

        if hasattr(molecules, "positions"):
            molecules = [molecules]
        if how != "best_match":
            raise NotImplementedError(
                "no other methods implemented for molecule replacement"
            )
        from copy import deepcopy

        reference_mols = self.asym_mols()
        best_matches = [(-1, None, 1e100)] * len(reference_mols)
        for i, ref in enumerate(reference_mols):
            found_a_match = False
            for j, mol in enumerate(molecules):
                try:
                    if method == "original":
                        overlayed, order, rmsd = ref.overlay(
                            mol, reorder_atoms_to=reorder_atoms_to
                        )
                    elif method == "rdkit":
                        overlayed, order, rmsd = ref.overlay_by_rdkit(
                            mol, reorder_atoms_to=reorder_atoms_to, **kwargs
                        )
                        LOG.debug(
                            f"rdkit rmsd between mol {i} of reference and mol {j} of target: {rmsd}"
                        )
                    elif method == "shake_and_overlay_by_rdkit":
                        LOG.warning(
                            "shake_and_overlay_by_rdkit reorders the atoms "
                            "of the molecules to match the other molecule"
                        )
                        overlayed, order, rmsd = ref.shake_and_overlay(
                            mol, **kwargs
                        )
                        LOG.debug(
                            f"rdkit rmsd between mol {i} of reference and mol {j} of target: {rmsd}"
                        )
                except Exception as e:
                    LOG.debug(f"{e}")
                    continue

                if best_matches[i][2] > rmsd:
                    found_a_match = True
                    best_matches[i] = (j, overlayed, rmsd)
            if not found_a_match:
                raise ValueError(f"no match found for molecule # {i}: {ref}")

        LOG.info(
            "RMSDs of best matches: \n%s",
            "\n".join(
                f"old {i} -> new {x[0]} {x[2]:.3f}" for i, x in enumerate(best_matches)
            ),
        )
        # order molecules by order of provided molecules, not by order from reference crystal
        best_match_js = [match[0] for match in best_matches]
        # best_matches = [match for _, match in sorted(zip(best_match_js, best_matches))]
        # The above line sometimes fails if the molecule ordering is inconsistent. The mol_inds list addresses this problem
        # It also provides the option for different sorting approaches
        mol_inds = [n for n in range(len(best_matches))]
        if reorder_mols:
            mol_inds = [ind for _, ind in sorted(zip(best_match_js, mol_inds))]
        else:
            mol_inds = [ind for ind, _ in sorted(zip(mol_inds, best_match_js))]
        best_matches = [best_matches[ind] for ind in mol_inds]
        new_asym_mols = [x[1] for x in best_matches]
        asym_pos = np.vstack([x.positions for x in new_asym_mols])
        asym_nums = np.hstack([x.atomic_numbers for x in new_asym_mols])
        asymmetric_unit = AsymmetricUnit(
            [Element[x] for x in asym_nums], self.to_fractional(asym_pos)
        )
        replaced_crystal = Crystal(
            deepcopy(self.unit_cell), deepcopy(self.space_group), asymmetric_unit
        )

        # Checking for molecular clashes
        if len(self.asym_mols()) != len(replaced_crystal.asym_mols()):
            LOG.warning("The number of molecules in the asymmetric unit has changed.")
            LOG.warning(f"Before replacement: {len(self.asym_mols())} molecules.")
            LOG.warning(f"After replacement: {len(replaced_crystal.asym_mols())} molecules.")
            LOG.warning("After replacement, the molecules in the asymmetric unit are:")
            for i, mol in enumerate(replaced_crystal.asym_mols()):
                LOG.warning(f"molecule-{i+1}: {mol}")
            LOG.warning("This may be due to clash of molecules after replacement.")

            if expansion_attempt + 1 > max_expansion_attempts:
                LOG.warning(
                    f"Maximum expansion attempts ({max_expansion_attempts}) "
                    "reached. Returning the last replaced crystal."
                )
            else:
                LOG.warning(
                    f"Expanding unit cell by 5% to avoid clashes: "
                    f"attempt #{expansion_attempt}"
                )
                expanded_cell = self.resize_volume(P=0.05)
                replaced_crystal = expanded_cell.replace_molecules(
                    deepcopy(molecules),
                    how=how,
                    reorder_atoms_to=reorder_atoms_to,
                    reorder_mols=reorder_mols,
                    method=method,
                    expansion_attempt=expansion_attempt + 1,
                    **kwargs,
                )
        return replaced_crystal

    def unique_mols(self) -> List[Molecule]:
        """Get the geometrically and compositionally unique molecules from the
        asymmetric unit of the crystal.

        Returns
        -------
        List[Molecule]
            List of all unique molecules in the asymmetric unit of this crystal.
        """
        return List(set(self.asym_mols()))

    def replace_molecules_by_mols_of(
        self,
        other: "Crystal",
        how: Literal["best_match"] = "best_match",
        reorder_atoms_to: Literal["self", "other"] = "other",
        reorder_mols: bool = False,
        method: Literal["original", "rdkit", "shake_and_overlay_by_rdkit"] = "rdkit",
        **kwargs,
    ) -> "Crystal":
        """Attempt to replace the molecules in the asymmetric unit of self with unique molecules
        in the asymmetric unit of other crystal.

        Parameters and Returns are the same as replace_molecules() except for the following:
        ----------
        other : Crystal
            The crystal from which to take the unique molecules
        how : str, optional
            The method for determining which molecules to replace with those in the list. Default is 'best_match'
        reorder_atoms_to : str, optional
            Reorder the atoms of the overlap to match self or other. Default is 'other'
        reorder_mols : bool, optional
            Reorder molecules in crystal to match order of replacement molecules. Default is 'False'
        method : str, optional
            The method to use for overlaying the molecules ('original', 'rdkit', or 
            'shake_and_overlay_by_rdkit'). Default is 'rdkit'.

        Returns
        -------
        Crystal
            Modified version of this crystal with the molecules replaced.
        """
        other_mols = other.unique_mols()
        LOG.info(f"Found {len(other_mols)} unique molecules in other crystal")
        for i, mol in enumerate(other_mols):
            LOG.info(f"molecule-{i+1}({mol}): {len(mol.elements)} atoms")
        return self.replace_molecules(
            other_mols,
            how=how,
            reorder_atoms_to=reorder_atoms_to,
            reorder_mols=reorder_mols,
            method=method,
            **kwargs,
        )
    
    def replace_molecules_by_mols_of_multipole_file(
        self,
        multipole_file: str,
        how: Literal["best_match"] = "best_match",
        reorder_atoms_to: Literal["self", "other"] = "other",
        reorder_mols: bool = False,
        method: Literal["original", "rdkit", "shake_and_overlay_by_rdkit"] = "shake_and_overlay_by_rdkit",
        **kwargs,
    ) -> "Crystal":
        """Attempt to replace the molecules in the asymmetric unit of self with unique molecules
        in the multipole_file.

        WARNING: when using foreshorten_hydrogens in neighcrys, replacing molecules with
        those from multipole file may lead to fail or incorrect results.

        Parameters and Returns are the same as replace_molecules() except for the following:
        ----------
        multipole_file : str
            The path to the multipole file from which to take the unique molecules
        how : str, optional
            The method for determining which molecules to replace with those in the list. Default is 'best_match'
        reorder_atoms_to : str, optional
            Reorder the atoms of the overlap to match self or other (multipole_file). Default is 'other'
        reorder_mols : bool, optional
            Reorder molecules in crystal to match order of replacement molecules. Default is 'False'
        method : str, optional
            The method to use for overlaying the molecules ('original', 'rdkit', or 
            'shake_and_overlay_by_rdkit'). Default is 'shake_and_overlay_by_rdkit'

        Returns
        -------
        Crystal
            Modified version of this crystal with the molecules replaced.
        """
        mults = DistributedMultipoles.from_dma_file(multipole_file)

        # Find unique molecules in the multipole file
        other_mols = list(set(mults.molecules))
        if len(other_mols) == 0:
            raise ValueError(
                f"No unique molecules found in multipole file: {multipole_file}"
            )
        LOG.info(
            f"Found {len(other_mols)} unique molecules in multipole file ({multipole_file})"
        )
        for i, mol in enumerate(other_mols):
            LOG.info(f"molecule-{i+1}({mol}): {len(mol.elements)} atoms")
        return self.replace_molecules(
            other_mols,
            how=how,
            reorder_atoms_to=reorder_atoms_to,
            reorder_mols=reorder_mols,
            method=method,
            **kwargs,
        )

    def replace_molecules_by_substructure_overlay(self, molecules: Union[List[Molecule], Molecule], crys_substructure: Union[List[int], None] = None,
        mols_substructure: Union[List[int], None] = None, how: Literal["best_match"] ="best_match"
    ) -> tuple[list[object],list[float],"Crystal"]:
        """Attempt to replace the molecules in the asymmetric_unit with
        those provided by substructure overlay. 
        
        !! If multicomponent crystal will have to do each seperately !!

        Parameters
        ----------
        molecule : cspy.chem.Molecule
            The candidate molecule for replacing those in this crystal
        crys_substructure : list of int
            The atom indices of the crystal molecules to be overlayed with
        mols_substructure : list of int
            The atom indices of the molecule to overlay with.
        how : str, optional
            The method for determining which molecules to replace with
            those in the list

        Returns
        -------
        list[object]:'new_asymm_mols'
             List of molecule objects, each is one of the 'new' molecules but with orientation/position data that it would have in the modified crystal

        list[float]: `rmsds'
             rmsds of the best overlay for each replaced moleucle

        :obj:`Crystal`
            Modified version of this crystal with the molecules replaced.
        
        """
        if hasattr(molecules, "positions"):
            molecules = [molecules]
        if how != "best_match":
            raise NotImplementedError(
                "no other methods implemented for molecule replacement"
            )
        from copy import deepcopy

        if mols_substructure is None and crys_substructure is None:
            raise ValueError('must provide substructure to overlay')
        elif mols_substructure is None:
            mols_substructure = crys_substructure
        elif crys_substructure is None:
            crys_substructure = mols_substructure

        reference_mols = self.asym_mols()
        best_matches = [(-1, None, 1e100)] * len(reference_mols)
        for i, ref in enumerate(reference_mols):
            for j, mol in enumerate(molecules):
                try:
                    overlayed, rmsd = ref.overlay_substructure(
                        mol, crys_substructure, mols_substructure)
                except NotImplementedError:
                    continue
                if best_matches[i][2] > rmsd:
                    best_matches[i] = (j, overlayed, rmsd)
        LOG.info(
            "RMSDs of best matches: \n%s",
            "\n".join(
                f"old {i} -> new {x[0]} {x[2]:.3f}" for i, x in enumerate(best_matches)
            ),
        )
        replaced_mols = [ 0 if x[1] is None else 1 for x in best_matches ]

        rmsds = [ x[2] if x[2] is not None else reference_mols[i]
            for i, x in enumerate(best_matches) ]
          



        if sum(replaced_mols) == 0:
            LOG.info('no matching molecules to replace')
            return None

        LOG.info(
            'replacing %d molecules', sum(replaced_mols)
        )
        new_asym_mols = [
            x[1] if x[1] is not None else reference_mols[i] 
            for i, x in enumerate(best_matches) 
        ]
        asym_pos = np.vstack([x.positions for x in new_asym_mols])
        asym_nums = np.hstack([x.atomic_numbers for x in new_asym_mols])
        asymmetric_unit = AsymmetricUnit(
            [Element[x] for x in asym_nums], self.to_fractional(asym_pos)
        )
        return new_asym_mols, rmsds, Crystal(
            deepcopy(self.unit_cell), deepcopy(self.space_group), asymmetric_unit
        )



    def symmetry_unique_molecules(self, bond_tolerance=0.3) -> List["Molecule"]:
        """Calculate a list of connected molecules which contain
        every site in the asymmetric_unit

        Populates the _symmetry_unique_molecules member, subsequent
        calls to this function will be a no-op.
       
        Parameters
        ----------
        bond_tolerance : float, optional
            Bonding tolerance (bonded if d < cov_a + cov_b + bond_tolerance)

        Returns
        -------
        :obj:`list` of :obj:`Molecule`
            List of all connected molecules in the asymmetric_unit of this
            crystal, i.e. the minimum list of connected molecules which contain
            all sites in the asymmetric unit.
            If the asymmetric is molecular, the list will be of length
            num_molecules_in_asymmetric_unit and the total number of atoms
            will be equal to the number of atoms in the asymmetric_unit
        """
        if hasattr(self, "_symmetry_unique_molecules"):
            return getattr(self, "_symmetry_unique_molecules")
        
        uc_molecules = self.unit_cell_molecules()
        asym_atoms = np.zeros(len(self.asymmetric_unit), dtype=bool)
        molecules = []
        # sort by % of identity symop
        order = lambda x: len(np.where(x.asym_symops == 16484)[0]) / len(x)
        for i, mol in enumerate(sorted(uc_molecules, key=order, reverse=True)):
            asym_atoms_in_g = np.unique(mol.properties["asymmetric_unit_atoms"])
            if np.all(asym_atoms[asym_atoms_in_g]):
                continue
            asym_atoms[asym_atoms_in_g] = True
            molecules.append(mol)
            if np.all(asym_atoms):
                break
        LOG.debug("%d symmetry unique molecules", len(molecules))
        setattr(self, "_symmetry_unique_molecules", molecules)
        return molecules

    def molecular_shell(self, mol_idx=0, radius=3.8, method="nearest_atom"):
        mol = self.symmetry_unique_molecules()[mol_idx]
        frac_origin = self.to_fractional(mol.center_of_mass)
        frac_radius = radius / np.array(self.unit_cell.lengths)
        hmax, kmax, lmax = np.ceil(frac_radius + frac_origin).astype(int) + 1
        hmin, kmin, lmin = np.floor(frac_origin - frac_radius).astype(int) - 1
        uc_mols = self.unit_cell_molecules()
        shifts = self.to_cartesian(
            cartesian_product(
                np.arange(hmin, hmax), np.arange(kmin, kmax), np.arange(lmin, lmax)
            )
        )
        neighbours = []
        for uc_mol in uc_mols:
            for shift in shifts:
                uc_mol_t = uc_mol.translated(shift)
                dist = mol.distance_to(uc_mol_t, method=method)
                if (dist < radius) and (dist > 1e-2):
                    neighbours.append(uc_mol_t)
        return neighbours

    def slab(self, bounds=((-1, -1, -1), (1, 1, 1))):
        """Calculate the atoms and associated information
        for a slab consisting of multiple unit cells.
        
        If unit cell atoms have not been calculated, this populates
        their information."""
        uc_atoms = self.unit_cell_atoms()
        (hmin, kmin, lmin), (hmax, kmax, lmax) = bounds
        h = np.arange(hmin, hmax + 1)
        k = np.arange(kmin, kmax + 1)
        l = np.arange(lmin, lmax + 1)
        cells = cartesian_product(
            h[np.argsort(np.abs(h))], k[np.argsort(np.abs(k))], l[np.argsort(np.abs(l))]
        )
        ncells = len(cells)
        uc_pos = uc_atoms["frac_pos"]
        n_uc = len(uc_pos)
        pos = np.empty((ncells * n_uc, 3), dtype=np.float64)
        slab_cells = np.empty((ncells * n_uc, 3), dtype=np.float64)
        for i, cell in enumerate(cells):
            pos[i * n_uc: (i + 1) * n_uc, :] = uc_pos + cell
            slab_cells[i * n_uc: (i + 1) * n_uc] = cell
        slab_dict = {
            k: np.tile(v, ncells) for k, v in uc_atoms.items() if not k.endswith("pos")
        }
        slab_dict["frac_pos"] = pos
        slab_dict["cell"] = slab_cells
        slab_dict["n_uc"] = n_uc
        slab_dict["n_cells"] = ncells
        slab_dict["cart_pos"] = self.to_cartesian(pos)
        return slab_dict

    def is_slab(self, tolerance=1.0):
        """
        Simple routine to assess whether crystal structure is a slab, i.e. contains 
        vacuum gaps. Projects atom coordinates on to each axis and evaluates if there 
        are gaps larger than the maximum van der Waal contact + tolerance

        Parameters
        ----------
        tolerance : float
            tolerance for potential van der Waals contacts between atoms. Default of 
            1.0 Angstrom is perhaps conservative.
            
        Returns
        -------
        bool
            True if slab structure else False
        """
        structure = self.standardized().as_P1()
        
        vdw_radii = [
            Element.from_atomic_number(x).vdw 
            for x in structure.unit_cell_atoms()["element"]
        ]
        max_vdw_contact = 2*max(vdw_radii) + tolerance
        
        # prepend final coord in previous cell to include gaps across cell boundaries
        frac_pos = np.sort(structure.unit_cell_atoms()["frac_pos"], axis=0)
        frac_gaps = np.diff(frac_pos, axis=0, prepend=[frac_pos[-1,:]-1])
        max_gap = np.max(frac_gaps * structure.unit_cell.lengths)
        
        return True if max_gap > max_vdw_contact else False

    def atoms_in_radius(self, radius, origin=(0, 0, 0), farthest_atom=True):
        #mol = self.asym_mols()[0]
        #centroid = mol.centroid
        centroids = []
        for mol in self.asym_mols():
            centroids.append(mol.centroid)
        #print(centroids)
        centroid = np.average(centroids, axis=0)
        farthest_atom_xyz = np.max(
            np.linalg.norm(
                mol.positions - mol.centroid, axis=1
            )
        )
        frac_origin = self.to_fractional(centroid)
        #frac_origin = np.mean(self.asymmetric_unit.positions, axis=0)
        if farthest_atom:
            radius = radius + farthest_atom_xyz
        frac_radius = radius / np.array(self.unit_cell.high())
        hmax, kmax, lmax = np.ceil(frac_radius + frac_origin).astype(int) + 1
        hmin, kmin, lmin = np.floor(frac_origin - frac_radius).astype(int) - 1
        slab = self.slab(bounds=((hmin, kmin, lmin), (hmax, kmax, lmax)))
        #print(hmax, kmax, lmax, hmin, kmin, lmin, frac_origin)
        #tree = KDTree(slab["cart_pos"])
        #idxs = sorted(tree.query_ball_point(origin, radius))
        #result = {k: v[idxs] for k, v in slab.items() if isinstance(v, np.ndarray)}
        #result["uc_atom"] = np.tile(np.arange(slab["n_uc"]), slab["n_cells"])[idxs]
        #return result
        return slab

    @property
    def site_labels(self):
        """array of labels for sites in the asymmetric_unit"""
        return self.asymmetric_unit.labels

    def __repr__(self):
        if "lattice_energy" in self.properties and "density" in self.properties:
            return "<Crystal {} {} ({:.3f}, {:.3f})>".format(
                self.asymmetric_unit.formula,
                self.space_group.symbol,
                self.properties["density"],
                self.properties["lattice_energy"],
            )
        return "<Crystal {} {}>".format(
            self.asymmetric_unit.formula, self.space_group.symbol
        )

    def standardized(self):
        """Calculate a standardized versio of this crystal using
        the Niggli cell"""
        from cspy.crystal.util import standardize_crystal

        return standardize_crystal(self)

    def as_primitive_P1(self):
        """calculate smallest primitive P1 cell"""
        from cspy.crystal.util import niggli_primitive_crystal
        
        new_crystal = niggli_primitive_crystal(self)
        
        if new_crystal:
            return new_crystal
        else:
            LOG.warning("failed primitive conversion, returning P1")
            return self.as_P1()

    @property
    def density(self):
        if "density" in self.properties:
            return self.properties["density"]
        uc_mass = sum(Element[x].mass for x in self.unit_cell_atoms()["element"])
        uc_vol = self.unit_cell.volume()
        return uc_mass / uc_vol / 0.6022

    @property
    def energy(self):
        return self.properties.get("lattice_energy", np.nan)

    @classmethod
    def load(cls, filename, **kwargs):
        """Load a crystal structure from file (.res, .cif)"""
        extension_map = {
            ".res": cls.from_shelx_file,
            ".cif": cls.from_cif_file,
            ".xyz": cls.from_xyz_file,
            ".exyz": cls.from_extended_xyz_file,
        }
        extension = os.path.splitext(filename)[-1].lower()
        return extension_map[extension](filename, **kwargs)

    def save(self, filename):
        """Save this crystal structure to file (.res)"""
        extension_map = {".res": self.to_shelx_file, ".cif": self.to_cif_file}
        extension = os.path.splitext(filename)[-1].lower()
        return extension_map[extension](filename)

    def to_shelx_file(self, filename):
        """Write this crystal structure as a shelx .res file"""
        with open(filename, "w") as f:
            f.write(self.to_shelx_string())

    def to_shelx_string(self, titl=None):
        """Represent this crystal structure as a shelx .res string"""
        fn = lambda x: x.replace("H", "Z")
        sfac = sorted([x.symbol for x in set(self.asymmetric_unit.elements)], key=fn)
        atom_sfac = [sfac.index(x.symbol) + 1 for x in self.asymmetric_unit.elements]
        shelx_data = {
            "TITL": self.titl if titl is None else titl,
            "CELL": list(self.unit_cell.parameters),
            "SFAC": sfac,
            "SYMM": [
                str(s)
                for s in self.space_group.reduced_symmetry_operations()
                if not s.is_identity()
            ],
            "LATT": self.space_group.latt,
            "ATOM": [
                "{:3} {:3} {: 20.12f} {: 20.12f} {: 20.12f}".format(l, s, *pos)
                for l, s, pos in zip(
                    self.asymmetric_unit.labels, atom_sfac, self.site_positions
                )
            ],
        }
        return to_res_contents(shelx_data)

    @classmethod
    def from_shelx_file(cls, filename, **kwargs):
        """Initialize a crystal structure from a shelx .res file"""
        p = Path(filename)
        titl = p.stem
        return cls.from_shelx_string(p.read_text(), titl=titl, **kwargs)

    @classmethod
    def from_shelx_string(cls, file_content, **kwargs):
        """Initialize a crystal structure from a shelx .res string"""
        shelx_dict = parse_shelx_file_content(file_content)
        asymmetric_unit = AsymmetricUnit.from_records(shelx_dict["ATOM"])
        space_group = SpaceGroup.from_symmetry_operations(
            shelx_dict["SYMM"], expand_latt=shelx_dict["LATT"]
        )
        unit_cell = UnitCell.from_lengths_and_angles(
            shelx_dict["CELL"]["lengths"], shelx_dict["CELL"]["angles"], unit="degrees"
        )
        return cls(
            unit_cell,
            space_group,
            asymmetric_unit,
            titl=kwargs.pop("titl", shelx_dict["TITL"]),
            **kwargs,
        )

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
        lattice = []
        for line in data["lattice bohr"]:
            lattice.append([float(x) for x in line.split()])
        elements = []
        positions = []
        for line in data["coord"]:
            x, y, z, el = line.split()
            positions.append((float(x), float(y), float(z)))
            elements.append(Element[el])
        direct = np.array(lattice) * 0.529177249
        pos_cart = np.array(positions) * 0.52917749
        uc = UnitCell(direct)
        pos_frac = uc.to_fractional(pos_cart)
        asym = AsymmetricUnit(elements, pos_frac)
        return cls(uc, SpaceGroup(1), asym)

    def to_turbomole_string(self):
        tmol_template = (
            "$periodic 3\n"
            "$cell\n"
            "{a} {b} {c} {alpha} {beta} {gamma}\n"
            "$coord frac\n"
            "{coords}\n"
        )
        uc_atoms = self.unit_cell_atoms()
        pos = uc_atoms["frac_pos"]
        el = [Element[x].symbol for x in uc_atoms["element"]]
        coords = "\n".join(
            f"    {x:10.6f} {y:10.6f} {z:10.6f} {e}" for (x, y, z), e in zip(pos, el)
        )
        bohr = 1 / 0.529177249
        a, b, c, alpha, beta, gamma = self.unit_cell.parameters
        return tmol_template.format(
            a=a * bohr,
            b=b * bohr,
            c=c * bohr,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            coords=coords,
        )

    def to_gen_file(self, filename, **kwargs):
        gen_template = (
            "{num_atoms} S\n"
            "{elements}\n"
            "{coords}\n"
            "0.000000 0.000000 0.000000\n"
            "{cell_matrix}\n"
        )
        try:
            std_crys = self.standardized()
            P1 = std_crys.as_P1()
        except Exception as e:
            LOG.debug(f'{self.titl} - Niggli reduction failed. Using original cell parameters')
            P1 = self.as_P1()
        uc_atoms = P1.unit_cell_atoms()
        pos = uc_atoms["cart_pos"]
        num_atoms = len(uc_atoms["cart_pos"])
        ele = [Element[x].symbol for x in uc_atoms["element"]]
        unique_ele = set([Element[x].symbol for x in sorted(uc_atoms["element"])])
        ele_order_dict = {item:i+1 for i, item in enumerate(unique_ele)}
        temp_coords = np.column_stack(([ele_order_dict[item] for item in ele], pos))
        sorted_temp_coords = temp_coords[temp_coords[:,0].argsort()]
        index = np.linspace(1, num_atoms, num_atoms)
        genfile_elements =" ".join(item for item in unique_ele)
        genfile_coords ="\n".join("{:.10g} {:.10g} {:.10f} {:.10f} {:.10f}".format(*item)
                                  for item in np.column_stack((index, sorted_temp_coords)))
        genfile_cell ="\n".join("{:.10f} {:.10f} {:.10f}".format(*item)
                                for item in P1.unit_cell.lattice)

        input_contents = gen_template.format(
            num_atoms=num_atoms,
            elements=genfile_elements,
            coords=genfile_coords,
            cell_matrix=genfile_cell,
            )
        with open(filename, "w") as f:
            f.write(input_contents)

    def to_cif_data(self, data_block_name=None,**kwargs):
        if data_block_name is None:
            data_block_name = self.titl
        if "cif_data" in self.properties:
            cif_data = self.properties["cif_data"]
        else:
            cif_data = {
                "audit_creation_method": "generated by cspy2.0",
                "symmetry_equiv_pos_site_id": list(
                    range(1, len(self.symmetry_operations) + 1)
                ),
                "symmetry_equiv_pos_as_xyz": [str(x) for x in self.symmetry_operations],
                "cell_length_a": self.unit_cell.a,
                "cell_length_b": self.unit_cell.b,
                "cell_length_c": self.unit_cell.c,
                "cell_angle_alpha": self.unit_cell.alpha_deg,
                "cell_angle_beta": self.unit_cell.beta_deg,
                "cell_angle_gamma": self.unit_cell.gamma_deg,
                "atom_site_label": self.asymmetric_unit.labels,
                "atom_site_type_symbol": [
                    x.symbol for x in self.asymmetric_unit.elements
                ],
                "atom_site_fract_x": self.asymmetric_unit.positions[:, 0],
                "atom_site_fract_y": self.asymmetric_unit.positions[:, 1],
                "atom_site_fract_z": self.asymmetric_unit.positions[:, 2],
                "atom_site_occupancy": self.asymmetric_unit.properties.get(
                    "occupation", np.ones(len(self.asymmetric_unit))
                ),
            }
        if "wavelength" in kwargs:
            cif_data["diffrn_radiation_wavelength"] = float(kwargs["wavelength"])
        return {data_block_name: cif_data}

    def to_cif_file(self, filename, **kwargs):
        cif_data = self.to_cif_data(**kwargs)
        return Cif(cif_data).to_file(filename)

    def to_cif_string(self, **kwargs):
        cif_data = self.to_cif_data(**kwargs)
        return Cif(cif_data).to_string()

    @classmethod
    def from_cif_data(cls, cif_data, titl=None):
        """Initialize a crystal structure from a dictionary
        of CIF data"""
        labels = cif_data.get("atom_site_label", None)
        symbols = cif_data.get("atom_site_type_symbol", labels)
        elements = [Element[x] for x in symbols]
        x = np.asarray(cif_data.get("atom_site_fract_x", []))
        y = np.asarray(cif_data.get("atom_site_fract_y", []))
        z = np.asarray(cif_data.get("atom_site_fract_z", []))
        occupation = np.asarray(cif_data.get("atom_site_occupancy", [1] * len(x)))
        frac_pos = np.array([x, y, z], dtype=np.float64).T
        asym = AsymmetricUnit(
            elements=elements, positions=frac_pos, labels=labels, occupation=occupation
        )
        lengths = [cif_data[f"cell_length_{x}"] for x in ("a", "b", "c")]
        angles = [cif_data[f"cell_angle_{x}"] for x in ("alpha", "beta", "gamma")]
        unit_cell = UnitCell.from_lengths_and_angles(lengths, angles, unit="degrees")
        if "symmetry_equiv_pos_as_xyz" in cif_data:
            space_group = SpaceGroup.from_symmetry_operations(
                [
                    SymmetryOperation.from_string_code(x)
                    for x in cif_data["symmetry_equiv_pos_as_xyz"]
                ]
            )
        elif "symmetry_Int_Tables_number" in cif_data:
            space_group = SpaceGroup(cif_data["symmetry_Int_Tables_number"])

        return Crystal(unit_cell, space_group, asym, cif_data=cif_data, titl=titl)

    @classmethod
    def from_cif_file(cls, filename, data_block_name=None):
        """Initialize a crystal structure from a CIF file"""
        cif = Cif.from_file(filename)
        if data_block_name is not None:
            return cls.from_cif_data(cif.data[data_block_name])

        crystals = {
            name: cls.from_cif_data(data, titl=name) for name, data in cif.data.items()
        }
        keys = list(crystals.keys())
        if len(keys) == 1:
            return crystals[keys[0]]
        return crystals

    @classmethod
    def from_cif_string(cls, file_content, **kwargs):
        cif = Cif.from_string(file_content)
        crystals = {
            name: cls.from_cif_data(data, titl=name) for name, data in cif.data.items()
        }
        keys = list(crystals.keys())
        if len(keys) == 1:
            return crystals[keys[0]]
        return crystals

    @classmethod
    def from_CONTCAR(cls, filename, **kwargs):
        with open(filename, 'r') as f:
            lines = [line.strip('\n') for line in f]

        return cls.from_CONTCAR_lines(lines, **kwargs)
    
    @classmethod
    def from_CONTCAR_string(cls, contents, **kwargs):
        lines = contents.strip('\n').split('\n')
        return cls.from_CONTCAR_lines(lines, **kwargs)

    @classmethod
    def from_CONTCAR_lines(cls, lines, **kwargs):
        scaling_factor = float(lines[1])
        unscaled_vectors = np.vstack([np.array([vector.split() for vector in lines[2:5]])]).astype(float)
        scaled_vectors = scaling_factor * unscaled_vectors
        elements_dict = {elem:lines[6].split()[idx] for idx, elem in enumerate(lines[5].split())}
        elements = []
        num_atoms = 0
        for key, value in elements_dict.items():
            elements += int(value) * [Element[key]]
            num_atoms += int(value)

        for i, line in enumerate(lines):
            if 'Cartesian' in line or 'Direct' in line:
                n_start = i+1
                break
        positions = np.vstack([np.array([pos.split()[:3] for pos in lines[n_start:num_atoms+n_start]])]).astype(float) # assumes Direct keyword

        unit_cell = UnitCell(scaled_vectors)
        asym = AsymmetricUnit(elements=elements,
                              positions=positions,
                              )

        return cls(unit_cell, SpaceGroup(1), asym, **kwargs)

    @classmethod
    def from_gen_file(cls, filename, **kwargs):
        """Initialize a crystal structure from a GEN file"""
        p = Path(filename)
        return cls.from_gen_string(p.read_text(), **kwargs)
        
    @classmethod
    def from_gen_string(cls, file_content, **kwargs):
        from cspy.formats.gen import parse_gen_file_content
        gen_dict = parse_gen_file_content(file_content)
        unit_cell = UnitCell(gen_dict['cell_vectors'])
        space_group = gen_dict['space_group']
        asym = AsymmetricUnit(elements=gen_dict['elements'],
                              positions=unit_cell.to_fractional(gen_dict['positions']))
        
        return cls(unit_cell, space_group, asym, **kwargs)

    @classmethod
    def from_molecules(cls, molecules, **kwargs):
        sort = kwargs.get("sort_atoms", False)
        if not isinstance(molecules, list):
            molecules = list(molecules)
        assert (len(molecules)) > 0, "Require at least 1 molecule"
        if sort:
            Molecule.group_atoms_by_element(molecules)
        boxa, boxb, boxc = 0, 0, 0
        for i, mol in enumerate(molecules):
            m_bbox = mol.bbox_size + 4.0
            c = -mol.centroid
            if i > 0:
                c[0] += boxa
            mol.translate(c)
            boxa += m_bbox[0] + 3.0
            boxb += m_bbox[1]
            boxc += m_bbox[2]
            boxa, boxb, boxc = max(5.0, boxa), max(5.0, boxb), max(5.0, boxc)

        unit_cell = UnitCell.orthorhombic(boxa, boxb, boxc)
        elements = np.hstack([x.elements for x in molecules])
        positions = np.vstack([x.positions for x in molecules])
        asym = AsymmetricUnit(elements, unit_cell.to_fractional(positions))
        return cls(unit_cell, SpaceGroup(1), asym, **kwargs)
        
    @classmethod
    def from_CONTCAR(cls, filename, **kwargs):
        with open(filename, 'r') as f:
            lines = [line.strip('\n') for line in f]

        scaling_factor = float(lines[1])
        unscaled_vectors = np.vstack([np.array([vector.split() for vector in lines[2:5]])]).astype(float)
        scaled_vectors = scaling_factor * unscaled_vectors
        elements_dict = {elem:lines[6].split()[idx] for idx, elem in enumerate(lines[5].split())}
        elements = []
        num_atoms = 0
        for key, value in elements_dict.items():
            elements += int(value) * [Element[key]]
            num_atoms += int(value)

        positions = np.vstack([np.array([pos.split() for pos in lines[8:num_atoms+8]])]).astype(float) # assumes Direct keyword

        unit_cell = UnitCell(scaled_vectors)
        asym = AsymmetricUnit(elements=elements,
                              positions=positions,
                              )

        return cls(unit_cell, SpaceGroup(1), asym, **kwargs)

    @classmethod
    def from_unit_cell_sg_molecules(cls, unit_cell, space_group, molecules, **kwargs):
        elements = []
        positions = []
        labels = []
        for mol in molecules:
            elements += mol.elements
            positions += list(mol.positions)
            #labels += list(mol.labels)
            labels += list(mol.properties["asymmetric_unit_labels"])
        positions = np.asarray(positions)
        asymmetric_unit = AsymmetricUnit(
            elements, unit_cell.to_fractional(positions),
        )
        return cls(unit_cell, space_group, asymmetric_unit)

    @classmethod
    def from_xyz_file(cls, filename, **kwargs):
        molecules = [Molecule.from_xyz_file(filename)]
        return cls.from_molecules(molecules, **kwargs)

    @classmethod
    def from_xyz_files(cls, filenames, **kwargs):
        molecules = [Molecule.from_xyz_file(filename) for filename in filenames]

      #R  print("molecules = ", molecules)
 
        return cls.from_molecules(molecules, **kwargs)

    @classmethod
    def from_extended_xyz_file(cls, filename: str, **kwargs) -> "Crystal":
        """Initialize a crystal structure from an extended XYZ file."""
        xyz_dict = parse_xyz_file(filename, extended_xyz=True)
        lattice = [float(x) for x in xyz_dict["Lattice"].split()]
        if len(lattice) == 9:
            latt = np.asarray(lattice).reshape((3, 3))
            unit_cell = UnitCell(latt)
        elif len(lattice) == 6:
            unit_cell = UnitCell.from_lengths_and_angles(lattice)
        else:
            raise ValueError("Lattice must have 6 or 9 elements.")

        elements = []
        positions = []
        for sym, pos in xyz_dict["atoms"]:
            elements.append(Element[sym])
            positions.append(pos)
        positions = np.asarray(positions)
        asymmetric_unit = AsymmetricUnit(
            elements, unit_cell.to_fractional(positions)
        )
        space_group = SpaceGroup(xyz_dict.get("space_group", 1))
        titl = xyz_dict.get("phase", None)
        return cls(unit_cell, space_group, asymmetric_unit, titl=titl)

    @classmethod
    def from_pmin_save(cls, pmin_save, space_group=SpaceGroup(1)):
        a, b, c, ca, cb, cg = pmin_save.params
        unit_cell = UnitCell.from_lengths_and_angles(
            np.array([a, b, c]), np.arccos([ca, cb, cg])
        )
        labels = []
        idxs = []
        frac = []

        for label, idx, frac_coord in pmin_save.atoms:
            labels.append(label)
            idxs.append(idx)
            frac.append(frac_coord)
        asym = AsymmetricUnit(
            elements=[Element[x] for x in labels],
            positions=np.array(frac),
            labelidx=np.array(idxs),
        )
        return cls(unit_cell, space_group, asym)

    @classmethod
    def from_pmin_save_string(cls, pmin_save_contents, space_group=SpaceGroup(1)):
        return cls.from_pmin_save(PminSave(pmin_save_contents), space_group=space_group)

    def neighcrys_setup(
        self, potential_type="F", file_content=None, bond_tolerance=0.4
    ):
        """Convenience method to add neighcrys axis information for this crystal"""
        from .neighcrys_axis import NeighcrysAxis

        self.neighcrys_axis = NeighcrysAxis(
            self, potential_type, file_content=file_content, alter_crystal=True
        )

        labels = []
        for m in self.symmetry_unique_molecules():
            labels += [x[:4] for x in m.properties["neighcrys_label"]]
        self.properties["unique_potential_labels"] = set(labels)
        elements = set(self.asymmetric_unit.elements)
        if "bondlength_cutoffs" not in self.properties:
            bondlength_cutoffs = {}
            for m in self.symmetry_unique_molecules():
                for k, v in m.neighcrys_bond_cutoffs().items():
                    if k not in bondlength_cutoffs or v > bondlength_cutoffs[k]:
                        bondlength_cutoffs[k] = v
            self.properties["bondlength_cutoffs"] = bondlength_cutoffs
        self.properties["neighcrys_potential_type"] = potential_type


    def map_multipoles(
        self,
        mults: DistributedMultipoles,
        foreshorten_hydrogens: bool = False,
        reorder_atoms_if_high_rmsd: bool = True,
        reorder_method: Literal["rdkit", "molecular_axis"] = "rdkit",
    ) -> List[Tuple[int, float, bool]]:
        """Attempt to map the multipoles in mults to the symmetry unique molecules
        in this crystal.

        Parameters
        ----------
        mults : DistributedMultipoles
            The multipoles to match the atom ordering to. Molecules are read from the multipoles object.
        foreshorten_hydrogens : bool, optional
            Whether to foreshorten hydrogens in the molecular axis frame.
        reorder_atoms_if_high_rmsd : bool, optional
            If True, try to reorder the atoms to match the multipoles if the mapping returns a high RMSD.
        reorder_method : str, optional
            The method to use for matching the atom ordering ('rdkit' or 'molecular_axis'). Default: 'rdkit'.

        Returns
        -------
        list of tuple
            List of (best_idx, best_rmsd, inv) for each molecule.
        """
        if not hasattr(self, "neighcrys_axis"):
            self.neighcrys_setup()

        LOG.debug("Attempting to map multipoles in %s", self)
        mapping = []
        for mol in self.symmetry_unique_molecules():
            pos = mol.positions_in_molecular_axis_frame(
                foreshorten_hydrogens=foreshorten_hydrogens
            )
            diff = mol.positions - pos
            rmsd_rotated = np.sqrt(np.vdot(diff, diff) / diff.shape[0])
            LOG.debug(
                "RMSD change from rotation to molecular_axis_frame: %.3f", rmsd_rotated
            )
            pos_inv = pos * np.array([1, 1, -1])
            best_idx = -1
            best_rmsd = 1e100
            inv = False
            for i, mult_mol in enumerate(mults.molecules):
                if len(pos) != len(mult_mol.positions):
                    continue
                diff = pos - mult_mol.positions
                rmsd = np.sqrt(np.vdot(diff, diff) / diff.shape[0])
                LOG.debug("mol %d no inversion: %.5g", i, rmsd)
                if rmsd < best_rmsd:
                    best_idx = i
                    best_rmsd = rmsd
                    inv = False
                diff_inv = pos_inv - mult_mol.positions
                rmsd_inv = np.sqrt(np.vdot(diff_inv, diff_inv) / diff_inv.shape[0])
                LOG.debug("mol %d with inversion: %.5g", i, rmsd_inv)
                if rmsd_inv < best_rmsd:
                    best_idx = i
                    best_rmsd = rmsd_inv
                    inv = True
            if best_rmsd > 1e-3:
                LOG.warning(
                    "High best RMSD in Crystal.%s mol = %s, match=%d%s RMSD= %.5g",
                    self.map_multipoles.__name__,
                    mol,
                    best_idx,
                    "i" if inv else "",
                    best_rmsd,
                )
                if reorder_atoms_if_high_rmsd:
                    LOG.info("Attempting to reorder atoms to fix high RMSD...")
                    self.match_atom_ordering_to_multipoles(
                        mults, reorder_method=reorder_method
                    )
                    return self.map_multipoles(
                        mults=mults,
                        foreshorten_hydrogens=foreshorten_hydrogens,
                        reorder_atoms_if_high_rmsd=False,
                        reorder_method=reorder_method,
                    )
            LOG.debug(
                "Best matching molecule %d%s in mults, RMSD=%.5g",
                best_idx,
                "i" if inv else "",
                best_rmsd,
            )
            if inv:
                mol.properties["multipoles"] = [
                    x.reflected("xy") for x in mults.molecules[best_idx].multipoles
                ]
            else:
                mol.properties["multipoles"] = [
                    x for x in mults.molecules[best_idx].multipoles
                ]
            mapping.append((best_idx, best_rmsd, inv))
        return mapping

    def assign_ff_atom_typing(self, typing_file, get_GULP_labels=False):
        # WIP function. Only works for single-species crystals
        atom_typing = []
        charges = []

        with open(typing_file, 'r') as f:
                for line in f.readlines():
                    label, x, y, z, type, c = line.split()
                    atom_typing.append(type)
                    charges.append(Multipole(moments=[c], rank=0))

        if get_GULP_labels:
            element_max = dict()
            typing_GULP_label = dict()

        for mol in self.symmetry_unique_molecules():
            mol.properties["potential_atom_types"] = atom_typing
            mol.properties["multipoles"] = charges
            LOG.debug("Loading atom typing")
            LOG.debug(atom_typing)
            LOG.debug("Loading charges")
            LOG.debug(charges)

            if get_GULP_labels:
                GULP_labels = []
                for ind, element in enumerate(mol.elements):
                    element = str(element)
                    atom_type = atom_typing[ind]
                    if not element in element_max.keys():
                        element_max[element] = 0
                    if not atom_type in typing_GULP_label.keys():
                        element_max[element] += 1
                        GULP_label = element + str(element_max[element])
                        typing_GULP_label[atom_type] = GULP_label
                    GULP_labels.append(typing_GULP_label[atom_type])
                mol.properties["GULP_atom_labels"] = GULP_labels
            LOG.debug("Assigning GULP atom labels: ")
            LOG.debug(GULP_labels)
    
    def match_atom_ordering_to(
        self,
        target_mols: List[Molecule],
        reorder_method: Literal["rdkit", "molecular_axis"] = "rdkit",
    ) -> "Crystal":
        """
        Reorder the atoms in this crystal to match the order of the listed molecules.

        Tries to find molecules in the crystal's asymmetric unit that are the same as the target molecules.
        Attempts to reorder the atoms in the asymmetric unit molecules to match the target molecules.
        If the atom ordering does not match, it will try to reorder the atoms in the asymmetric unit molecules.

        Parameters
        ----------
        target_mols : list of Molecule
            The molecules to match the atom ordering to.
        reorder_method : str, optional
            The method to use for matching the atom ordering ('rdkit' or 'molecular_axis'). Default: 'rdkit'

        Returns
        -------
        cspy.crystal.Crystal
            A new crystal with the atoms reordered to match the target molecules.
        """
        source_mols = self.asym_mols()
        updated_mols = []
        for smol in source_mols:
            matched = False
            for tmol in target_mols:
                try:
                    smol.match_atom_ordering_to(
                        tmol, method=reorder_method
                    )
                    matched = True
                    break
                except ValueError as e:
                    LOG.debug("Failed to match atoms: %s", e)
            if not matched:
                LOG.warning(
                    "No matching atom ordering found for molecule: %s", smol
                )
            updated_mols.append(smol)

        if len(updated_mols) != len(source_mols):
            raise ValueError("Failed to match atoms in all molecules")

        positions = []
        atomic_nums = []
        for uc_mol in updated_mols:
            positions.append(uc_mol.positions)
            atomic_nums.append(uc_mol.atomic_numbers)

        asym_pos = np.vstack(positions)
        asym_nums = np.hstack(atomic_nums)
        new_asymmetric_unit = AsymmetricUnit(
            [Element[x] for x in asym_nums],
            self.unit_cell.to_fractional(asym_pos),
        )
        new_crystal = Crystal(
            self.unit_cell, self.space_group, new_asymmetric_unit
        )
        return new_crystal

    def match_atom_ordering_to_multipoles(
        self,
        mults: DistributedMultipoles,
        reorder_method: Literal["rdkit", "molecular_axis"] = "rdkit",
    ):
        """Reorders the atoms in this crystal to match the order of the atoms in the multipoles file.

        Parameters
        ----------
        mults : DistributedMultipoles
            The multipoles to match the atom ordering to. Molecules are read from the multipoles object.
        reorder_method : str, optional
            The method to use for matching the atom ordering. Default is 'rdkit'.

        Returns
        -------
        cspy.crystal.Crystal
            A new crystal with the atoms reordered to match the multipoles.
        """
        target_mols = mults.molecules
        return self.match_atom_ordering_to(target_mols, reorder_method=reorder_method)

    def match_atom_ordering_to_crystal(
        self, target_crystal: "Crystal", reorder_method: Literal["rdkit", "molecular_axis"] = "rdkit"
    ) -> "Crystal":
        """Reorders the atoms in this crystal to match the order of the atoms in the target crystal.

        Parameters
        ----------
        target_crystal : Crystal
            The crystal to match the atom ordering to.
        reorder_method : str, optional
            The method to use for matching the atom ordering. Default is 'rdkit'.

        Returns
        -------
        Crystal
            A new crystal with the atoms reordered to match the target crystal.
        """
        target_mols = target_crystal.asym_mols()
        return self.match_atom_ordering_to(target_mols, reorder_method=reorder_method)

    def convert_to_molecular_asymmetric_unit(self):
        LOG.warning(
            "Space group subgroups are not properly "
            "implemented, this will not work in many cases, so "
            "check the results carefully."
        )
        mols = self.asym_mols()
        n_uc_mols = len(self.unit_cell_molecules())
        elements = []
        positions = []
        for mol in mols:
            elements += mol.elements
            positions += self.to_fractional(mol.positions).tolist()

        asym = AsymmetricUnit(elements, positions)

        if len(asym) == len(self.asymmetric_unit):
            LOG.warning(
                "Crystal is already has a molecular asymmetric unit:" " returning self"
            )
            return self

        generator_symops = np.unique(
            np.hstack([x.properties["generator_symop"] for x in mols])
        )
        generator_symops = [
            SymmetryOperation.from_integer_code(x)
            for x in generator_symops
            if x != 16484
        ]
        subgroups = self.space_group.subgroups()
        crystal = None
        for subgroup in sorted(
            subgroups, key=lambda x: x.international_tables_number, reverse=True
        ):
            sub_symops = subgroup.symmetry_operations
            if any(symop in sub_symops for symop in generator_symops):
                continue
            else:
                new_space_group = subgroup
                crystal = Crystal(self.unit_cell, new_space_group, asym)
                if len(crystal.unit_cell_molecules()) == n_uc_mols:
                    break

        return crystal

    def choose_trigonal_lattice(self, choice="H"):
        """Change the choice of lattice for this crystal to either
        rhombohedral or hexagonal cell

        Parameters
        ----------
        choice: str, optional
            The choice of the resulting lattice, either 'H' for hexagonal
            or 'R' for rhombohedral (default 'H').
        """
        if not self.space_group.has_hexagonal_rhombohedral_choices():
            raise ValueError("Invalid space group for choose_trigonal_lattice")
        if self.space_group.choice == choice:
            return
        cart_asym_pos = self.to_cartesian(self.asymmetric_unit.positions)
        assert choice in ("H", "R"), "Valid choices are H, R"
        if self.space_group.choice == "R":
            T = np.array(((-1, 1, 0), (1, 0, -1), (1, 1, 1)))
        else:
            T = 1 / 3 * np.array(((-1, 1, 1), (2, 1, 1), (-1, -2, 1)))
        new_uc = UnitCell(np.dot(T, self.unit_cell.direct))
        self.unit_cell = new_uc
        self.asymmetric_unit.positions = self.to_fractional(cart_asym_pos)
        self.space_group = SpaceGroup(
            self.space_group.international_tables_number, choice=choice
        )

    def as_P1(self):
        """Create a copy of this crystal in space group P 1, with the new
        asymmetric_unit consisting of self.unit_cell_molecules()"""
        return self.as_P1_supercell((1, 1, 1))

    def as_P1_supercell(self, size):
        """Create a supercell of this crystal in space group P 1.

        Parameters
        ----------
        size: tuple of int
            Size of the P 1 supercell to be created.

        Returns
        -------
        Crystal
            Crystal object of a supercell in space group P 1.
        """
        import itertools as it

        umax, vmax, wmax = size
        a, b, c = self.unit_cell.lengths
        sc = UnitCell.from_lengths_and_angles(
            (umax * a, vmax * b, wmax * c), self.unit_cell.angles
        )
        u = np.arange(umax)
        v = np.arange(vmax)
        w = np.arange(wmax)

        trans = np.array(list(it.product(u, v, w))) @ self.unit_cell.lattice
        positions = []
        atomic_nums = []
        # First u * v * w mols are sc equivalents of first mol in uc
        # [u * v * w: 2 * u * v * w] mols are sc equivalents of 2nd mol in uc
        for uc_mol in self.unit_cell_molecules():
            for tran in trans:
                positions.append(uc_mol.positions + tran)
                atomic_nums.append(uc_mol.atomic_numbers)

        asym_pos = np.vstack(positions)
        asym_nums = np.hstack(atomic_nums)
        asymmetric_unit = AsymmetricUnit(
            [Element[x] for x in asym_nums], sc.to_fractional(asym_pos)
        )
        new_crystal = Crystal(sc, SpaceGroup(1), asymmetric_unit)
        new_crystal.titl = self.titl + "-P1-{}-{}-{}".format(*size)
        return new_crystal
    

    def expand_cell_isotropically(self, scale_factor : float):
        """Increase the volume of the cell by a scale factor"""
        old_lengths = np.asarray(self.unit_cell.lengths)
        new_lengths = old_lengths * scale_factor
        mol_positions = []
        old_mol_centroids_cart = []
        mol_centroids_frac = []
        new_poss = []
        for mol in self.unit_cell_molecules():
             positions = mol.positions
             centroid = mol.centroid
             mol_positions.append(positions)
             old_mol_centroids_cart.append(centroid)
             mol_centroids_frac.append(self.unit_cell.to_fractional(centroid))

        self.unit_cell.set_lengths_and_angles(new_lengths, self.unit_cell.angles)
        for mol_ind, centroid in enumerate(mol_centroids_frac):
            new_mol_centroid_cart = self.unit_cell.to_cartesian(centroid)
            translation = new_mol_centroid_cart - old_mol_centroids_cart[mol_ind]
            new_pos = mol_positions[mol_ind] + translation
            new_poss.append(new_pos)
            self.unit_cell_molecules()[mol_ind].positions = new_pos

        new_poss = np.concatenate(new_poss)

        self.asymmetric_unit.positions = self.unit_cell.to_fractional(new_poss)


    def calculate_lattice_energy(self, **kwargs):
        """Calculate multipoles, then calculate a single point energy
        for this structure

        Returns
        _______
        this crystal, with the energy set by the minimizer
        """
        kwargs["single_point"] = True
        from cspy.minimize import calculate_multipoles_and_minimize

        return calculate_multipoles_and_minimize(self, **kwargs)

    def minimize(self, **kwargs):
        """Calculate multipoles, then optimize this structure
        
        Returns a minimized version of this crystal.
        """
        from cspy.minimize import calculate_multipoles_and_minimize

        return calculate_multipoles_and_minimize(self, **kwargs)

    def minimize_with_dftb(self, **kwargs):
        """Optimizes the structure using dftb

        Returns a minimized version of this crystal
        """
        from cspy.minimize import dftb_calculator
        return dftb_calculator(self, **kwargs)

    def minimize_with_vasp(self, **kwargs):
        """Optimizes the structure using vasp

        Returns a minimized version of this crystal
        """
        from cspy.minimize import vasp_calculator
        return vasp_calculator(self, **kwargs)

    def minimize_with_mace(
        self,
        minimizer: Union["ASEMinimizer", None] = None,
        **kwargs
    ) -> "Crystal":
        """
        Geometry optimize the structure using a MACE pre-trained model.

        Parameters
        ----------
        minimizer : ASEMinimizer, optional
            An instance of ASEMinimizer to use for optimization. If None, a default ASEMinimizer with mace_mp will be created.

        **kwargs : dict
            Keyword arguments passed to ASEMinimizer. See ASEMinimizer documentation for details.

        Returns
        -------
        Crystal
            A minimized version of this crystal.
        """
        if minimizer is None:
            from cspy.minimize import ASEMinimizer
            minimizer = ASEMinimizer(**kwargs)
        return minimizer.minimize(self, **kwargs)

    def minimize_with_symmetry_reduction(self, size, **kwargs):
        """Reduce to P1 supercell, calculate multipoles, then optimize
        the structure

        Returns a minimized version of this crystal.
        """
        kwargs["symmetry_reduction"] = True
        from cspy.minimize import calculate_multipoles_and_minimize

        crystal = self.as_P1_supercell(size)
        return calculate_multipoles_and_minimize(crystal, **kwargs)

    def minimize_and_calculate_phonons(self, **kwargs):
        """Calculate multipoles, then optimize this structure with the
        phonons calculations turned on, it is recommended that the
        crystal structure is optimize with dmacrys with
        minimize_with_symmetry_reduction before using this function

        Returns a minimized version of this crystal, with phonon
        properties in the crystal properties dictionary
        """
        kwargs["phonon"] = True
        from cspy.minimize import calculate_multipoles_and_minimize

        if self.space_group.symbol != "P1":
            raise NotImplementedError(
                "Phonon code in cspy only tested for crystals in P1, "
                "it is therefore recommended that you minimize with "
                "minimize_with_symmetry_reduction before using "
                "this function"
            )

        return calculate_multipoles_and_minimize(self, **kwargs)

    def create_jmol_phonon_animation_file(self, filename, **kwargs):
        from cspy.crystal.phonon import jmol_phonon_xyz_file
        jmol_phonon_xyz_file(self, filename, **kwargs)

    def calculate_powder_pattern(
        self, method="platon", wavelength=None, **kwargs
    ):
        """Calculate the powder pattern for this crystal."""
        from cspy.ml.descriptors import PowderPattern

        if method == "pymatgen":
            return PowderPattern.from_pymatgen_cif_string(
                self.to_cif_string(),
                wavelength="CuKa" if wavelength is None else wavelength,
                **kwargs,
            )
        if method != "platon":
            raise NotImplementedError(f"Unknown powder-pattern method: {method}")

        if wavelength:
            try:
                pattern = PowderPattern.from_cif_string(
                    self.to_cif_string(wavelength=float(wavelength)), **kwargs
                )
            except ValueError:
                pattern = PowderPattern.from_cif_string(
                    self.to_cif_string(), wavelength=wavelength, **kwargs
                )
        else:
            pattern = PowderPattern.from_cif_string(self.to_cif_string(), **kwargs)

        return pattern

    def calculate_structure_factors(self, method="platon"):
        """Calculate the structure factors for this crystal using platon"""
        if method != "platon":
            raise NotImplementedError
        from cspy.ml.descriptors import StructureFactors

        return StructureFactors.from_cif_string(self.to_cif_string())

    def calculate_symmetry_functions(self, **kwargs):
        """Calculate the symmetry functions for this crystal"""
        from cspy.ml.descriptors import SymmetryFunctions

        return SymmetryFunctions.from_crystal(self, **kwargs)
    
    def neutralise_cell_charge(self):
        """Ensure total charge in cell is zero by smearing excess charge over all atoms"""
        from copy import deepcopy
        net_charge = 0
        num_charges = 0
        for mol in self.symmetry_unique_molecules():
            charges = [x.charge for x in mol.multipoles]
            mol_charge = np.sum(charges)
            num_charges += len(charges) 
            net_charge += mol_charge

        charge_correction = (net_charge / num_charges) * -1

        for ind, mol in enumerate(self.symmetry_unique_molecules()):
            mol.properties["multipoles"] = deepcopy(mol.multipoles)
            for x in mol.multipoles:
                x.charge += charge_correction

    @property
    def dmacrys_normalizing_matrix(self) -> np.ndarray:
        """
        Calculates the normalizing matrix used to define the orthonormal coordinate system in DMACRYS.
        Instructions are taken from the "Lattice Vector and Basis input" appendix of the DMACRYS manual.
        """
        if self.space_group.international_tables_number in (143, 168):
            LOG.info(
                "Based on DMACRYS manual, NEIGHCRYS cannot handle space groups 143 (P3) and 168 (P6). "
                "Returning dmacrys_normalizing_matrix as identity matrix."
            )
            return np.eye(3)

        alpha, beta, gamma = self.unit_cell.angles  # in radians
        a, b, c = self.unit_cell.lengths
        cell_type = self.unit_cell.cell_type

        if cell_type in ("triclinic", "monoclinic"):
            s = (alpha + beta + gamma) / 2
            V = (
                2
                * a
                * b
                * c
                * np.sqrt(
                    np.sin(s)
                    * np.sin(s - alpha)
                    * np.sin(s - beta)
                    * np.sin(s - gamma)
                )
            )
            a_star = b * c * np.sin(alpha) / V
            N = np.array(
                [
                    [1 / (a_star * c), 0, 0],
                    [
                        a * (np.cos(gamma) - np.cos(alpha) * np.cos(beta))
                        / (c * np.sin(alpha)),
                        b * np.sin(alpha) / c,
                        0,
                    ],
                    [a * np.cos(beta) / c, b * np.cos(alpha) / c, 1],
                ]
            )
        elif cell_type in ("tetragonal", "orthorhombic", "cubic"):
            N = np.array(
                [
                    [a / c, 0, 0],
                    [0, b / c, 0],
                    [0, 0, 1],
                ]
            )
        elif cell_type in ("trigonal", "hexagonal"):
            N = np.array(
                [
                    [a / c, -a / (2 * c), 0],
                    [0, np.sqrt(3) * a / (2 * c), 0],
                    [0, 0, 1],
                ]
            )
        else:
            LOG.info(
                f"Unexpected cell type: {cell_type}. Returning dmacrys_normalizing_matrix as identity matrix."
            )
            N = np.eye(3)
        return N

    @property
    def dmacrys_L_matrix(self) -> np.ndarray:
        """
        L matrix for centred unit cells. For more information look at Appendix F of DMACRYS manual.
        """
        # primitive, P
        L = np.eye(3)

        # aface, A
        if self.space_group.centering == "aface":
            L = np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.5],
                [0.0, 0.0, 0.5],
            ])
        # bface, B
        elif self.space_group.centering == "bface":
            L = np.array([
                [1.0, 0.0, 0.5],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.5],
            ])
        # cface, C
        elif self.space_group.centering == "cface":
            L = np.array([
                [1.0, 0.5, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 1.0],
            ])
        # body, I
        elif self.space_group.centering == "body":
            L = np.array([
                [-0.5, 0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, -0.5],
            ])
        # face, F
        elif self.space_group.centering == "face":
            L = np.array([
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ])
        # rcenter, R
        elif self.space_group.centering == "rcenter":
            L = np.array([
                [2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0],
                [1.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0],
                [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            ])

        return L

    @property
    def dmacrys_lattice_vectors(self) -> np.ndarray:
        """DMACRYS lattice vectors which are the traspose of normalizing matrix."""
        return self.unit_cell.c * np.dot(self.dmacrys_normalizing_matrix, self.dmacrys_L_matrix).T

    @property
    def dmacry_lattice_to_cspy_lattice_transformation(self) -> np.ndarray:
        """Calculates tranformation matrix from DMACRYS lattice vectors to CSPy lattice vectors."""
        return np.linalg.inv(self.dmacrys_lattice_vectors) @ self.unit_cell.lattice

    def dmacry_vector_to_cspy_vector(self, v: np.ndarray) -> np.ndarray:
        """
        Convert a 3D vector in DMACRYS orthonormal coordinates to CSPy cartesian coordinates.

        Parameters
        ----------
        v : np.ndarray
            The input vector in DMACRYS coordinates.

        Returns
        -------
        np.ndarray
            The vector in CSPy cartesian coordinates.
        """
        dmacrys_unit_cell = UnitCell(self.dmacrys_lattice_vectors)
        v_frac = dmacrys_unit_cell.to_fractional(v)
        return self.to_cartesian(v_frac)

    def to_vasp_conformation_input(self, vacuum: float = 20, i_mol: int = 0) -> "Crystal":
        """
        Put the ith molecule of the asymmetric unit into a large box and return
        the corresponding crystal object.

        Parameters
        ----------
        vacuum : float, optional
            The vacuum distance to add to the box. Default is 20.
        i_mol : int, optional
            The index of the molecule in the asymmetric unit to put in the box. Default is 0.

        Returns
        -------
        Crystal
            A new crystal object with the ith molecule in a large box.
        """
        mol = self.unit_cell_molecules()[i_mol]
        return mol.to_crystal(self, vacuum=vacuum)

    def to_extended_xyz_string(self) -> str:
        """Convert the crystal to an extended XYZ string format."""
        from cspy.ml.nnp.utils import extended_xyz_str

        atom_symbols = []
        atomic_numbers = []
        for mol in self.unit_cell_molecules():
            for el in mol.elements:
                atom_symbols.append(el.symbol)
                atomic_numbers.append(el.atomic_number)

        if "energy_eV" not in self.properties:
            self.properties["energy_eV"] = None
        if "atom_forces" not in self.properties:
            self.properties["atom_forces"] = None

        extended_xyz_string = extended_xyz_str(
            atomic_numbers=atomic_numbers,
            atom_symbols=atom_symbols,
            positions=self.unit_cell_atoms()["cart_pos"],
            lattice_vectors=self.unit_cell.lattice,
            energy=self.properties["energy_eV"],
            forces=self.properties["atom_forces"],
        )
        return extended_xyz_string

    def to_extended_xyz_file(self, output_fname: Union[str, None] = None) -> None:
        """Write the crystal to an extended XYZ file.

        Parameters
        ----------
        output_fname : str, optional
            The output filename. If None, uses the crystal title with .xyz extension.
        """
        if output_fname is None:
            output_fname = f"{self.titl}.xyz"
        with open(output_fname, "w") as f:
            f.write(self.to_extended_xyz_string())

    def to_ase_atoms(self) -> Atoms:
        """Convert the crystal to an ASE Atoms object."""
        uc_atoms = self.unit_cell_atoms()
        atoms = Atoms(
            numbers=uc_atoms["element"],
            positions=uc_atoms["cart_pos"],
            cell=self.unit_cell.lattice,
            pbc=True,
        )
        return atoms

    def single_point_with_mace(
        self,
        minimizer: Union["ASEMinimizer", None] = None,
        **kwargs
    ) -> None:
        """Single point calculation on the structure using Mace pre-trained model.

        Args:
            minimizer (ASEMinimizer): ASEMinimizer object.
            kwargs: Keyword arguments used to set up ASEMinimizer. See ASEMinimizer documentation for details.

        Results are stored in the properties dictionary of the crystal object.
        """
        if minimizer is None:
            from cspy.minimize import ASEMinimizer
            minimizer = ASEMinimizer(**kwargs)
        output_crys = minimizer.single_point(self, **kwargs)
        self.properties["mace_energy_eV"] = output_crys.properties["final_energy"]
        self.properties["mace_atom_forces"] = output_crys.properties["ase_atom_forces"]

    def single_point_with_dmacrys(self, multipole: str, silent: bool = False, **kwargs) -> None:
        """Single point calculation on the structure using DMACRYS.
        Results are stored in the properties dictionary of the crystal object.

        Important: To avoid confusion, explicitly specify 
            potential='nothing' or whatever in kwargs.

        Args:
            multipole (str): The multipole file to be used in the DMACRYS calculation.
            silent (bool): If True, suppresses the log messages.
        """
        from cspy.util.constants import EV2KJ_PER_MOL
        self.minimize(
            multipole=multipole,
            single_point=True,
            atom_forces=True,
            **kwargs,
        )
        kjmol_to_ev = 1/EV2KJ_PER_MOL
        self.properties['dmacrys_energy_eV'] = (
            self.properties['lattice_energy'] * kjmol_to_ev * len(self.unit_cell_molecules())
        )
        self.properties['dmacrys_atom_forces'] = self.properties['atom_forces']
        if not silent:
            LOG.info(f"Single point calculation with DMACRYS is done for {self.titl}")
            LOG.info(f"Energy (eV): {self.properties['dmacrys_energy_eV']}")
            LOG.info(f"Atom forces (eV/Ang): {self.properties['dmacrys_atom_forces']}")

    def single_point_with_sum_dmacrys_plus_mace(
        self, multipole: str, silent: bool = False, **kwargs
    ) -> None:
        """Single point calculation on the structure using DMACRYS and Mace pre-trained model.
        Results are stored in the properties dictionary of the crystal object.

        Args:
            multipole (str): The multipole file to be used in the DMACRYS calculation.
            silent (bool): If True, suppresses the log messages.
            kwargs: Look at other kwargs in MaceMinimizer.

        Important: To avoid confusion, explicitly specify
                   potential='nothing' or whatever in kwargs.
        """
        self.single_point_with_dmacrys(multipole, silent=silent, **kwargs)
        self.single_point_with_mace(silent=silent, **kwargs)
        self.properties["energy_eV"] = (
            self.properties["dmacrys_energy_eV"] + self.properties["mace_energy_eV"]
        )
        self.properties["atom_forces"] = (
            self.properties["dmacrys_atom_forces"] + self.properties["mace_atom_forces"]
        )

    def single_point_with_weighted_dmacrys_plus_mace(
        self,
        multipole: str,
        alpha: float = 0.5,
        silent: bool = False,
        **kwargs,
    ) -> None:
        """Single point calculation using DMACRYS and Mace pre-trained model.

        Results are stored in the properties dictionary of the crystal object.

        Args:
            multipole (str): The multipole file to be used in the DMACRYS calculation.
            alpha (float): Weight of the Mace energy and forces in the total energy and atom forces. Default is 0.5.
            silent (bool): If True, suppresses the log messages.
            kwargs: Additional keyword arguments for MaceMinimizer.

        Important: To avoid confusion, explicitly specify potential='nothing' or similar in kwargs.
        """
        self.single_point_with_dmacrys(multipole, silent=silent, **kwargs)
        self.single_point_with_mace(silent=silent, **kwargs)
        self.properties["energy_eV"] = (
            (1 - alpha) * self.properties["dmacrys_energy_eV"]
            + alpha * self.properties["mace_energy_eV"]
        )
        self.properties["atom_forces"] = (
            (1 - alpha) * self.properties["dmacrys_atom_forces"]
            + alpha * self.properties["mace_atom_forces"]
        )

    @classmethod
    def from_ase_atoms(cls, atoms: Atoms, titl: str = "from_ase_atoms") -> "Crystal":
        """Create a P1 crystal from an ASE Atoms object.

        Args:
            atoms (ase.Atoms): ASE Atoms object.
            titl (str): Title of the crystal.

        Returns:
            Crystal: P1 crystal.
        """
        sc = UnitCell.from_lengths_and_angles(
            lengths=atoms.cell.lengths(),
            angles=atoms.cell.angles(),
            unit="degrees",
        )
        positions = atoms.positions
        atomic_nums = atoms.numbers
        asymmetric_unit = AsymmetricUnit(
            [Element[x] for x in atomic_nums], sc.to_fractional(positions)
        )
        new_crystal = cls(sc, SpaceGroup(1), asymmetric_unit)
        new_crystal.titl = titl
        return new_crystal

    def has_close_contacts(self, cutoff: float = 0.7) -> bool:
        """Check if the crystal has close contacts between atoms.

        Finds the minimum distance between atoms in a unit cell and its
        26 neighboring cells. If the minimum distance is less than the cutoff,
        it is considered a close contact. A single cutoff value is used for all
        atoms in the crystal to keep the calculation cost low.

        Args:
            cutoff (float): The cutoff distance for close contacts.
                Default is 0.7 Angstrom.

        Returns:
            bool: True if the crystal has close contacts, False otherwise.
        """
        # Do not use self.as_P1_supercell() because the central cell must be
        # listed first. Only atom pairs with at least one atom in the central cell
        # are considered.

        c_standard = self.standardized()
        translations = np.array(
            np.meshgrid([0, -1, 1], [0, -1, 1], [0, -1, 1])
        ).reshape(3, -1).T
        frac_pos = c_standard.unit_cell_atoms()['frac_pos']
        supercell = []
        for translation in translations:
            pos_ = frac_pos + translation
            supercell.append(c_standard.to_cartesian(pos_))
        supercell = np.vstack(supercell)

        for i, p in enumerate(c_standard.unit_cell_atoms()['cart_pos']):
            dists = np.linalg.norm(supercell - p, axis=1)
            dists[i] = 1e5  # exclude the self distance
            if np.any(dists < cutoff):
                LOG.info(f"Close contact found for atom {i} in the crystal.")
                LOG.info(f"Distances below cutoff: {dists[dists < cutoff]}")
                return True

        return False
    
    def check_Buckingham_catastrophe(self, G=1):
        """Check if the crystal has Buckingham catastrophes (BC).
        This works by calling the unit_cell_connectivity function
        to find the molecules in the cell.
        Then we simply count the number of molecules and see if it
        matches the number we expect.
        We check firstly in the unit cell and if no BC is detected,
        check again in a 3x3x3 supercell.

        Args:
        -----
            G (int): The number of molecules in the asymmetric unit.
            Default is 1.

        Returns:
        --------
            bool: True if the crystal has a Buckingham catastrophe, False otherwise.
        """
        # from cspy.crystal.space_group import _sgdata_dict

        # n_symops = len(_sgdata_dict[str(self.space_group.international_tables_number)][0][8])
        from cspy.crystal.space_group import n_symops_from_SG
        n_symops = n_symops_from_SG(SG=self.space_group.international_tables_number, context='default')

        if hasattr(self, "_symmetry_unique_molecules"):
            delattr(self, "_symmetry_unique_molecules")
        if hasattr(self, "_unit_cell_molecules"):
            delattr(self, "_unit_cell_molecules")
        self.symmetry_unique_molecules()

        num_unique_mols = len(self._symmetry_unique_molecules)
        expected_uc_mols = num_unique_mols * n_symops
        num_uc_mols = len(self.unit_cell_molecules())

        if num_unique_mols == G and expected_uc_mols == num_uc_mols:
            u = 3
            v = 3
            w = 3
            num_unit_cells = u * v * w
            crystal_sc = self.as_P1_supercell((u, v, w))
            sc_mols = crystal_sc.unit_cell_molecules()
            
            if not len(sc_mols) == (num_uc_mols * num_unit_cells):
                # got a BC
                return True
            else:
                # no BC
                return False
            
        else:
            # got a BC
            return True

    def resize_volume(self, P: Union[float, int, List, Tuple, np.ndarray]) -> "Crystal":
        """
        Expand the unit cell lengths while keeping the fractional coordinates centroid of the
        molecules fixed. 

        Parameters
        ----------
        P: float, int, list, tuple, numpy.ndarray
            The fractional expansion factor for the crystal. If P is a float or an int,
            the same expansion factor is applied to all the unit cell lengths. If P is
            a list, tuple or numpy array, the expansion factors for the unit cell lengths
            are P[0], P[1], and P[2], respectively.
            Unit cell angles are kept fixed.

            Example: - To expand the crystal by 10% in all directions, P = 0.1.
                     - To expand the crystal by 10% in x, 20% in y, and 30% in z directions,
                          P = [0.1, 0.2, 0.3].

        Returns
        -------
        Crystal: Crystal
            The expanded (contracted) crystal.
        """
        lengths = self.unit_cell.lengths
        angles = self.unit_cell.angles

        molecules_coordinates_at_origin = []
        fractional_centroids = []
        for mol in self.asym_mols():
            fractional_centroids.append(self.to_fractional(mol.centroid))
            molecules_coordinates_at_origin.append(mol.positions - mol.centroid)

        if isinstance(P, (float, int)):
            P = np.array([P, P, P])
            LOG.info(f"Fraction of expansion along all lattice vectors: {P[0]}")
        elif isinstance(P, (list, tuple, np.ndarray)):
            P = np.array(P)
            LOG.info(
            f"Fraction of expansion along a, b, c lattice vectors: {P[0]}, {P[1]}, {P[2]}"
            )
        else:
            raise ValueError(
            "P must be a float, int, list, tuple or numpy array of floats."
            )

        new_unit_cell = UnitCell.from_lengths_and_angles(
            lengths * (1 + P), angles
        )

        asym_pos_frac = np.vstack(
            [
            new_unit_cell.to_fractional(mol_positions) + frac_centroid
            for mol_positions, frac_centroid in zip(
                molecules_coordinates_at_origin, fractional_centroids
            )
            ]
        )

        asym_nums = np.hstack([mol.atomic_numbers for mol in self.asym_mols()])
        new_asymmetric_unit = AsymmetricUnit(
            [Element[x] for x in asym_nums], asym_pos_frac
        )
        updated_crystal = Crystal(
            new_unit_cell, self.space_group, new_asymmetric_unit
        )
        return updated_crystal

    def to_poscar_string(
        self,
        fractional: bool = True,
        scaling: str = "1.0",
        comment: str = "Made using CSPy",
    ) -> List[str]:
        """Convert the crystal to a POSCAR string.

        Parameters
        ----------
        fractional : bool, optional
            If True, the positions are in fractional coordinates. If False, the positions
            are in Cartesian coordinates. Default is True.
        scaling : str, optional
            The scaling factor for the lattice vectors. Default is "1.0".
        comment : str, optional
            The comment line for the POSCAR file. Default is "Made using CSPy".

        Returns
        -------
        List[str]
            The POSCAR file as a list of strings.
        """
        elem_list = []
        pos_list = []
        for mol in self.unit_cell_molecules():
            for pos, el in zip(mol.positions, mol.elements):
                elem_list.append(el)
                if fractional:
                    pos_list.append(self.to_fractional(pos))
                else:
                    pos_list.append(pos)

        elem_set = list(set(elem_list))
        elem_counts = [elem_list.count(el) for el in elem_set]
        poscar = []
        latt_vecs = self.unit_cell.lattice
        poscar.append(comment)
        poscar.append(scaling)
        for i in range(3):
            poscar.append("  {:12.6f} {:12.6f} {:12.6f}".format(*latt_vecs[i]))
        poscar.append("  " + " ".join("{:>4s}".format(str(el)) for el in elem_set))
        poscar.append("  " + " ".join("{:4d}".format(count) for count in elem_counts))

        ordered_uc_atoms = []
        for el in elem_set:
            ordered_uc_atoms += [i for i, e in enumerate(elem_list) if e == el]

        ordered_coords = []
        if fractional:
            poscar.append("Direct")
        else:
            poscar.append("Cartesian")

        for i in ordered_uc_atoms:
            ordered_coords.append(pos_list[i])

        LOG.debug("Ordered coordinates in make_poscar are:")
        LOG.debug("\n".join([str(c) for c in ordered_coords]))
        for coord in ordered_coords:
            poscar.append("  {:12.6f} {:12.6f} {:12.6f}".format(*coord))
        return poscar

    def to_poscar_file(
        crystal,
        filename: str = "POSCAR",
        fractional: bool = True,
        scaling: str = "1.0",
        comment: str = "Made using CSPy",
    ) -> None:
        """Write the crystal to a POSCAR file.

        Parameters
        ----------
        filename : str, optional
            The name of the POSCAR file. Default is "POSCAR".
        fractional : bool, optional
            If True, the positions are in fractional coordinates. If False, the positions
            are in Cartesian coordinates. Default is True.
        scaling : str, optional
            The scaling factor for the lattice vectors. Default is "1.0".
        comment : str, optional
            The comment line for the POSCAR file. Default is "Made using CSPy".

        Returns
        -------
        None
        """
        poscar = crystal.to_poscar_string(
            fractional=fractional, scaling=scaling, comment=comment
        )
        with open(filename, "w") as f:
            f.write("\n".join(poscar))

    def write_vasp_input_files(
        self, working_directory: str = ".", settings: dict = None
    ) -> None:
        """Write VASP input files (POSCAR, POTCAR, INCAR) for the crystal.

        Parameters
        ----------
        working_directory : str, optional
            The working directory where the VASP input files will be written. Default is ".".
        settings : dict, optional
            The settings for the VASP input files. Default is reading from cspy.toml file in the current directory.
        """
        from cspy.formats.vasp_input import create_vasp_inputs

        if settings is None:
            LOG.info("No settings provided. Reading from cspy.toml file.")
            from cspy.configuration import CONFIG
            settings = CONFIG["vasp"]

        create_vasp_inputs(
            self,
            working_directory=working_directory,
            settings=settings,
        )

    def get_repulsion_dispersion_cutoff(self, minimum_cutoff: float = 15.0) -> float:
        """
        Get the repulsion-dispersion cutoff for the crystal structure.

        Parameters
        ----------
        minimum_cutoff : float, optional
            The default repulsion-dispersion cutoff value in Angstrom. Default is 15.0.

        Returns
        -------
        float
            The repulsion-dispersion cutoff value.
        """
        from scipy.spatial.distance import pdist

        cutoff = minimum_cutoff
        for mol in self.asym_mols():
            if len(mol.positions) <= 1:
                continue
            max_distance = np.max(pdist(mol.positions))
            cutoff = max(cutoff, 1.5 * max_distance)
        return cutoff
        
    def resize_volume(self, P):
        """
        Expand the whole crystal, including unit cell lengths and positions of
        molecules by a fractional. The centroid of the crystal is kept fixed.

        Parameters
        ----------
        P: float, int, list, tuple, numpy.ndarray
            The fractional expansion factor for the crystal. If P is a float or an int,
            the same expansion factor is applied to all the unit cell lengths. If P is
            a list, tuple or numpy array, the expansion factors for the unit cell lengths
            are P[0], P[1], and P[2], respectively.
            Unit cell angles are kept fixed.

            Example: - To expand the crystal by 10% in all directions, P = 0.1.
                     - To expand the crystal by 10% in x, 20% in y, and 30% in z directions,
                          P = [0.1, 0.2, 0.3].

        Returns
        -------
        Crystal: cspy.Crystal object
            The expanded (contracted) crystal.
        """
        lengths = self.unit_cell.parameters[:3]
        angles = self.unit_cell.parameters[-3:]

        moleclues_coordinates_at_origin = []
        fractional_centroids = []
        for mol_positions in self.asym_mols():
            fractional_centroids.append(self.to_fractional(mol_positions.centroid))
            moleclues_coordinates_at_origin.append(mol_positions.positions - mol_positions.centroid)

        if isinstance(P, (float, int)):
            P = np.array([P, P, P])
            LOG.info(f'Fraction of expansion in all directions: {P[0]}')
        elif isinstance(P, (list, tuple, np.ndarray)):
            P = np.array(P)
            LOG.info(f'Fraction of expansion in x, y, z directions: {P[0]}, {P[1]}, {P[2]}')
        else:
            raise ValueError('P must be a float, or an int, or a list, tuple or numpy array of floats.')
        
        new_unit_cell = UnitCell.from_lengths_and_angles(lengths * (1 + P),
                                                    np.radians(angles))

        asym_pos_frac = np.vstack([new_unit_cell.to_fractional(mol_positions) + frac_centroid for
                              mol_positions, frac_centroid in zip(moleclues_coordinates_at_origin, fractional_centroids)])

        asym_nums = np.hstack([mol.atomic_numbers for mol in self.asym_mols()])
        new_asymmetric_unit = AsymmetricUnit(
            [Element[x] for x in asym_nums], asym_pos_frac
        )
        updated_crystal = Crystal(new_unit_cell,
                           self.space_group,
                           new_asymmetric_unit)
        return updated_crystal
    
    def to_poscar_string(self, fractional=True, scaling="1.0", comment="Made using CSPy"):
        """Convert the crystal to a POSCAR string.

        Parameters
        ----------
        fractional: bool, optional
            If True, the positions are in fractional coordinates. If False, the positions
            are in Cartesian coordinates. Default is True.
        scaling: str, optional
            The scaling factor for the lattice vectors. Default is "1.0".
        comment: str, optional
            The comment line for the POSCAR file. Default is "Made using CSPy".

        Returns
        -------
        list: list of strings
            The POSCAR file as a list of strings.
        """

        elem_list = []
        pos_list = []
        for mol in self.unit_cell_molecules():
            for pos, el in zip(mol.positions, mol.elements):
                elem_list.append(el)
                if fractional:
                    pos_list.append(self.to_fractional(pos))
                else:
                    pos_list.append(pos)

        elem_set = list(set(elem_list))
        elem_counts = [elem_list.count(el) for el in elem_set]
        poscar = []
        latt_vecs = self.unit_cell.lattice
        poscar.append(comment)
        poscar.append(scaling)
        for i in range(0, 3):
            poscar.append("  "+"{:12.6f} {:12.6f} {:12.6f}".format(*latt_vecs[i]))
        poscar.append("  "+("{:>4s}"*len(elem_set)).format(*[str(el) for el in elem_set]))
        poscar.append("  "+("{:4d}"*len(elem_set)).format(*elem_counts))

        ordered_uc_atoms = []
        for el in elem_set:
            ordered_uc_atoms += [i for i,e in enumerate(elem_list) if e == el]     

        ordered_coords = []
        if fractional:
            poscar.append("Direct")
        else:
            poscar.append("Cartesian")

        for i in ordered_uc_atoms:
            ordered_coords.append(pos_list[i])

        LOG.debug("Ordered coordinates in make_poscar are:")
        LOG.debug("\n".join([str(c) for c in ordered_coords]))
        for coord in ordered_coords:
            poscar.append("  "+"{:12.6f} {:12.6f} {:12.6f}".format(*coord))
        return poscar
    
    def to_poscar_file(crystal, filename='POSCAR', fractional=True, scaling="1.0", comment="Made using CSPy"):
        """Write the crystal to a POSCAR file.

        Parameters
        ----------
        filename: str, optional
            The name of the POSCAR file. Default is "POSCAR".
        fractional: bool, optional
            If True, the positions are in fractional coordinates. If False, the positions
            are in Cartesian coordinates. Default is True.
        scaling: str, optional
            The scaling factor for the lattice vectors. Default is "1.0".
        comment: str, optional
            The comment line for the POSCAR file. Default is "Made using CSPy".
        
        Returns
        -------
        None
        """
        poscar = crystal.to_poscar_string(fractional=fractional, scaling=scaling, comment=comment)
        with open(filename, "w") as f:
            f.write("\n".join(poscar))

    def write_vasp_input_files(self, working_directory='.', settings=None):
        """ Write VASP input files (POSCAR, POTCAR, INCAR) for the crystal.

        Parameters
        ----------
        working_directory: str, optional
            The working directory where the VASP input files will be written. Default is ".".
        settings: dict, optional
            The settings for the VASP input files. Default is reading from cspy.toml file in the current directory.

        Returns
        -------
        None
        """
        from cspy.formats.vasp_input import create_vasp_inputs

        if not settings:
            LOG.info("No settings provided. Reading from cspy.toml file.")
            from cspy.configuration import CONFIG
            settings = CONFIG['vasp']
        create_vasp_inputs(self,
                          working_directory=working_directory,
                          settings=settings
                          )
