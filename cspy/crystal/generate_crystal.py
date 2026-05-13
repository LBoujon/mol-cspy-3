from cspy.crystal.space_group import SpaceGroup
from cspy.crystal.unit_cell import UnitCell
from cspy.crystal.crystal import AsymmetricUnit
from cspy.crystal.crystal import Crystal
from cspy.chem import Element
import itertools as it
from scipy.spatial import cKDTree as KDTree
from cspy.sample.sobol import sobol_vector
from copy import deepcopy
import numpy as np
from math import sin, cos, pi, sqrt, asin, acos, degrees
import time
from typing import Tuple, Union, Optional


def direct_matrix(a, b, c, ca, cb, cg, sg, v):
    return np.transpose([
        [a, b * cg, c * cb],
        [0, b * sg, c * (ca - cb * cg) / sg],
        [0, 0, v / (a * b * sg)]
    ])


def inverse_matrix(a, b, c, ca, cb, cg, sg, v):
    return np.array([
        [1 / a, 0.0, 0.0],
        [-cg / (a * sg), 1 / (b * sg), 0],
        [
            b * c * (ca * cg - cb) / v / sg,
            a * c * (cb * cg - ca) / v / sg,
            a * b * sg / v,
        ]
    ])


def vec2rot(v):
    x = (1.0 - v[0]) ** 0.5 * sin(2.0 * pi * v[1])
    y = (1.0 - v[0]) ** 0.5 * cos(2.0 * pi * v[1])
    z = v[0] ** 0.5 * sin(2.0 * pi * v[2])
    w = v[0] ** 0.5 * cos(2.0 * pi * v[2])
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y + 2*z*w, 2*x*z - 2*y*w],
        [2*x*y - 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z + 2*x*w],
        [2*x*z + 2*y*w, 2*y*z - 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])


class CrystalGenerator:

    def __init__(self, molecules, space_group, angles=(45.0, 135.0), tvp=2.5, nudge=0, adaptcell=False, asi=False):
        """Crystal structure generation class which can generate
        pseudorandom crystal structures for the initialized
        molecules and space group.

        Parameters
        ----------
        molecules : list of cspy.chem.molecule.Molecule
            A list of molecules to generate the crystals from.
        space_group : int
            The space groups to generated crystal structures in.
        angles : tuple of float
            The angle range to map the pseudorandom numbers to.
        tvp : float
            Target volume parameters used to scale the max volume of
            the unit cell.
        nudge : int
            Number of times each molecule may be nudged to relieve collisions.
            Only trigers after max volume reached.
        adaptcell : boolean
            Optimise cell parameters adaptively based on atomic collisions
        asi : boolean
            Allow centroids of molecules to be superimposed. (allow super impose)
        """

        self.space_group = SpaceGroup(space_group)
        self.lattice_type = self.space_group.lattice_type
        self.nsymops = len(self.space_group.symmetry_operations)
        self.molecules = [deepcopy(mol) for mol in molecules]
        self.orient_molecules()

        self.angles = (np.pi * angles[0] / 180, np.pi * angles[1] / 180)
        vdws = [[i.vdw for i in mol.elements] for mol in self.molecules]
        self.max_vdw = max([i for j in vdws for i in j])
        self.max_volume = sum(self.mol_box_vols()) * self.nsymops * tvp

        self.n_mols = len(self.molecules)
        self.n_atoms = [len(mol) for mol in self.molecules]
        self.asym_nums = []
        for mol in self.molecules:
            self.asym_nums += list(mol.atomic_numbers)

        covs = [[i.cov for i in mol.elements] for mol in self.molecules]
        self.max_cov = max([i for j in covs for i in j])
        self.sc_covs = []
        for cov in covs:
            self.sc_covs += cov * self.nsymops * 125
        self.sc_covs = np.array(self.sc_covs)

        self.trans = np.array(sorted(
            list(it.product(np.arange(-2, 3), repeat=3)),
            key=lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
        )[1:]
        self.opt_stage = 0
        self.nudge = nudge
        self.adaptcell = adaptcell
        self.asi = asi

    def orient_molecules(self):
        """Orient the molecules to an axis.
        """
        for mol in self.molecules:
            mol.positions = mol.positions_in_molecular_axis_frame(method="moi")
            
    @property
    def search_dimensions(self):
        """Works out the search dimensions of the initialized molecules
        and space group.

        Returns
        -------
            The search dimensions info.
        """
        if hasattr(self, "_search_dimensions"):
            return self._search_dimensions
        dims = {}
        mols = self.molecules
#        print(" Mols = ", mols)
        dims["positions"] = 3 * len(mols)
        dims["orientations"] = 3 * len([m for m in mols if len(m) > 1])
        sg_var = self.space_group.lattice_variables
        lengths = tuple(sorted(set([x for x in sg_var if x in ("a", "b", "c")])))
        angles = tuple(
            sorted(set([x for x in sg_var if x in ("alpha", "beta", "gamma")]))
        )
        dims["uc_lengths"] = lengths
        dims["uc_angles"] = angles
        ndim = (dims["positions"] + dims["orientations"]
                + len(dims["uc_lengths"]) + len(dims["uc_angles"]))
        self._search_dimensions = ndim, dims
        return self._search_dimensions

    def map_angle(self, value):
        """Maps a pseudorandom number to a cell angle in radians.

        Parameters
        ----------
        value : float
            The pseudorandom number.

        Returns
        -------
            A pseudorandom cell angle.
        """
        l, u = self.angles
        delta = u - l
        a = asin(2 * value - 1.0) / np.pi
        return (0.5 + a) * delta + l

    def convert_angles(self, p):
        """Converts the angle parameters to a tuple of cell angles for
        the chosen space group.

        Parameters
        ----------
        p : dict
            A dictionary of the cell angle params.

        Returns
        -------
            Cell angles.
        """
        if self.lattice_type == "triclinic":
            alpha = self.map_angle(p["alpha"])
            beta = self.map_angle(p["beta"])
            gamma = self.map_angle(p["gamma"])
            return alpha, beta, gamma
        elif self.lattice_type == "monoclinic":
            beta = self.map_angle(p["beta"])
            return np.pi / 2, beta, np.pi / 2
        elif self.lattice_type in ["orthorhombic", "tetragonal", "cubic"]:
            return np.pi / 2, np.pi / 2, np.pi / 2
        elif self.lattice_type == "rhombohedral":
            alpha = self.map_angle(p["alpha"])
            return alpha, alpha, alpha
        elif self.lattice_type == "hexagonal":
            return np.pi / 2, np.pi / 2, 2 * np.pi / 3

    def convert_lengths(self, p, mol_pos, inverse, sqrt_x, seed):
        """Converts the length parameters to a tuple of cell length for
        the chosen space group.

        Parameters
        ----------
        p : dict
            A dictionary of the cell length params.
        mol_pos : List of np.array
            A list of molecule atom positions.
        inverse : np.array
            The inverse matrix of the unit cell with unit lengths.
        sqrt_x : float
            Angular part of the volume of the unit cell.
        seed : int
            The seed of the quasi random number.

        Returns
        -------
            Cell lengths.
        """

        s_mins = np.zeros(3)
        for pos in mol_pos:
            i = pos @ inverse
            diff = np.max(i, axis=0) - np.min(i, axis=0)
            s_mins = np.maximum(s_mins, diff)
        s_mins += self.max_vdw * 2

        if s_mins[0] * s_mins[1] * s_mins[2] * sqrt_x > self.max_volume:
            return None

        if self.lattice_type in ["triclinic", "monoclinic", "orthorhombic"]:
            x, y, z = list(it.permutations((0, 1, 2), 3))[seed % 6]
            v = {}
            s_max = self.max_volume / (sqrt_x * s_mins[y] * s_mins[z])
            v[x] = s_mins[x] + p[x] * (s_max - s_mins[x])
            s_max = self.max_volume / (sqrt_x * v[x] * s_mins[z])
            v[y] = s_mins[y] + p[y] * (s_max - s_mins[y])
            s_max = self.max_volume / (sqrt_x * v[x] * v[y])
            v[z] = s_mins[z] + p[z] * (s_max - s_mins[z])
            return v[0], v[1], v[2]
        elif self.lattice_type in ["tetragonal", "hexagonal"]:
            x = seed % 2
            s_min_a = max(s_mins[0: 2])
            s_min_c = s_mins[2]
            if x == 0:
                s_max = sqrt(self.max_volume / (sqrt_x * s_min_c))
                a = s_min_a + p[0] * (s_max - s_min_a)
                s_max = self.max_volume / (sqrt_x * a**2)
                c = s_min_c + p[1] * (s_max - s_min_c)
            else:
                s_max = self.max_volume / (sqrt_x * s_min_a**2)
                c = s_min_c + p[1] * (s_max - s_min_c)
                s_max = sqrt(self.max_volume / (sqrt_x * c))
                a = s_min_a + p[0] * (s_max - s_min_a)
            return a, a, c
        elif self.lattice_type in ["rhombohedral", "cubic"]:
            s_max = (self.max_volume / sqrt_x)**(1/3)
            s_min = max(s_mins)
            a = s_min + p[0] * (s_max - s_min)
            return a, a, a

    def get_rotated_mol_pos(self, rotations : np.array) -> list:
        """
        Rotate the molecules with their sobol vector rotations
        and then return the atomic coordinates in a list.

        Parameters
        ----------
        rotations : list[np.array]
            List containing one rotation vector per molecule
    
        Returns
        -------
        mol_pos : list[np.array]
            List of numpy arrays. Each array is the atomic
            coordinates of a molecule after the molecule has
            been rotated.
        """

        mol_pos = []
        i = 0
        for ind, mol_orig in enumerate(self.molecules):
            mol = deepcopy(mol_orig)
            if len(mol) == 1:
                # if only 1 atom in molecule, don't rotate
                pos_i = mol.positions
            else:
                pos_i = mol.positions @ vec2rot(rotations[i])
            mol_pos.append(pos_i)
            i += 1

        return mol_pos

    def generate(self, seed: int, overrides: dict = dict(), vector: Optional[np.array] = None) -> Crystal:
        """Generates a random crystal structure for the input seed.
        Runs a crude minimization which attempts to remove all
        collision of a 5x5x5 supercell and then reduces its volume.
        Returns None if it is unable to remove collisions and keep the
        volume less then the max volume.

        Parameters
        ----------
        seed : int
            Must be greater then 0.

        overrides : dict
            Generate crystal with parameters from dictionary. 
            Only supports angles and lengths so far.

        Returns
        -------
            A randomly generated crystal structure or None if the seed
            is unable to create crystal structure with a cell volume
            less then the maximum volume with no molecular collisions.
        """

        # generate a quasirandom vector and map to the parameters
        ndim, dim = self.search_dimensions

        npos = dim["positions"]
        nrot = dim["orientations"]
        nang = len(dim["uc_angles"])
        nlen = len(dim["uc_lengths"])
        if not np.any(vector):
            vector = sobol_vector(seed, ndim)

        # set up molecular translations 
        translations = overrides.get("translations", vector[0: npos].reshape(-1, 3))
        if not self.asi:
            # bin structures where molecules are superimposed
            num_unique_translations = len(set(tuple(array) for array in translations))
            if num_unique_translations < self.n_mols:
                return None
        if not np.any(translations):
            translations = [np.zeros(3) for m in range(self.n_mols)]

        # map cell angles
        angle_params = {}
        for i, k in enumerate(dim["uc_angles"]):
            angle_params[k] = vector[i + npos + nrot]
        alpha, beta, gamma = self.convert_angles(angle_params)
        alpha = overrides.get("alpha", alpha)
        beta = overrides.get("beta", beta)
        gamma = overrides.get("gamma", gamma)

        ca = cos(alpha)
        cb = cos(beta)
        cg = cos(gamma)
        sg = sin(gamma)
        x = 1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg
        if x <= 0:
            return None

        # set up rotations of molecules
        rotations = vector[npos: npos + nrot].reshape(-1, 3)
        rotations = overrides.get("rotations", rotations)
        if not np.any(rotations):
            rotations = [np.zeros(3) for m in range(self.n_mols)]
        mol_pos = self.get_rotated_mol_pos(rotations)

        # map cell lengths
        sqrt_x = sqrt(x)
        unit_inverse = inverse_matrix(1, 1, 1, ca, cb, cg, sg, sqrt_x)
        lengths = self.convert_lengths(
            vector[npos + nrot + nang:], mol_pos, unit_inverse, sqrt_x, seed
        )
        if lengths is None:
            return None
        a, b, c = lengths
        a = overrides.get("a", a)
        b = overrides.get("b", b)
        c = overrides.get("c", c)

        # run a crude minimisation
        nudged_mols = np.zeros(self.n_mols)
        mol_collisions = np.zeros(self.n_mols)
        nudged_translations = [np.zeros(3) for m in range(self.n_mols)]
        nudge_count = 1
        self.opt_stage = 0
        prev_len = None
        prev_asym = None
        if self.nudge > 0:
            unexpanded_cell = [a, b, c, ca, cb, cg, sg]
        col_vectors = []
        opt_vector = np.array([0.0, 0.0, 0.0])
        opt_sf = np.array([0.0, 0.0, 0.0])
        expansion_count = 0
        valid = True

        # if cell is still valid, do loop where cell is expanded until collisions are relieved
        while valid == True:
            v = a * b * c * sqrt_x
            if v > self.max_volume:
                if self.nudge > 0:
                    # if nudge, don't give up
                    a, b, c, ca, cb, cg, sg = unexpanded_cell
                    v = a * b * c * sqrt_x
                    mol_col_ref = mol_collisions.copy()
                    mol_col_ref[nudged_mols == nudge_count] = 0
                    if np.max(mol_col_ref) == 0:
                        nudge_count = nudge_count + 1
                    mol_collisions[nudged_mols == self.nudge] = 0
                    if np.max(mol_collisions) == 0:
                        # if every clashing mol has reached max nudges, give up 
                        return None
                    mol_collisions[nudged_mols == nudge_count] = 0
                    test_nudged_translations = nudged_translations.copy()
                    for i, col_mol in enumerate(mol_collisions):
                        if col_mol > 0:
                            test_nudged_translations[i] = test_nudged_translations[i] + (col_vectors[i] / 1)
                            nudged_mols[i] = nudged_mols[i] + 1
                    
                    test_sc_pos, test_asym_pos= self.supercell(a, b, c, ca, cb, cg, sg, v, mol_pos, translations, test_nudged_translations)
                    collision, test_mol_collisions, react_col_vectors, sum_col_vectors = self.collision(test_sc_pos, [])
                    col_vectors = np.array(col_vectors)
                    react_col_vectors = np.array(react_col_vectors)
                    react_col_vectors[col_vectors == 0] = 0
                    # setup copy of vectors with 0s replaced with 1s to avoid horrible div by 0 errors
                    col_vectors_copy = col_vectors.copy().flatten()
                    col_vectors_copy[col_vectors.flatten() == 0] = 1
                    # damp proportionally to ratio of new and old vectors
                    damping = np.abs(col_vectors_copy) / (np.abs(col_vectors_copy) + np.abs(react_col_vectors).flatten())
                    damping[col_vectors.flatten() == 0] = 1
                    damping = np.reshape(damping, (-1, 3))
                    # product of new and old vectors is negative when they point in opposite directions
                    vector_prod = np.multiply(col_vectors, react_col_vectors)
                    mask_prod = [vector_prod >= 0]
                    damping = np.where(mask_prod, 1/damping, damping)[0]
                    col_vectors = np.multiply(col_vectors, np.abs(damping))
                    np.nan_to_num(col_vectors, 0)
                    for i, col_mol in enumerate(mol_collisions):
                        if col_mol > 0:
                            nudged_translations[i] = nudged_translations[i] + col_vectors[i] / 1
                    col_vectors = []
                    self.opt_stage = 0
                else:
                    return None

            sc_pos, asym_pos = self.supercell(a, b, c, ca, cb, cg, sg, v, mol_pos, translations, nudged_translations)
            collision, mol_collisions, col_vectors, sum_col_vectors = self.collision(sc_pos, col_vectors)
            
            if self.adaptcell:
                if np.all(self.opt_stage == 0) and not collision:

                    self.opt_stage = 1
                    opt_vector = self.find_void(sc_pos)
                    opt_sf = np.array([0.5, 0.5, 0.5])
                    prev_len = (a, b, c)
                    prev_asym = asym_pos
                    a, b, c = self.optimise_lengths(a, b, c, opt_vector, opt_sf)
                    v = a * b * c * sqrt_x
                    sc_pos, asym_pos = self.supercell(a, b, c, ca, cb, cg, sg, v, mol_pos, translations, nudged_translations)
                    collision, mol_collisions, col_vectors, sum_col_vectors = self.collision(sc_pos, col_vectors)
                    if collision:
                        a, b, c = prev_len
                        asym_pos = prev_asym
                        v = a * b * c * sqrt_x
                        sc_pos, asym_pos = self.supercell(a, b, c, ca, cb, cg, sg, v, mol_pos, translations, nudged_translations)
                    else:
                        prev_len = (a, b, c)
                        prev_asym = asym_pos

            if not self.adaptcell or self.opt_stage != 0:
                if self.opt_stage == 0 and not collision:
                    prev_len = (a, b, c)
                    prev_asym = asym_pos
                    self.opt_stage = 1
                elif self.opt_stage != 0 and collision:
                    a, b, c = prev_len
                    asym_pos = prev_asym
                    self.opt_stage += 1
                elif self.opt_stage != 0 and not collision:
                    prev_len = (a, b, c)
                    prev_asym = asym_pos

                if self.opt_stage == 4 and nlen == 3 or \
                        self.opt_stage == 3 and nlen == 2 or \
                        self.opt_stage == 2 and nlen == 1:
                    break

            if self.adaptcell and np.all(self.opt_stage == 0):
                expansion_count = expansion_count + 1
                col_dif_positive = False
                test_opt_sf = np.asarray([1, 1, 1])
                while not col_dif_positive:
                    test_a, test_b, test_c = self.optimise_lengths(a, b, c, sum_col_vectors, test_opt_sf)
                    test_v = test_a * test_b * test_c * sqrt_x
                    test_sc_pos, test_asym_pos = self.supercell(test_a, test_b, test_c, ca, cb, cg, sg, test_v, mol_pos, translations, nudged_translations)
                    test_collision, test_mol_collisions, test_col_vectors, test_sum_col_vectors = self.collision(test_sc_pos, [])
                    col_dif = sum_col_vectors - test_sum_col_vectors
                    if np.min(col_dif) < 0:
                        ext_array = np.asarray([1, 1, 1])
                        ext_array[col_dif >= 0] = 0
                        ext_array[col_dif < 0] = 1
                        test_opt_sf = test_opt_sf + ext_array
                        if np.max(test_opt_sf) > 3:
                            test_opt_sf = np.asarray([1, 1, 1])
                            a = a + 0.1
                            b = b + 0.1
                            c = c + 0.1
                            v = a * b * c * sqrt_x
                            if v > self.max_volume:
                                return None
                            sc_pos, asym_pos = self.supercell(a, b, c, ca, cb, cg, sg, v, mol_pos, translations, nudged_translations)
                            collision, mol_collisions, col_vectors, sum_col_vectors = self.collision(sc_pos, [])
                    else:
                        col_dif_positive = True

                opt_vector = sum_col_vectors
                col_dif[col_dif == 0] = 1
                opt_sf = np.divide(np.multiply(sum_col_vectors, test_opt_sf), col_dif)
                opt_sf = np.nan_to_num(opt_sf, 1)

            elif np.all(self.opt_stage == 0):
                opt_sf = np.array([1, 1, 1])

            # without this, cells can get stuck in a loop where the collision is very small but the change in cell params are smaller than rounding
            if np.max(opt_vector * opt_sf) < 0.001 and self.adaptcell and self.opt_stage == 0:
                a = a + 0.001
                b = b + 0.001
                c = c + 0.001
            else:
                a, b, c = self.optimise_lengths(a, b, c, opt_vector, opt_sf)

        # finally if there are no collisions, create and return the
        # crystal structure
        unit_cell = UnitCell.from_lengths_and_angles(
            (a, b, c), (alpha, beta, gamma)
        )
        asymmetric_unit = AsymmetricUnit(
            [Element[x] for x in self.asym_nums], np.vstack(asym_pos)
        )

        crystal = Crystal(
            unit_cell, deepcopy(self.space_group), asymmetric_unit
        )

        return crystal

    
    def supercell(self, a, b, c, ca, cb, cg, sg, v, mol_pos, translations, nudged_translations):
        """Sets up unit cell by constructing cell with sobol 
        cell parameters and then applying  sobol molecular 
        translations and applying symmetry operations. 
        Then builds 5x5x5 supercell and returns supercell
        atomic positions and asymetric unit atomic positions.


        Parameters
        ----------
        a, b, c : float
            Unit cell lattice parameters

        ca, cb, cg : float
            Cos alpha, Cos beta, and Cos gamma of unit cell angles

        sg : float
            Sin gamma of unit cell angles

        v : float
            Unit cell volume

        mol_pos : List of np.array
            A list of molecule atom positions.

        translations : np.array
            Molecular translations from sobol vector

        nudged_translations : np.array
            Molecular translations from nudge function

        Returns
        -------
        sc_pos : np.array
            An array of atoms' positions of the 5x5x5 supercell.

        asym_pos : np.array
            An array of atoms' positions of the asymetric unit.
        """

        direct = direct_matrix(a, b, c, ca, cb, cg, sg, v)
        inverse = inverse_matrix(a, b, c, ca, cb, cg, sg, v)

        # translate the molecules and generate a 5x5x5 supercell
        # ready for collision detection test
        trans = self.trans @ direct
        asym_pos = []
        sc_pos = []
        for i in range(self.n_mols):
            pos_i = (mol_pos[i] + nudged_translations[i] ) @ inverse + translations[i]
            asym_pos.append(pos_i)
            # uc_pos = self.space_group.apply_all_symops(pos_i)[1] @ direct
            uc_pos_frac = self.space_group.apply_all_symops(pos_i)[1]
            uc_pos = uc_pos_frac @ direct

            # move all the molecules so that their centroids are
            # in the unit cell
            centroids = self.space_group.apply_all_symops(
                np.reshape(translations[i], (1, 3)))[1]
            for j, cent in enumerate(centroids):
                if j == 0:
                    continue
                move_to_uc = (np.mod(cent, 1) - cent) @ direct
                start = j * self.n_atoms[i]
                end = (j + 1) * self.n_atoms[i]
                uc_pos[start: end] = uc_pos[start: end] + move_to_uc

            sc_pos.append(uc_pos)
            for tran in trans:
                sc_pos.append(uc_pos + tran)

        return np.vstack(sc_pos), asym_pos


    def collision(self, 
                  sc_pos : np.array, 
                  col_vectors : np.array, 
                  tol: float=0.5
                  ) -> Tuple[bool, np.array, np.array, np.array]:
        """There is a collision between the molecules if the distance
        between any two atoms are less then the sum of their covalent
        radii + 0.5 ang which is slightly greater then CCDC
        recommendations for deciding whether two atoms are bonded,
        the sum of covalent radii + 0.4.

        Parameters
        ----------
        sc_pos : np.array
            An array of atoms positions of the 5x5x5 supercell.

        col_vectors : np.array
            Vector of resultant collisions acting on each molecule.

        Returns
        -------
        collision : boolean
            True if there was a collision detected False if not.

        mol_collisions : np.array
            Magnitude of total collision experienced by each molecule.

        col_vectors : np.array
            Vector of resultant collisions acting on each molecule
            Used to determine nudge vector for each molecule.

        sum_col_vectors : np.array
            Sum of col_vectors over all molecules in asymetric unit.
            Used for adaptive lattice parameter optimisation.
        """
        start = 0
        mol_collisions = []
        if self.adaptcell:
            col_vectors_init = col_vectors.copy()
            col_vectors = []
        sum_col_neg = np.asarray([0, 0, 0])
        sum_col_pos = np.asarray([0, 0, 0])

        for i in range(self.n_mols):
            n_atoms = self.n_atoms[i]

            end = start + n_atoms
            mol_pos = sc_pos[start: end]
            mol_covs = self.sc_covs[start: end]
            mask_pos = np.ones(sc_pos.shape, dtype=bool)
            mask_pos[start: end] = False
            mask_sc = np.ones(self.sc_covs.shape, dtype=bool)
            mask_sc[start: end] = False
            #neighs = sc_pos[end:]
            #neighs = np.reshape(sc_pos[mask_pos], (int(len(sc_pos[mask_pos])/3), 3))
            neighs = np.reshape(sc_pos[mask_pos], (-1, 3))
            #neigh_covs = self.sc_covs[end:]
            neigh_covs = self.sc_covs[mask_sc]
            start += n_atoms * self.nsymops * 125

            # only generate neighs kd-tree for points within a box
            # around the test molecule
            mol_min = np.min(mol_pos, axis=0) - self.max_cov * 2 - tol
            mol_max = np.max(mol_pos, axis=0) + self.max_cov * 2 + tol

            box_idxs = np.all(
                np.logical_and(mol_min < neighs, neighs < mol_max), axis=1
            )
            sum_collision = 0
            if not np.any(box_idxs):
                mol_collisions.append(sum_collision)
                if len(col_vectors) < self.n_mols:
                    col_vectors.append(np.array([0, 0, 0]))
                continue

            neighs = neighs[box_idxs]
            neigh_covs = neigh_covs[box_idxs]

            tree_1 = KDTree(mol_pos, compact_nodes=False, balanced_tree=False)
            tree_2 = KDTree(neighs, compact_nodes=False, balanced_tree=False)
            contact = tree_1.query_ball_tree(tree_2, self.max_cov * 2 + tol)
            at_col_vectors = []
            for j, idxs in enumerate(contact):
                if len(idxs) == 0:
                    at_col_vector = np.zeros(3)
                    continue
                diff = mol_pos[j] - neighs[idxs]
                dist = np.sqrt(np.einsum('ij,ij->i', diff, diff))
                sum_radii = mol_covs[j] + neigh_covs[idxs] + tol

                if np.any(dist < sum_radii):
                    if not self.nudge > 0 and not self.adaptcell:
                        return True, None, [], []
                    else:
                        sum_collision = sum_collision + np.sum(np.maximum(0, sum_radii - dist))
                        if len(col_vectors) < self.n_mols:
                            # if no collision, below gives float between 0 and 1
                            dist[dist == 0] = 1E-9
                            col_factor = sum_radii / dist
                            # set < 1 values to 1 so target_diff is unchanged
                            col_factor[col_factor < 1] = 1
                            col_factor = \
                                np.stack((col_factor, col_factor, col_factor), axis=1)
                            target_diff = diff * col_factor
                            neigh_col_vector = target_diff - diff
                            # if atom dist = 0, set vector to sum_radii
                            neigh_col_vector = np.nan_to_num(neigh_col_vector, \
                                                            nan=np.stack((sum_radii, sum_radii, sum_radii), axis=1))
                            neigh_x_vectors, neigh_y_vectors, neigh_z_vectors = np.hsplit(neigh_col_vector, 3)
                            x_comp = np.min(np.minimum(0, neigh_x_vectors)) + np.max(np.maximum(0, neigh_x_vectors))
                            y_comp = np.min(np.minimum(0, neigh_y_vectors)) + np.max(np.maximum(0, neigh_y_vectors))
                            z_comp = np.min(np.minimum(0, neigh_z_vectors)) + np.max(np.maximum(0, neigh_z_vectors))
                            at_col_vector = np.array([x_comp, y_comp, z_comp])
                else:
                    at_col_vector = np.zeros(3)

                if len(col_vectors) < self.n_mols:
                    at_col_vectors.append(at_col_vector)

            if len(col_vectors) < self.n_mols:
                if len(at_col_vectors) == 0:
                    x_comp = 0
                    y_comp = 0
                    z_comp = 0
                else:
                    at_x_vectors, at_y_vectors, at_z_vectors = np.hsplit(np.array(at_col_vectors), 3)
                    x_neg = np.min(np.minimum(0, at_x_vectors))
                    x_pos = np.max(np.maximum(0, at_x_vectors))
                    x_comp =  x_neg + x_pos
                    y_neg = np.min(np.minimum(0, at_y_vectors))
                    y_pos = np.max(np.maximum(0, at_y_vectors))
                    y_comp = y_neg + y_pos
                    z_neg = np.min(np.minimum(0, at_z_vectors))
                    z_pos = np.max(np.maximum(0, at_z_vectors))
                    z_comp = z_neg + z_pos
                    if self.adaptcell:
                        sum_col_neg = sum_col_neg + np.asarray([x_neg, y_neg, z_neg])
                        sum_col_pos = sum_col_pos + np.asarray([x_pos, y_pos, z_pos])
                col_vector = np.array([x_comp, y_comp, z_comp])
            
            mol_collisions.append(sum_collision)
            # only keep vectors from collisions with original cell dimensions
            if len(col_vectors) < self.n_mols:
                col_vectors.append(col_vector)

        if self.adaptcell:
            #sum_col_vectors = np.sum(np.abs(np.asarray(col_vectors)), axis=0)
            sum_col_vectors = ( sum_col_neg * -1 ) + sum_col_pos
        else:
            sum_col_vectors = []

        if self.nudge > 0 or self.adaptcell:
            if self.adaptcell and not len(col_vectors_init) == 0:
                col_vectors = col_vectors_init
            if np.max(mol_collisions) > 0:
                return True, np.array(mol_collisions), col_vectors, sum_col_vectors
            else:
                return False, np.array(mol_collisions), col_vectors, sum_col_vectors
        else:
            return False, None, [], []
        

    def find_void(self, sc_pos : np.array) -> np.array:
        """Find the size of any voids in the x, y, and z directions.
        Return a compression vector that can be used to reduce cell lengths.
        If all components are applied at the same time, this may cause unforseen
        collisions.

        Parameters
        ----------
        sc_pos : numpy array
            Coordinates of atoms in supercell

        Returns
        -------
        compression_vector : numpy array
            Vector with 3 components for x, y, and z for possible
            cell compression
        """

        Z_void_length = 999999
        X_void_length = 999999
        Y_void_length = 999999
        start = 0

        for i in range(self.n_mols):
            n_atoms = self.n_atoms[i]

            end = start + n_atoms
            mol_pos = sc_pos[start: end]
            mol_covs = self.sc_covs[start: end]
            mask_pos = np.ones(sc_pos.shape, dtype=bool)
            mask_pos[start: end] = False
            mask_sc = np.ones(self.sc_covs.shape, dtype=bool)
            mask_sc[start: end] = False
            neighs = np.reshape(sc_pos[mask_pos], (-1, 3))
            neigh_covs = self.sc_covs[mask_sc]
            start += n_atoms * self.nsymops * 125

            for j, atom in enumerate(mol_pos):
                # look for atoms in column around test atom
                atom_min = atom - self.max_cov * 2 - 0.5
                atom_max = atom + self.max_cov * 2 + 0.5
                neighs_XY = neighs[:,[0,1]]
                neighs_XZ = neighs[:,[0,2]]
                neighs_YZ = neighs[:,[1,2]]

                Z_idxs = np.all(
                    np.logical_and(atom_min[[0,1]] < neighs_XY, neighs_XY < atom_max[[0,1]]), axis=1
                )
                Y_idxs = np.all(
                    np.logical_and(atom_min[[0,2]] < neighs_XZ, neighs_XZ < atom_max[[0,2]]), axis=1
                )
                X_idxs = np.all(
                    np.logical_and(atom_min[[1,2]] < neighs_YZ, neighs_YZ < atom_max[[1,2]]), axis=1
                )

                if np.any(X_idxs):
                    diff = mol_pos[j] - neighs[X_idxs]
                    X_min_dist = np.min(np.sqrt(np.einsum('ij,ij->i', diff, diff)) - neigh_covs[X_idxs] - mol_covs[j] - 0.5)
                    if X_min_dist < X_void_length:
                        X_void_length = X_min_dist

                if np.any(Y_idxs):
                    diff = mol_pos[j] - neighs[Y_idxs]
                    Y_min_dist = np.min(np.sqrt(np.einsum('ij,ij->i', diff, diff)) - neigh_covs[Y_idxs] - mol_covs[j] - 0.5)
                    if Y_min_dist < Y_void_length:
                        Y_void_length = Y_min_dist

                if np.any(Z_idxs):
                    diff = mol_pos[j] - neighs[Z_idxs]
                    Z_min_dist = np.min(np.sqrt(np.einsum('ij,ij->i', diff, diff)) - neigh_covs[Z_idxs] - mol_covs[j] - 0.5)
                    if Z_min_dist < Z_void_length:
                        Z_void_length = Z_min_dist

        compression_vector = np.array([X_void_length, Y_void_length, Z_void_length])

        return np.round(compression_vector, 3)
    
    def optimise_lengths(self, a, b, c, opt_vector, opt_sf):
        """Changes the cell lengths by increasing the smallest length
        if expanding and decreasing a length if contracting.

        Parameters
        ----------
        a : float
            Cell length.
        b : float
            Cell length.
        c : float
            Cell length.

        Returns
        -------
            Cell lengths
        """
        if self.adaptcell:
            opt_vector = opt_vector * opt_sf

        if self.lattice_type in ["triclinic", "monoclinic", "orthorhombic"]:
            if self.opt_stage == 0:
                if self.adaptcell:
                    a += opt_vector[0]
                    b += opt_vector[1]
                    c += opt_vector[2]
                else:
                    min_len = min([a, b, c])
                    if a == min_len:
                        a += 1
                    elif b == min_len:
                        b += 1
                    elif c == min_len:
                        c += 1
            else:
                if self.adaptcell and self.opt_stage == 0:
                    a -= opt_vector[0]
                    b -= opt_vector[1]
                    c -= opt_vector[2]
                else:
                    max_len = sorted([a, b, c], reverse=True)[self.opt_stage - 1]
                    if a == max_len:
                        a -= 1
                    elif b == max_len:
                        b -= 1
                    elif c == max_len:
                        c -= 1
            return a, b, c
        elif self.lattice_type in ["tetragonal", "hexagonal"]:
            if self.opt_stage == 0:
                if self.adaptcell:
                    a += np.min([opt_vector[0], opt_vector[1]])
                    c += opt_vector[2]
                else:
                    min_len = min([a, c])
                    if a == min_len:
                        a += 1
                    elif c == min_len:
                        c += 1
            else:
                if self.adaptcell and self.opt_stage == 0:
                    a -= np.min([opt_vector[0], opt_vector[1]])
                    c -= opt_vector[2]
                else:
                    max_len = sorted([a, c], reverse=True)[self.opt_stage - 1]
                    if a == max_len:
                        a -= 1
                    elif c == max_len:
                        c -= 1
            return a, a, c
        elif self.lattice_type in ["rhombohedral", "cubic"]:
            if self.opt_stage == 0:
                if self.adaptcell:
                    a += np.min([opt_vector[0], opt_vector[1], opt_vector[2]])
                else:
                    a += 1
            else:
                if self.adaptcell and self.opt_stage == 0:
                    a -= np.min([opt_vector[0], opt_vector[1], opt_vector[2]])
                else:
                    a -= 1
            
            return a, a, a

    def mol_box_vols(self):
        """The minimum volume of a box required enclose the molecule.

        Returns
        -------
            A list of box volumes for each molecule.
        """
        box_vols = []
        for mol in self.molecules:
            min_xyz = np.array([np.inf, np.inf, np.inf])
            max_xyz = np.array([-np.inf, -np.inf, -np.inf])
            for element, pos in zip(mol.elements,  mol.positions):
                for i in range(3):
                    min_xyz[i] = min([min_xyz[i], pos[i] - element.vdw])
                    max_xyz[i] = max([max_xyz[i], pos[i] + element.vdw])
            box_vols.append(np.prod(max_xyz - min_xyz))
        return box_vols