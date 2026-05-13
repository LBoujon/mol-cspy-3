import numpy as np
import logging
import random
from numpy.random import (
    Generator,
    default_rng,
)
import time
from collections import OrderedDict
from copy import deepcopy
from scipy.spatial.transform import Rotation
from cspy.crystal import Crystal, SpaceGroup, AsymmetricUnit, UnitCell

LOG = logging.getLogger(__name__)

# Constants
epsilon = 1e-5
R_const = 8.3144598  # unit:J/(mol*K) source:CODATA 2015


def calc_beta(temp):
    """Calculate beta from temperature"""
    beta = R_const * temp
    beta = 1 / (beta + epsilon)
    return beta


def mc_accept(en_new, en_old, beta):
    """Decide whether to accept the MC move
    
    Args:
        en_new: float, energy of new generated crystal by MC move
        en_old: float, energy of old crystal
        beta: float, temperature related factor, calculated as1/kT
    
    Returns:
        accept: boolean, accept the new crystal or not
        lowen: boolean, new crystal has lower energy than old one or not
    """
    # Calculate energy difference and convert the energy unit from kJ to J
    delta_E_kj = en_new - en_old
    delta_E = delta_E_kj * 1000

    # Decide whether to accpet the new structure
    lowen = False
    accept = False
    if delta_E > 0:
        # If energy rise, calculate the Boltzmann factor and generate a random
        # number
        rng = default_rng()
        mc_frac = np.exp(-beta * delta_E)
        ran_num = rng.random()
        if ran_num < mc_frac:
            # If the random number is less than the Boltsmann factor, accept the
            # new crystal
            accept = True
    else:
        # If energy decrease, accept the new crystal
        lowen = True
        accept = True
    return accept, lowen

class MC_Move():
    def __init__(self, param_list):
        self.probability = param_list[0]
        self.cutoff = param_list[1]

class Translation_Move(MC_Move):
    def __init__(self, param_list):
        super().__init__(param_list)

class Rotation_Move(MC_Move):
    def __init__(self, param_list):
        super().__init__(param_list)

class Unit_Cell_Length_Move(MC_Move):
    def __init__(self, param_list):
        super().__init__(param_list)

class Unit_Cell_Angle_Move(MC_Move):
    def __init__(self, param_list):
        super().__init__(param_list)

class Unit_Cell_Volume_Move(MC_Move):
    def __init__(self, param_list):
        super().__init__(param_list)

class Reflect_Move(MC_Move):
    def __init__(self, param_list):
        super().__init__(param_list)
        
class Atoms_Move(MC_Move):
    def __init__(self, param_list):
        super().__init__(param_list)
        self.n_move = int(param_list[2])


class MC:
    """Controller of MC move

    Contains all the available Monte Carlo moves, sat expansion and
    auto-arrangement of choosing probability as well as cut-off for each chose
    move according to space group.
    """

    def __init__(
        self, 
        move_list, 
        auto_prob=False, 
        auto_cutoff=False, 
        all_s=False, 
        sat_s=False, 
        cut_off_scale=1.0,
        rng=None,
    ):
        """Initialize

        Args:
            move_list: list, contain name of MC moves, choosing probabiliy and
            cut-off
            auto_prob: boolean, auto rearrange the choosing probabiliy
            according to space group or not
            auto_cutoff: boolean, auto rearrange the move cut-off according to
            space group or not
            all_s: boolean, apply all kinds of MC move at each step or not
            sat_s: boolean, use sat expansion after each move or not
            cut_off_scale: float, scale cut-off on-th-fly, used for threshold
            only
            rng: numpy rng object
        """
        self.all_s = all_s
        self.prob_s = auto_prob
        self.cut_s = auto_cutoff
        self.moves = self.get_moves(move_list, cut_off_scale=cut_off_scale)
        self.sat_s = sat_s
        
        if rng is None:
            self.rng = default_rng()
        else:
            self.rng = default_rng(rng)
        #LOG.info(self.rng.bit_generator.state)
        #self.rng.random()

        self.sg = None
        self.num_mol = None
        self.index = None
        self.z = None

    @property
    def mc_move(self):
        """Whether to change all degrees of freedom or one degree of
        freedom at one MC step and set the mc_move function
        """
        if self.all_s:
            return self.move_all
        else:
            return self.move_one

    def get_moves(self, move_list, cut_off_scale=1.0):
        """Get the probability and cutoff for each type of MC move

        Args:
            move_list: list, contain name of MC moves, choosing probabiliy and
            cut-off

        Returns:
            moves: OrderedDict, contain name of ordered MC moves, choosing
            probabiliy and cut-off

        Raises:
            ValueError: invalid input MC move information
        """
        move_types = {
            'tra': Translation_Move,
            'rot': Rotation_Move,
            'u_l': Unit_Cell_Length_Move,
            'u_a': Unit_Cell_Angle_Move,
            'vol': Unit_Cell_Volume_Move,
            'ref': Reflect_Move,
            'atoms': Atoms_Move,
        }

        moves = OrderedDict()
        prob_tmp = 0.0
        cut_tmp = 0.0
        for i_move in move_list:
            try:
                type_tmp = str(i_move[0])
                move_params = list(i_move[1:])

                if not self.prob_s:
                    move_params[0] = prob_tmp + float(move_params[0])

                move_params[1] = float(move_params[1]) * cut_off_scale
                
                moves[type_tmp] = move_types[type_tmp](move_params)

            except ValueError:
                raise ValueError(
                    "wrong type variable for move input, should be [str float float]"
                )
        return moves

    def update_sg(self, crys_res):
        """Update the information from space group, including number of
        molecules Z, probability and cutoff for MC move

        Args:
            sg: int, space group
            crys_res: string, res string of a crystal structure
        """
        crystal_systems = [
            "triclinic",
            "monoclinic",
            "orthorhombic",
            "tetragonal",
            "rhombohedral",
            "hexagonal",
            "cubic",
        ]

        #self.index = CrystallographicSpaceGroups.get(int(self.sg)).crystal_system.index
        crys_tmp = Crystal.from_shelx_string(crys_res)
        self.sg = crys_tmp.space_group.international_tables_number
        self.index = crystal_systems.index(crys_tmp.space_group.lattice_type) + 1
        self.z = len(crys_tmp.unit_cell_molecules())
        self.num_mol = len(crys_tmp.symmetry_unique_molecules())
        self.num_atoms = len(crys_tmp.unit_cell_atoms()["label"])

        # Automatically adjust the probability and cutoff according to the space
        # group
        if self.prob_s and not self.all_s:
            self.calc_prob()
        if self.cut_s:
            self.calc_cut()
        # LOG.info("Calculated move set probabilities and cut-offs", self.moves)

    def choose_move(self):
        """Choose the MC move randomly according to the MC move probability

        Returns:
            i_move: string, name of chosen MC move
        """
        rand = self.rng.random()
        for move_type, move in self.moves.items():
            if rand < move.probability:
                return move_type


    def move_one(self, old_res):
        """MC move changing one degree of freedom at a time

        Args:
            old_res: string, res string for old crystal

        Returns:
            new_res: string, res string of generated crystal by MC move
            list, [string, float], [name of chosen MC move, random number 
            generated for move displacement]

        Raises:
            NameError: invalid type of MC move
        """
        crys_old = Crystal.from_shelx_string(old_res)
        while True:

            move = self.choose_move()

            if move == "tra":
                crys_new, rand = self.translation(crys_old)
            elif move == "rot":
                crys_new, rand = self.rotation(crys_old)
            elif move == "u_l":
                crys_new, rand = self.unit_length(crys_old)
            elif move == "u_a":
                crys_new, rand = self.unit_angle(crys_old)
            elif move == "vol":
                crys_new, rand = self.volume(crys_old)
            elif move == "ref":
                crys_new, rand = self.reflect(crys_old)
            elif move == "atoms":
                crys_new, rand = self.atoms(crys_old)
            else:
                raise NameError("Incorrect type of MC move %s" % move)

            # Check overlapping and sat expansion to remove the overlap,
            # generally turned off
            #if self.sat_s:
            #    crys_new = self.check_overlap_by_convex_hull(crys_new)

            if crys_new is None:
                continue
            else:
                new_res = crys_new.to_shelx_string()
                return new_res, [move, rand]

    def move_all(self, old_res):
        """MC move changing all degrees of freedom at a time (Test)
        Args:
            old_res: string, res string for old crystal

        Returns:
            new_res: string, res string of generated crystal by MC move
            list: [string, float], ["all", 0.0]
        """
        crys_old = Crystal.from_shelx_string(old_res)
        while True:

            # Apply all the moves to the crys. Should change all the degree of
            # freedom
            crys_tmp, rand = self.translation(crys_old)
            crys_tmp, rand = self.rotation(crys_tmp)
            crys_tmp, rand = self.unit_length(crys_tmp)
            crys_tmp, rand = self.unit_angle(crys_tmp)
            crys_new, rand = self.volume(crys_tmp)

            # Check overlapping and sat expansion to remove the overlap,
            # generally turned off
            #if self.sat_s:
            #    crys_new = self.check_overlap_by_convex_hull(crys_new)

            if crys_new is None:
                continue
            else:
                new_res = crys_new.to_shelx_string()
                return new_res, ["all", 0.0]
            
    def atoms(self, crystal):
        """atom translation move

        Random direction and random translation distance

        Args:
            crystal: Crystal(core.models), crystal before MC move

        Returns:
            crystal: Crystal(core.models), crystal after MC move
            rand: float, random number generated for move displacement
        """
        
        old_positions = list(crystal.asymmetric_unit.positions)
        inds =  list(range(len(old_positions)))
        elements = crystal.asymmetric_unit.elements
        labels = crystal.asymmetric_unit.labels
        new_positions = deepcopy(old_positions)
        
        n_move = self.moves["atoms"].n_move
        # inds = self.rng.integers(low=0, high=len(old_positions)-1, endpoint=True)
        inds = self.rng.choice(
            inds, 
            size=n_move,
            replace=False,
        )
        
        for i in inds:
            #i = self.rng.integers(low=0, high=len(old_positions)-1, endpoint=True)
            atom_coords = old_positions[i]
        
            theta = self.rng.random() * np.pi
            sin_theta = np.sin(theta)
            phi = self.rng.random() * 2 * np.pi
            rand = self.rng.random()
            R = rand * self.moves["atoms"].cutoff

            trans_vector = np.array([
                R * sin_theta * np.cos(phi), R * sin_theta * np.sin(phi), R * np.cos(theta)
            ])
            
            new_coords = atom_coords + trans_vector
            new_positions[i] = new_coords
            
            LOG.debug("Translation of atom %i by vector %s", i, trans_vector)
        
        asym_unit = AsymmetricUnit(
            elements=elements,
            labels=labels,
            positions=new_positions
        )
        
        return Crystal(
            asymmetric_unit=asym_unit,
            unit_cell=deepcopy(crystal.unit_cell), 
            space_group=deepcopy(crystal.space_group),
        ), rand


        #Move the centroid of molecules back to unit cell
        for mol in mols:
            cen_frac = crystal.to_fractional(mol.centroid)
            move_frac = np.floor(cen_frac) * -1
            move = crystal.to_cartesian(move_frac)
            mol.positions += move


    def translation(self, crystal):
        """Molucular translation move

        Random direction and random translation distance

        Args:
            crystal: Crystal(core.models), crystal before MC move

        Returns:
            crystal: Crystal(core.models), crystal after MC move
            rand: float, random number generated for move displacement
        """
        mols = crystal.asym_mols()
        im = self.rng.integers(low=0, high=len(mols)-1, endpoint=True)# if len(mols) > 0 else 0

        theta = self.rng.random() * np.pi
        sin_theta = np.sin(theta)
        phi = self.rng.random() * 2 * np.pi
        rand = self.rng.random()
        R = rand * self.moves["tra"].cutoff

        trans_vector = np.array([
            R * sin_theta * np.cos(phi), R * sin_theta * np.sin(phi), R * np.cos(theta)
        ])

        mols[im].translate(trans_vector)

        #Move the centroid of molecules back to unit cell
        for mol in mols:
            cen_frac = crystal.to_fractional(mol.centroid)
            move_frac = np.floor(cen_frac) * -1
            move = crystal.to_cartesian(move_frac)
            mol.positions += move

        LOG.debug("Translation of molecule %i by vector %s", im, trans_vector)
        return Crystal.from_unit_cell_sg_molecules(
            deepcopy(crystal.unit_cell), 
            deepcopy(crystal.space_group),
            mols,
        ), rand

    def rotation(self, crystal):
        """Molucular rotation move

        Random direction and random rotation angle, implemented by quaternion

        Args:
            crystal: Crystal(core.models), crystal before MC move

        Return:
            crystal: Crystal(core.models), crystal after MC move
            rand: float, random number generated for move displacement
        """
        mols = crystal.asym_mols()
        im = self.rng.integers(low=0, high=len(mols)-1, endpoint=True)# if len(mols) > 0 else 0

        theta = self.rng.random() * np.pi
        sin_theta = np.sin(theta)
        phi = self.rng.random() * 2 * np.pi
        rand = self.rng.random()
        angle = (rand * 2 - 1) * np.pi * self.moves["rot"].cutoff

        cos_a_2 = np.cos(angle / 2)
        sin_a_2 = np.sin(angle / 2)
        # Use the quaternion to specify the rotation
        rotation_quaternion = np.array([
            sin_a_2 * sin_theta * np.cos(phi),
            sin_a_2 * sin_theta * np.sin(phi),
            sin_a_2 * np.cos(theta),
            cos_a_2,
        ])
        r = Rotation.from_quat(rotation_quaternion)

        center = mols[im].centroid
        mols[im].positions -= center
        mols[im].positions = r.apply(mols[im].positions)
        mols[im].positions += center

        LOG.debug(
            "Rotation of molecule %i by quaternion %s", im, rotation_quaternion
        )
        return Crystal.from_unit_cell_sg_molecules(
            deepcopy(crystal.unit_cell), 
            deepcopy(crystal.space_group),
            mols,
        ), rand

    def unit_length(self, crystal):
        """Unit cell length change

        The degree of freedom related could be difference due to space group

        Args:
            crystal: Crystal(core.models), crystal before MC move

        Return:
            crystal: Crystal(core.models), crystal after MC move
            rand: float, random number generated for move displacement
        """
        index = self.index
        rand = self.rng.random()
        R = (rand * 2 - 1) * self.moves["u_l"].cutoff
        a, b, c, alpha, beta, gamma = crystal.unit_cell.parameters
        mols = crystal.asym_mols()

        # Change the unit cell length accroding to crystal system
        if index == 1 or index == 2 or index == 3:
            rand_d = self.rng.integers(low=0, high=2, endpoint=True)
            if rand_d == 0:
                a += R
            elif rand_d == 1:
                b += R
            elif rand_d == 2:
                c += R

        elif index == 4 or index == 6:
            rand_d = self.rng.integers(low=0, high=1, endpoint=True)
            if rand_d == 0:
                a += R
                b = a
            elif rand_d == 1:
                c += R

        elif index == 5 or index == 7:
            rand_d = "A"
            a += R
            b = a
            c = a

        LOG.debug(
            "Unit cell length change of dimension %i " "by distance %8.3f", rand_d, R
        )
        return Crystal.from_unit_cell_sg_molecules(
            UnitCell.from_lengths_and_angles([a, b, c], [alpha, beta, gamma], unit="degree"),
            deepcopy(crystal.space_group),
            mols,
        ), rand

    def unit_angle(self, crystal):
        """Unit cell angle change.

        The range of the accepted angle is from 45 to 135 degrees except from
        hexagonal crystal. The change of angle has a higher probability to get
        closer to 90 degrees by a linear shift. The degree of freedom related
        could be difference due to space group, so in some lattice system this
        move is a waste.

        Args:
            crystal: Crystal(core.models), crystal before MC move

        Return:
            crystal: Crystal(core.models), crystal after MC move
            rand: float, random number generated for move displacement
        """
        index = self.index
        # Specify the up and down limit for unit cell angle to prevent the unit
        # cell being too flat
        c_angle = 90.0
        u_angle = 135.0
        d_angle = 45.0
        rand = -0.1
        a, b, c, alpha, beta, gamma = crystal.unit_cell.parameters
        mols = crystal.asym_mols()

        # Change the unit angle length accroding to the feature of unit cell
        if index == 1:
            rand_d = self.rng.integers(low=0, high=2, endpoint=True)
            new_angle = [alpha, beta, gamma]
            shift = get_shift(new_angle[rand_d], c_angle, u_angle, d_angle)
            if shift > 1 or shift < -1:
                LOG.info(
                    "Out of unit cell_angle range, angle: %8.3f", new_angle[rand_d]
                )
                return crystal, rand
            for i in range(15):

                rand = self.rng.random()
                rand_r = rand * 2 - 1 + shift
                R = rand_r * self.moves["u_a"].cutoff
                new_angle[rand_d] += R

                # Check whether the new unit cell is pratically existing or not
                if (
                    ((new_angle[0] + new_angle[1] + new_angle[2]) < 360)
                    and ((new_angle[0] + new_angle[1]) > new_angle[2])
                    and ((new_angle[1] + new_angle[2]) > new_angle[0])
                    and ((new_angle[0] + new_angle[2]) > new_angle[1])
                    and (new_angle[rand_d] > d_angle)
                    and (new_angle[rand_d] < u_angle)
                ):

                    if rand_d == 0:
                        alpha = new_angle[rand_d]
                    elif rand_d == 1:
                        beta = new_angle[rand_d]
                    elif rand_d == 2:
                        gamma = new_angle[rand_d]
                    break

        if index == 2:
            rand_d = 1
            alpha = 90.0
            gamma = 90.0
            shift = get_shift(beta, c_angle, u_angle, d_angle)
            if shift > 1 or shift < -1:
                LOG.info(
                    "Out of unit cell_angle range, angle: %8.3f", beta
                )
                return crystal, rand
            for i in range(15):
                rand = self.rng.random()
                rand_r = rand * 2 - 1 + shift
                R = rand_r * self.moves["u_a"].cutoff
                new_angle = beta + R
                if (new_angle > d_angle and new_angle < u_angle) and (
                    abs(new_angle - 90.0) > epsilon
                ):
                    beta = new_angle
                    break

        if index == 3 or index == 4 or index == 7:
            rand_d = "N"
            R = 0.0
            alpha = 90.0
            beta = 90.0
            gamma = 90.0

        if index == 5:
            rand_d = "A"
            u_angle = min(120.0, u_angle)
            shift = get_shift(alpha, c_angle, u_angle, d_angle)
            if shift > 1 or shift < -1:
                LOG.info(
                    "Out of unit cell_angle range, angle: %8.3f", alpha
                )
                return crystal, rand
            for i in range(15):
                rand = self.rng.random()
                rand_r = rand * 2 - 1 + shift
                R = (self.rng.random() * 2 - 1) * self.moves["u_a"].cutoff
                new_angle = alpha + R
                if (new_angle > d_angle and new_angle < u_angle) and (
                    abs(new_angle - 90.0) > epsilon
                ):
                    alpha = new_angle
                    beta = alpha
                    gamma = alpha
                    break

        if index == 6:
            rand_d = "N"
            R = 0.0
            alpha = 90.0
            beta = 90.0
            gamma = 120.0

        LOG.debug(
            "Unit cell angle change of dimension %i " "by angle %8.3f", rand_d, R
        )
        return Crystal.from_unit_cell_sg_molecules(
            UnitCell.from_lengths_and_angles([a, b, c], [alpha, beta, gamma], unit="degree"),
            deepcopy(crystal.space_group),
            mols,
        ), rand

    def volume(self, crystal):
        """Volume change MC move

        Expand or contract the whole crystal, including unit cell length and
        positions of molecules by a fractional.

        Args:
            crystal: Crystal(core.models), crystal before MC move

        Return:
            crystal: Crystal(core.models), crystal after MC move
            rand: float, random number generated for move displacement
        """
        rand = self.rng.random()
        vol_c = (2 * rand - 1) * self.moves["vol"].cutoff
        # If the move is too big (larger than 0.9 * original volume), set the
        # move to 0.9 * volume
        if abs(vol_c) >= 0.9 * crystal.unit_cell.volume():
            LOG.info("Too big cutoff for volume move, " "set vol_f to 0.9")
            vol_f = 0.9
        else:
            vol_f = vol_c / crystal.unit_cell.volume()

        a, b, c, alpha, beta, gamma = crystal.unit_cell.parameters
        mols = crystal.asym_mols()
        P = pow((1 + vol_f), 1.0 / 3.0) - 1

        # Change the unit cell with fractional coordinate
        for mol in mols:
            cen_frac = crystal.to_fractional(mol.centroid)
            move_frac = cen_frac * P
            #print(move_frac)
            move = crystal.to_cartesian(move_frac)
            mol.positions += move

        a = (1 + P) * a
        b = (1 + P) * b
        c = (1 + P) * c

        LOG.debug("Volume change by propotion %8.5f", P)
        return Crystal.from_unit_cell_sg_molecules(
            UnitCell.from_lengths_and_angles([a, b, c], [alpha, beta, gamma], unit="degree"),
            deepcopy(crystal.space_group),
            mols,
        ), rand

    def reflect(self, crystal, plane="xy"):
        """
        Reflect the molecule across specified image. Currently only x-y
        relfection is implemented(Molecule.relfect). Should only be used in P1
        sampling

        Args:
            crystal: Crystal(core.models), crystal before MC move

        Returns:
            crystal: Crystal(core.models), crystal after MC move
            im: integer, the index of mirrored molecule
        """
        mols = crystal.asym_mols()
        im = self.rng.integers(low=0, high=len(mols)-1, endpoint=True) #if len(mols) > 1 else 1
        
        center = mols[im].centroid
        mols[im].positions -= center
        mols[im].reflect(plane)
        mols[im].positions += center

        LOG.debug("Reflect molecule %i over %s plane", im, plane)
        return Crystal.from_unit_cell_sg_molecules(
            deepcopy(crystal.unit_cell), 
            deepcopy(crystal.space_group),
            mols,
        ), im

    def calc_prob(self):
        """Calculate the probability for each type of move based on the degree
        of freedom
        """
        index = self.index
        num_mol = self.num_mol
        types = []
        for i_move in self.moves:
            types.append(i_move)

        dof = self.get_dof(types, num_mol, index)
        dof_sum = 0
        for i_dof in dof.values():
            dof_sum += i_dof

        prob_tmp = 0
        for i_type in types:
            prob_tmp += float(dof[i_type]) / float(dof_sum)
            self.moves[i_type].probability = prob_tmp

        return

    def get_dof(self, types, num_mol, index):
        """Calculate the degree of freedom for each kind of move in the unit
        cell

        Args:
            types: list of string, contain name of all MC move
            num_mol: int, number of molecules in asymmetric unit
            index: int, range from 1 to 7, indicate the crystal class

        Returns:
            dof: dict, contain degree of freedom fo each kind of move
        """
        dof = {}

        if "tra" in types:
            if self.sg == 1 and num_mol == 1:
                dof["tra"] = 0
            else:
                dof["tra"] = num_mol * 3
        if "rot" in types:
            dof["rot"] = num_mol * 3
        if "vol" in types:
            dof["vol"] = 1
        if "ref" in types:
            dof["ref"] = 1
        if "atoms" in types:
            dof["atoms"] = self.num_atoms * 3

        if index == 1:
            if "u_l" in types:
                dof["u_l"] = 3
            if "u_a" in types:
                dof["u_a"] = 3
        elif index == 2:
            if "u_l" in types:
                dof["u_l"] = 3
            if "u_a" in types:
                dof["u_a"] = 1
        elif index == 3:
            if "u_l" in types:
                dof["u_l"] = 3
            if "u_a" in types:
                dof["u_a"] = 0
        elif index == 4:
            if "u_l" in types:
                dof["u_l"] = 2
            if "u_a" in types:
                dof["u_a"] = 0
        elif index == 5:
            if "u_l" in types:
                dof["u_l"] = 1
            if "u_a" in types:
                dof["u_a"] = 1
        elif index == 6:
            if "u_l" in types:
                dof["u_l"] = 2
            if "u_a" in types:
                dof["u_a"] = 0
        elif index == 7:
            if "u_l" in types:
                dof["u_l"] = 1
            if "u_a" in types:
                dof["u_a"] = 0

        return dof

    def calc_cut(self):
        """Calculating the cutoff according to the number of molecules Z in the
        unit cell
        """
        for type in self.moves:
            if type == "vol":
                self.moves[type].cutoff = self.moves[type].cutoff * self.z
        return

    #def check_overlap_by_convex_hull(self, crystal):
    #    """Sat expansion

    #    Args:
    #        crystal: Crystal(core.models), crystal before MC move

    #    Return:
    #        crystal: Crystal(core.models), crystal after MC move
    #    """
    #    max_volume = 2.5 * box_volume(crystal)
    #    LOG.debug("Sat expansion max volume: %8.3f", max_volume)
    #    crystal.space_group = CrystallographicSpaceGroups.get(int(self.sg))
    #    sat_engine = CrystalExpansionBySAT(
    #        crystal,
    #        cell_cutoff=3,
    #        max_volume=max_volume,
    #        max_retries=50,
    #        use_volume_metric=False,
    #    )
    #    sat_engine._construct_hulls(crystal)
    #    try:
    #        sat_engine.sat_all_overlaps(crystal)
    #        if crystal.lattice.volume > max_volume:
    #            return
    #        else:
    #            return crystal
    #    except:
    #        return


def get_shift(angle, c_angle, u_angle, d_angle):
    """Calculate the shift based on the current angle for unit cell angle
    change. The idea is to make the new angle more likely close to 90.0

    Args:
        angle: float, current angle needed to be changed
        c_angle: float, central angle to be closed to, usually be 90.0
        u_angle: float, upper limit of new angle
        d_angle: float, lower limit of new angle

    Returns:
        shift: float, shifts of the center for random angle displacement
    """
    if angle >= c_angle:
        shift = (c_angle - angle) / (u_angle - c_angle)
    else:
        shift = (angle - c_angle) / (d_angle - c_angle)
    return shift
