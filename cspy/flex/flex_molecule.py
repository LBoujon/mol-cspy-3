import logging
import numpy as np
import json
from cspy.cspy_exceptions import CSPyException
from cspy.util.file_utils import is_file_readable
from cspy.flex.listmath import (
    set_impropertorsion_st_vectors_fast,
    set_dihedral_st_vectors_fast,
    set_angle_st_vectors_fast,
    set_bond_st_vectors_fast,
    list3norm,
    list3normalize,
    list3diff,
    list3angle,
    list3multiply,
    list3cross,
    list3add,
    list3dot,
    list3subtract,
    list3mag,
    quaternion_rotatn,
    rotation_by_q,
)
from collections import OrderedDict
import re
import math
# from cspy.flex.sobolf90 import (
#     i4_sobol,
# )
from cspy.sample.sobol import sobol_vector
import copy
from cspy.flex.internals import (
    Internal,
)
from cspy.db import CspDataStoreFlex
from cspy.util.path import Path
from cspy.apps.dma import (
    generate_combined_name,
    generate_combined_res,
    generate_multipoles,
)
from typing import Any, List, Union, Tuple
import traceback

LOG = logging.getLogger(__name__)


def create_flex_database(filename: str = "flex") -> None:
    """Create a new SQL database to store conformations that can be
    used in flexible-molecule CSP.

    Args:
        filename (str, optional): The name of the new database. Defaults to "flex".
    """
    filename = filename + "_flex.db"
    ds = CspDataStoreFlex.create_and_connect(filename)
    ds.disconnect()


def add_to_flex_database(data: dict, filename: str = "flex") -> None:
    """Add results to the SQL database

    Args:
        data (dict): A dictionary containing data to add to the SQL database
        filename (str, optional): The name of the SQL database to add data to. Defaults to "flex".
    """
    from cspy.db.datastore_writer_flex import DatastoreWriterFlex

    filename = filename + "_flex.db"

    db_writer = DatastoreWriterFlex(
        data,
        filename,
    )

    db_writer.run()


def pirange(x: float) -> float:
    """Function to constrain angles between -pi and +pi

    Args:
        x (float): angle in radians

    Returns:
        float: Adjusted angle in the range -pi to +pi
    """
    if x > math.pi:
        return x - math.pi * 2.0
    elif x < -math.pi:
        return x + math.pi * 2.0
    else:
        return x


def zmatrix_list_to_xyz_list(zmat: List) -> List:
    """Reproduces Gaussian 09 zmatrix to xyz

    Args:
        zmat (List): Z-matrix in list format

    Returns:
        List: XYZ format in list format
    """
    lzmat = len(zmat)
    xyz = []
    if lzmat > 0:
        xyz.append([zmat[0][0], 0.0, 0.0, 0.0])
    if lzmat > 1:
        xyz.append([zmat[1][0], 0.0, 0.0, float(zmat[1][2])])
    if lzmat > 2:
        bnd = float(zmat[2][2])
        ang = math.degrees(float(zmat[2][4]))

        l1 = map(float, xyz[zmat[2][1] - 1][1:4])
        l2 = map(float, xyz[zmat[2][3] - 1][1:4])
        l3 = list3diff(l2, l1)
        l4 = list3normalize(l3)
        l5 = rotation_by_q(l4, ang, [0.0, -1.0, 0.0])
        l6 = list3multiply(l5, bnd)
        l7 = list3add(l1, l6)
        xyz.append([zmat[2][0], l7[0], l7[1], l7[2]])
    if lzmat > 3:
        for c in range(3, lzmat):
            bnd = float(zmat[c][2])
            ang = math.radians(float(zmat[c][4]))
            dih = math.radians(float(zmat[c][6]))

            p1 = map(float, xyz[zmat[c][3] - 1][1:4])
            p2 = map(float, xyz[zmat[c][1] - 1][1:4])
            p3 = list3diff(p2, p1)

            q1 = map(float, xyz[zmat[c][3] - 1][1:4])
            q2 = map(float, xyz[zmat[c][5] - 1][1:4])
            q3 = list3diff(q2, q1)

            n = list3cross(p3, q3)

            l1 = map(float, xyz[zmat[c][1] - 1][1:4])
            l2 = map(float, xyz[zmat[c][3] - 1][1:4])
            l3 = list3diff(l2, l1)
            l4 = list3normalize(l3)
            l5 = rotation_by_q(l4, -ang, n)
            l51 = rotation_by_q(l5, dih, p3)
            l6 = list3multiply(l51, bnd)
            l7 = list3add(map(float, xyz[int(zmat[c][1]) - 1][1:4]), l6)
            xyz.append([zmat[c][0], l7[0], l7[1], l7[2]])
    return xyz


def rot_trans_xyz(xyz_list: List, rotation: List, translation) -> List:
    """Perform rotation and/or translation of xyz

    Args:
        xyz_list (List): List of XYZ coordinates
        rotation (List): Rotation matrix
        translation (List): translation matrix

    Returns:
        List: New coordinates after rotation/translation of xyz
    """
    # Rotate xyz around the centroid of xyz
    # Handles having label in lxyz or not
    s = len(xyz_list[0]) - 3
    if not s in [0, 1]:
        LOG.error(
            "XYZ format incorrect in rot_trans_xyz."
            + " Rows have length "
            + str(len(xyz_list[0]))
            + " s="
            + str(s)
        )
        exit()
    do_rot = any([v != 0.0 for v in rotation])

    if do_rot:
        centre = [0.0, 0.0, 0.0]
        for line in xyz_list:
            centre = list3add(centre, line[s:])
        centre = list3multiply(centre, 1.0 / float(len(xyz_list)))
        centreptrans = list3add(centre, translation)
        xyz_list_rot_trans = []
        for line in xyz_list:
            xyz = list3diff(line[s:], centre)
            xyz = rotation_by_q(xyz, rotation[0], [1.0, 0.0, 0.0])
            xyz = rotation_by_q(xyz, rotation[1], [0.0, 1.0, 0.0])
            xyz = rotation_by_q(xyz, rotation[2], [0.0, 0.0, 1.0])
            xyz = list3add(xyz, centreptrans)
            if s == 1:
                xyz_list_rot_trans.append([line[0]] + xyz)
            else:
                xyz_list_rot_trans.append(xyz)
    else:
        xyz_list_rot_trans = []
        for line in xyz_list:
            xyz = list3add(line[s:], translation)
            if s == 1:
                xyz_list_rot_trans.append([line[0]] + xyz)
            else:
                xyz_list_rot_trans.append(xyz)

    return xyz_list_rot_trans


class FlexAtom:
    """
    Class to define an Atom
    """

    atomic_numbers = {
        "X": 1000,
        "H": 1,
        "He": 2,
        "Li": 3,
        "Be": 4,
        "B": 5,
        "C": 6,
        "N": 7,
        "O": 8,
        "F": 9,
        "Ne": 10,
        "Na": 11,
        "Mg": 12,
        "Al": 13,
        "Si": 14,
        "P": 15,
        "S": 16,
        "Cl": 17,
        "Ar": 18,
        "K": 19,
        "Ca": 20,
        "Sc": 21,
        "Ti": 22,
        "V": 23,
        "Cr": 24,
        "Mn": 25,
        "Fe": 26,
        "Co": 27,
        "Ni": 28,
        "Cu": 29,
        "Zn": 30,
        "Br": 35,
        "Xe": 54,
        "I": 53,
    }
    atomic_labels = {v: k for k, v in atomic_numbers.items()}
    vdw_radii = {
        "X": 1.09,
        "H": 1.09,
        "He": 1.40,
        "Li": 1.82,
        "Be": 2.00,
        "B": 2.00,
        "C": 1.70,
        "N": 1.55,
        "O": 1.52,
        "F": 1.47,
        "Ne": 1.54,
        "Na": 2.27,
        "Mg": 1.73,
        "Al": 2.00,
        "Si": 2.10,
        "P": 1.80,
        "S": 1.80,
        "Cl": 2.0,
        "Ar": 1.88,
        "Br": 1.85,
        "Xe": 2.16,
        "I": 1.98,
    }

    def __init__(self, label: str, coord: List, check=True) -> None:
        """Initiate using label and coordinates

        Args:
            label (str): A string which contains the element
            e.g. H, Cl123
            coord (List): The coordinates of the atom
            check (bool, optional): True will set the atomic number by parsing the label. Defaults to True

        >>> a=Atom('H',[1.0,1.0,1.0])

        """
        self.label = ""
        self.set_label(label, check=check)
        # Atoms position in xyz or zmatrix form
        self.coord_type = None
        self.xyz = [0.0, 0.0, 0.0]  # X,Y,Z
        # bonded_to,distance,angle_to,angle,dihedral_to,dihedral
        self.zmatrix = []
        if len(coord) == 3:  # XYZ definition
            self.coord_type = "XYZ"
            self.xyz = coord
            self.zmatrix = None
        else:
            self.coord_type = "ZMATRIX"
            for i in range(0, min(6, len(coord)), 2):
                self.zmatrix += [coord[i], coord[i + 1]]
            self.xyz = None

        # Extra information that can be determined from label
        self.mass = 0.0
        self.atomic_number = 0
        if check:
            self.set_atomic_number_from_label()
        self.potential_label = ""
        self.attributes = {}
        self.atom_id = None

    def set_label(self, label_or_atomic_number: str, check=True) -> None:
        """Sets up the label and cleaned up label for the atom used in initiating atoms.
        Can also take atomic number as input.

        Args:
            label_or_atomic_number (str): A string containig a label or atomic number for the atom
            check (bool, optional): True will set the atomic number by parsing the label_or_atomic_number variable.
            Defaults to True.

        Raises:
            CSPyException: If label_or_atomic_number is not a str, cannot be converted to
            an int or cannot be identified in the atomic numbers dictionary.

        >>> a=Atom('H',[1.0,1.0,1.0])
        >>> a.label
        'H'
        >>> a=Atom('H123',[1.0,1.0,1.0])
        >>> a.label
        'H123'
        """
        # Try to convert label to integer in case given an atomic number
        try:
            atomic_number = int(label_or_atomic_number)
        except ValueError as exc:
            # If not, double check we have been given a string
            # and use this for the label
            # Later we check if this is acceptable
            label = label_or_atomic_number
            if isinstance(label, str):
                self.label = label
                self.make_clean_label()
                if check and not FlexAtom.atomic_numbers.get(self.clean_label()):
                    er = "Label is not known in Atom.set_label - " + str(label)
                    raise CSPyException(er)
            else:
                er = (
                    "Label in Atom.set_label is neither a string or "
                    + "possible to convert to int - "
                    + str(label)
                )
                raise CSPyException(er)
        else:
            if check:
                self.label = FlexAtom.atomic_labels[atomic_number]
                self.make_clean_label()

    def clean_label(self) -> str:
        """Returns the cleaned label.

        >>> a=Atom('H123',[1.0,1.0,1.0])
        >>> a.clean_label()
        'H'
        >>> a=Atom('CL',[1.0,1.0,1.0])
        >>> a.clean_label()
        'Cl'
        """
        return self.cl

    def make_clean_label(self) -> None:
        """Create the cleaned label and stores it in self.cl.
        Access this through clean_label() function
        Cleaning is done while setting it to save time later

        >>> a=Atom('H123',[1.0,1.0,1.0])
        >>> a.clean_label()
        'H'
        >>> a=Atom('CL',[1.0,1.0,1.0])
        >>> a.clean_label()
        'Cl'
        >>> a=Atom('H',[1.0,1.0,1.0])
        >>> a.clean_label()
        'H'
        """
        # Start with self.label, then apply re.sub s
        # only need first 2 characters
        self.cl = self.label[0:2]
        self.cl = self.cl.replace(" ", "")  # turns " C123_A" to "C123_A"
        self.cl = re.sub("\d+", " ", self.cl)  # turns "C123_A" to "C   _A"
        self.cl = re.sub("[\(\_].*", " ", self.cl)  # turns "C   _A" to C    A"
        self.cl = self.cl.split()[0]  # returns "C"
        self.cl = self.cl.capitalize()

    def set_potential_label(self, p_label: str) -> None:
        """Set the potential label by storing input to self.potential_label

        Args:
            p_label (str): The potential label
        """
        self.potential_label = p_label

    def get_retyped_label(self) -> str:
        """Get the retyped label

        Raises:
            CSPyException: If self.rt has not been set

        Returns:
            str: the retyped label according to the typing rules (hardcoded at the moment)
        """
        try:
            return self.rt
        except AttributeError as exc:
            er = "Retyped labelled has not been defined for atom " + self.label
            raise CSPyException(er)

    def set_retyped_label(self, label: str) -> None:
        """Set the retyped label attribute

        Args:
            label (str): The label to store to this attribute
        """
        self.rt = label

    def VdW_radius(self) -> float:
        """Get the Van der Waals radius of the atom

        Returns:
            float: The Van der Waals radius of Atom

        >>> Atom('H',[1.0,1.0,1.0]).VdW_radius()
        1.09
        >>> Atom('F',[1.0,1.0,1.0]).VdW_radius()
        1.47
        """
        return FlexAtom.vdw_radii[self.clean_label()]

    def set_atomic_number_from_label(self) -> None:
        """Stores atomic number of stored label in self.atomic_number.
        This is called when the atom is initialised

        >>> a=Atom('H',[1.0,1.0,1.0]); a.atomic_number
        1
        >>> a=Atom('F',[1.0,1.0,1.0]); a.atomic_number
        9
        >>> a=Atom('CL123',[1.0,1.0,1.0]); a.atomic_number
        17
        """
        self.atomic_number = FlexAtom.atomic_numbers[self.clean_label()]

    def xyz_string_form(
        self, clean_label: bool = True, retyped_label: bool = False
    ) -> str:
        """Return xyz line for atom. i.e. 'LABEL X Y Z'

        Args:
            clean_label(bool, optional): Whether to clean the stored label
            retyped_label(bool, optional): Whether to get the retyped label

        Raises:
            CSPyException: If the Atom class was not initialised with XYZ coordinates
            then an error will be raised.

        Returns:
            str: A string containing the XYZ format of the atom
        """
        if self.coord_type != "XYZ":
            raise CSPyException("Atom not in XYZ form. Only ZMATRIX available")
        v = tuple(
            self.xyz_list_form(clean_label=clean_label, retyped_label=retyped_label)
        )
        return "%-6s %16.9f %16.9f %16.9f " % v

    def xyz_list_form(
        self, clean_label: bool = True, retyped_label: bool = False
    ) -> List:
        """Return xyz line for atom in list form. i.e. ['LABEL', X, Y, Z]

        Args:
            clean_label(bool, optional): Whether to clean the stored label
            retyped_label(bool, optional): Whether to get the retyped label

        Raises:
            CSPyException: If the Atom class was not initialised with XYZ coordinates
            then an error will be raised.

        Returns:
            List: A list containing the XYZ format of the atom

        >>> a=Atom('H',[1.0,1.0,1.0]); a.xyz_list_form()
        ['H', 1.0, 1.0, 1.0]
        >>> a=Atom('F',[1.0,1.0,1.0]); a.xyz_list_form()
        ['F', 1.0, 1.0, 1.0]
        >>> a=Atom('CL123',[1.0,1.0,1.0]); a.xyz_list_form()
        ['Cl', 1.0, 1.0, 1.0]
        >>> a=Atom('CL123',[1.0,1.0,1.0]); a.xyz_list_form(clean_label=False)
        ['CL123', 1.0, 1.0, 1.0]
        """
        if self.coord_type != "XYZ":
            raise CSPyException("Atom not in XYZ form. Only ZMATRIX available")
        if retyped_label:
            return [self.get_retyped_label()] + self.xyz[0:3]
        elif clean_label:
            return [self.clean_label()] + self.xyz[0:3]
        else:
            return [self.label] + self.xyz[0:3]

    def distance_to(self, atomB: "FlexAtom") -> float:
        """Calculates the distance between two atoms if in XYZ

        Args:
            atomB (Atom): An atom object of the 2nd atom

        Raises:
            CSPyException: If the Atom class was not initialised with XYZ coordinates
            then an error will be raised.

        Returns:
            float: The distance between the two atoms

        >>> a=Atom('H',[1.0,1.0,1.0]);
        >>> b=Atom('H',[10.0,1.0,1.0]);
        >>> a.distance_to(b) # doctest: +NUMBER
        9.0

        >>> a.distance_to(a)
        0.0
        """
        if self.coord_type != "XYZ":
            raise CSPyException("Atom not in XYZ form. Only ZMATRIX available")
        return list3norm(self.xyz, atomB.xyz)

    def distance_closest_atomtype(
        self, target_atom_label: str, molecule: "FlexMolecule"
    ) -> float:
        """Calculate the distance to the closet atom in a molecule with target_atom_label

        Args:
            target_atom_label(str): A string containing the target atom label
            molecule (Molecule): A molecule containing the target atom

        Raises:
            CSPyException: If the Atom class was not initialised with XYZ coordinates
            then an error will be raised.

        Returns:
            float: The distance to the closet atom in molecule with target_atom_label
        """
        if target_atom_label not in "".join([a.label for a in molecule.atoms]):
            er = "No " + str(target_atom_label) + " in that molecule"
            raise CSPyException(er)
        distance = 1000000.0
        for atom in molecule.atoms:
            if target_atom_label == atom.label:
                this_distance = list3mag(list3subtract(atom.xyz, self.xyz))
                distance = min(this_distance, distance)
        return distance

    @staticmethod
    def invert_xyz_coords(
        reflection_plane: str,
        coords: List,
        invert_all: bool = False,
        string_conv: bool = True,
    ) -> List:
        """Inverts atomic coordinates from dictionary about the specified plane

        Args:
            reflection_plane (str): A string containing the reflection plane
            coords (List): A list of atomic coordinates to invert
            invert_all (bool,optional): If true, inverts along x, y and z axes. Defaults to False
            string_conv (bool,optional): If true, converts input coordinates to strings. Defaults to True

        Returns:
            List: A list of coordinates as the type they were initially input as.

        """

        def invert_number(number: Union[str, float, int]) -> Union[str, float, int]:
            """Inverts an input coordinate

            Args:
                number (Union[str, float, int]): An input coordinate that can be in str, float or int type.

            Returns:
                Union[str, float, int]: The inverted coordinate as str, float or int type.
            """
            if isinstance(number, str):
                return ("-" + number).replace("--", "")
            else:
                return -1 * number

        inversion_coord = {"x": 0, "y": 1, "z": 2}
        if not invert_all:
            for c in list(reflection_plane):
                del inversion_coord[c]

        init_type = map(type, coords)
        if string_conv:
            coords = map(str, coords)

        for c, i in inversion_coord.iteritems():
            coords[i] = invert_number(coords[i])

        coords = [init_type[i](coords[i]) for i in range(len(coords))]

        return coords


class FlexMolecule:
    """
    A class specific for Molecules in the flexible-molecule CSP workflow
    """

    bonds_dictionary = {
        ("C", "H"): 1.28,
        ("C", "C"): 1.65,
        ("C", "N"): 1.55,
        ("C", "O"): 1.55,
        ("C", "F"): 1.45,
        ("C", "S"): 2.00,
        ("S", "C"): 2.00,
        ("C", "Cl"): 1.85,
        ("Cl", "C"): 1.85,
        ("C", "Br"): 1.95,
        ("N", "H"): 1.20,
        ("N", "N"): 1.55,
        ("N", "O"): 1.55,
        ("N", "S"): 1.80,
        ("N", "Hg"): 2.80,
        ("O", "H"): 1.3,
        ("O", "C"): 1.55,
        ("O", "O"): 1.70,
        ("O", "S"): 1.60,
        ("B", "F"): 1.45,
        ("B", "C"): 1.65,
        ("I", "Hg"): 2.80,
        ("Br", "Hg"): 2.50,
        ("Br", "C"): 2.00,
        ("S", "S"): 2.5,
        ("H", "H"): 0.85,
        ("C", "I"): 2.2,
        ("S", "Cl"): 1.7,
        ("S", "F"): 1.6,
        ("Cl", "H"): 1.35,
        ("Br", "Br"): 1.95,
        ("Br", "Br"): 2.35,
        ("C", "Br"): 2.05,
        ("Cl", "Br"): 2.20,
        ("F", "Br"): 2.60,
        ("H", "Br"): 1.50,
        ("X", "H"): 0.85,
        ("X", "C"): 1.28,
        ("X", "N"): 1.20,
        ("X", "O"): 1.30,
        ("X", "Cl"): 1.35,
        ("X", "Br"): 1.50,
        ("X", "X"): 0.85,
    }

    def __init__(self) -> None:
        """Initialise an instance of the Molecule class"""
        self.xyz_translation_offset = [0.0, 0.0, 0.0]
        self.xyz_rotation_offset = [0.0, 0.0, 0.0]
        self.atoms = []
        self.parameters = OrderedDict()
        self.coord_type = ""
        self.static = False
        self.static_xyz = None
        self.cart_dof_matrix = None
        self.int_dof_matrix = None
        self._num_int_dof = None
        self._num_atoms = 0
        self._min_max = [[9e9, -9e9], [9e9, -9e9], [9e9, -9e9]]
        self.atom_types = set()
        self.energy = 0.0

        self.static_molecule(False)
        self.fractional_centre = [0.0, 0.0, 0.0]
        self.convex_hull_vertices = None
        self.convex_hull_vertices_id = None
        self.convex_hull_faces = None
        self.convex_hull_edges = None
        self.convex_hull_edge_vecs = None
        self.moments_inertia_eig_vecs = None
        self.moments_inertia_eig_vals = None
        self.box_lengths = None
        self.volume = None
        self.convex_hull_volume = None
        self.name = ""
        self.position_sobol = None
        self.orientation_sobol = None
        self.flex_df_sobol = {}
        self.history = None
        self.bad_keys = {
            "o_bond_keys": None,
            "o_angle_keys": None,
            "o_dihedral_keys": None,
            "o_improper_keys": None,
        }

        self._set_potential_done = False
        self.flexible_normal_modes = []
        self.flexible_internal_coordinates = []

    def flat_atom_coord_list(self) -> List:
        """Flatterns the atom coordinates into a list

        Returns:
            List: A flatterned list containing the atomic coordinates
        """
        return [atom.xyz for atom in self.atoms]

    def flat_atom_vdw_radii_list(self) -> List:
        """Flatterns the atom Van der Waals radii into a list

        Returns:
            List: A flatterned list containing the atom Van der Waals radii
        """
        return [self.atoms[0].vdw_radii[v.clean_label()] for v in self.atoms]

    def num_atoms(self) -> int:
        """Returns the number of atoms in the molecule"""
        return self._num_atoms

    def num_int_dof(self) -> int:
        """Returns the number of degrees of freedom in the molecule"""
        return self._num_int_dof

    def is_static(self) -> bool:
        """Returns whether the molecule is considered static or not"""
        return self.static

    def centroid(self, use_offsets: bool = True) -> List:
        """Get the centroid of the molecule

        Args:
            use_offsets (bool, optional): Whether to use offsets in self.xyz_rotation_offset
            and self.xyz_translation_offset to rotate/translate molecule. Defaults to True.

        Returns:
            List: The centroid coordinates
        """
        xyz = self.xyz_form(use_offsets=use_offsets)
        cent = [0.0, 0.0, 0.0]
        for line in xyz:
            cent = list3add(cent, line[1:4])
        cent = list3multiply(cent, float(float(1) / float(self.num_atoms())))
        return cent

    def rotate_atoms(self, quat) -> None:
        """Rotate the atoms of the molecule

        Args:
            quat (List): A list representing a rotation quaternion
        """
        origin = self.centroid()
        if list3mag(origin) < 0.0001:
            for ia in range(len(self.atoms)):
                self.atoms[ia].xyz = quaternion_rotatn(self.atoms[ia].xyz, quat)
            return
        else:
            self.displace_atoms(list3multiply(origin, -1.0))
            for ia in range(len(self.atoms)):
                self.atoms[ia].xyz = quaternion_rotatn(self.atoms[ia].xyz, quat)
            self.displace_atoms(origin)

    def displace_atoms(self, vector: List) -> None:
        """Displace the atoms of a molecule

        Args:
            vector (List): A list containing the vector to displace atoms along
        """
        for ia in range(0, self.num_atoms()):
            self.atoms[ia].xyz = list3add(self.atoms[ia].xyz, vector)

    def connectivity_matrix(self, forward_only: bool = False) -> List[List]:
        """Get the connectivity matrix of the molecule

        Args:
            forward_only (bool, optional): If true, only record forward connectivity
            (i.e. A-B = 1, B-A = 0). Defaults to False.

        Returns:
            List[List]: The connectivity matrix for the molecule
        """
        from itertools import combinations

        conn = [[0 for _ in range(self.num_atoms())] for _ in range(self.num_atoms())]

        for A, B in combinations(enumerate(self.atoms), 2):
            std = self.bonds_dictionary[(A[1].clean_label(), B[1].clean_label())]
            di = A[1].distance_to(B[1])
            if di < std:
                conn[A[0]][B[0]] = 1
                if forward_only:
                    conn[B[0]][A[0]] = 0
                else:
                    conn[B[0]][A[0]] = 1
            else:
                conn[A[0]][B[0]] = 0
                conn[B[0]][A[0]] = 0
        return conn

    def xyz_form(
        self,
        use_offsets: bool = True,
        clean_labels: bool = False,
        retyped_labels: bool = False,
    ) -> List[List]:
        """Get the atomic positions in XYZ format. If the molecule has been called
        static then just return that for zmatrix atoms it performs the conversion.

        Args:
            use_offsets (bool, optional): Whether to use offsets to rotate/translate atoms using rot_trans_xyz function. Defaults to True
            clean_labels (bool, optional): Whether to clean the labels of each atom. Defaults to False.
            retyped_labels (bool, optional): Whether to use the retyped labels of each atom. Defaults to False.

        Returns:
            List[List]: A list containing the atomic positions in xyz format as a list

        """
        from copy import deepcopy

        if self.static:
            if self.static_xyz:
                return self.static_xyz
        xyz = []
        if self.coord_type == "ZMATRIX":
            xyz = self.zmatrix_to_xyz(use_offsets, clean_labels=clean_labels)
        else:
            for atom in self.atoms:
                if clean_labels:
                    xyz.append(
                        [atom.clean_label(), atom.xyz[0], atom.xyz[1], atom.xyz[2]]
                    )
                elif retyped_labels:
                    xyz.append(
                        [
                            atom.get_retyped_label(),
                            atom.xyz[0],
                            atom.xyz[1],
                            atom.xyz[2],
                        ]
                    )
                else:
                    xyz.append([atom.label, atom.xyz[0], atom.xyz[1], atom.xyz[2]])
            if use_offsets:
                xyz = rot_trans_xyz(
                    xyz, self.xyz_rotation_offset, self.xyz_translation_offset
                )
        if self.static:
            self.static_xyz = deepcopy(xyz)
        return xyz

    def xyz_string_form(
        self,
        use_offsets: bool = True,
        clean_labels: bool = False,
        retyped_labels: bool = False,
    ) -> List[str]:
        """Get the atomic positions in XYZ format as a List of strings.

        Args:
            use_offsets (bool, optional): Whether to use offsets to rotate/translate atoms using rot_trans_xyz function. Defaults to True
            clean_labels (bool, optional): Whether to clean the labels of each atom. Defaults to False.
            retyped_labels (bool, optional): Whether to use the retyped labels of each atom. Defaults to False.

        Returns:
            List[str]: A List of strings containing the atomic positions in xyz format
        """
        xyz_s = []
        xyz = self.xyz_form(
            use_offsets, clean_labels=clean_labels, retyped_labels=retyped_labels
        )
        for line in xyz:
            atom = FlexAtom(line[0], line[1:4], check=not retyped_labels)
            xyz_s.append(atom.xyz_string_form(clean_label=False))
        return xyz_s

    def xyz_string_form_print(
        self,
        use_offsets: bool = True,
        comment: str = "comment line",
        retyped_labels: bool = False,
    ) -> None:
        """Log the atomic positions in XYZ format.

        Args:
            use_offsets (bool, optional): Whether to use offsets to rotate/translate atoms using rot_trans_xyz function. Defaults to True
            comment (str, optional): A string for the XYZ comment line on line 2. Defaults to "comment line".
            retyped_labels (bool, optional): Whether to use the retyped labels of each atom. Defaults to False.
        """
        LOG.info(self.num_atoms())
        LOG.info(comment)
        xyz = self.xyz_string_form(use_offsets, retyped_labels=retyped_labels)
        for line in xyz:
            LOG.info(line)
        return

    def xyz_string_form_info(
        self, use_offsets: bool = True, comment: str = "comment line"
    ) -> None:
        """Log the atomic positions in XYZ format using info level.

        Args:
            use_offsets (bool, optional): Whether to use offsets to rotate/translate atoms using rot_trans_xyz function. Defaults to True
            comment (str, optional): A string for the XYZ comment line on line 2. Defaults to "comment line".
        """
        LOG.info(str(self.num_atoms()))
        LOG.info(comment)
        xyz = self.xyz_string_form(use_offsets)
        for line in xyz:
            LOG.info(line)
        return

    def xyz_string_form_debug(
        self, use_offsets: bool = True, comment: str = "comment line"
    ) -> None:
        """Log the atomic positions in XYZ format using info level.

        Args:
            use_offsets (bool, optional): Whether to use offsets to rotate/translate atoms using rot_trans_xyz function. Defaults to True
            comment (str, optional): A string for the XYZ comment line on line 2. Defaults to "comment line".
        """
        LOG.debug(str(self.num_atoms()))
        LOG.debug(comment)
        xyz = self.xyz_string_form(use_offsets)
        for line in xyz:
            LOG.debug(line)
        return

    def xyz_string_to_database(self, comment: str = "Comment line") -> str:
        """Prepares XYZ coordinate string for storage in SQL database

        Args:
            comment (str, optional): String to place on the comment line. Defaults to "Comment line".

        Returns:
            str: XYZ coordinates as a single string
        """
        lines = [str(self.num_atoms()), comment]
        for line in self.xyz_string_form(
            use_offsets=True, clean_labels=False, retyped_labels=False
        ):
            lines.append(line)
        return "\n".join(lines)

    def xyz_print_to_file(
        self,
        filename: str,
        use_offsets: bool = True,
        comment: str = "comment line",
        retyped_labels: bool = False,
    ) -> None:
        """Writes the xyz coordinates to a file

        Args:
            filename (str): Name of the file to write to
            use_offsets (bool, optional): Whether to use offsets to rotate/translate atoms using rot_trans_xyz function. Defaults to True
            comment (str, optional): A string for the XYZ comment line on line 2. Defaults to "comment line".
            retyped_labels (bool, optional): Whether to use the retyped labels of each atom. Defaults to False.
        """
        with open(filename, "w+") as f:
            f.write(str(self.num_atoms()) + "\n")
            f.write(comment + "\n")
            for line in self.xyz_string_form(
                use_offsets, retyped_labels=retyped_labels
            ):
                f.write(line + "\n")

    def add_atom(self, atom: "FlexAtom") -> None:
        """Add an atom to the molecule

        Args:
            atom (Atom): An atom object representing the atom to add to the molecule

        Raises:
            CSPyException: Raised if the coordinate type of the atom being added is different
            to the coordinate type of the atoms in the molecule.
        """
        from itertools import permutations

        if not self.atoms:
            self.coord_type = atom.coord_type
        else:
            if atom.coord_type != self.coord_type:
                er = "Adding atom of different coordinate type."
                er += "Should be" + str(self.coord_type)
                LOG.error(er)
                raise CSPyException(er)
        self.atoms.append(atom)
        l = len(self.atoms)
        if l == 1:
            self._num_int_dof = 0
        if l == 2:
            self._num_int_dof = 1
        if l == 3:
            self._num_int_dof = 3
        if l >= 4:
            self._num_int_dof = 3 * l - 6
        self._num_atoms = l
        if atom.coord_type == "XYZ":
            self._min_max[0][0] = min(self._min_max[0][0], atom.xyz[0])
            self._min_max[0][1] = max(self._min_max[0][1], atom.xyz[0])
            self._min_max[1][0] = min(self._min_max[1][0], atom.xyz[1])
            self._min_max[1][1] = max(self._min_max[1][1], atom.xyz[1])
            self._min_max[2][0] = min(self._min_max[2][0], atom.xyz[2])
            self._min_max[2][1] = max(self._min_max[2][1], atom.xyz[2])
        self.atom_types.add(atom.clean_label())
        for a, b in permutations(self.atom_types, 2):
            ab = (a, b)
            self.bonds_dictionary[ab] = FlexMolecule.bonds_dictionary.get(ab)
            if self.bonds_dictionary[ab] is None:
                std_bl = (FlexAtom.vdw_radii[a] + FlexAtom.vdw_radii[b]) * 0.5
                self.bonds_dictionary[ab] = std_bl
        acl = atom.clean_label()
        tu = (acl, acl)
        self.bonds_dictionary[tu] = FlexMolecule.bonds_dictionary.get(tu)
        if self.bonds_dictionary[tu] is None:
            self.bonds_dictionary[tu] = (FlexAtom.vdw_radii[acl] * 2.0) * 0.5

    def add_atom_from_str(self, atom_str: str) -> None:
        """Add an atom just from a string XYZ file
        e.g. 'H 0.111 1.2222 2.3333'

        Args:
            atom_str (str): A string containing the atomic coordinates in XYZ format.

        Raises:
            CSPyException: Raised if the string does not containing four items corresponding to
            a symbol, x coordinate, y coordinate and z coordinate.
        """
        str_split = atom_str.lstrip().split()
        if len(str_split) != 4:
            LOG.info("Problem line:", atom_str)
            er = "add_atom_from_str was given a str that did not contain"
            er += "four items split by spaces"
            raise CSPyException(er)
        else:
            a = FlexAtom(str_split[0], tuple(map(float, str_split[1:4])))
            self.add_atom(a)

    def static_molecule(self, static: bool = True) -> None:
        """Set whether the molecule is static

        Args:
            static (bool, optional): Whether the molecule is static or not. Defaults to True.
        """
        self.static = static
        if not self.static:
            self.static_xyz = None

    def set_translation_offset(self, offset: List) -> None:
        """Set the vector used for translational offset

        Args:
            offset (List): A translational vector represented as a list
        """
        if len(offset) != 3:
            LOG.error("wrong offset in set_translation_offset")
            exit()
        self.xyz_translation_offset = offset

    def set_rotation_offset(self, offset: List) -> None:
        """Set the vector used for rotational offset

        Args:
            offset (List): A rotational vector represented as a list
        """
        if len(offset) != 3:
            LOG.error("wrong offset in set_rotation_offset")
            exit()
        self.xyz_rotation_offset = offset
        return

    def connected_keys(
        self,
        level: int,
        sort: bool = True,
        forward_only: bool = True,
        linear_thr: float = 181.0 * math.pi / 180.0,
    ) -> List[Tuple]:
        """Function used to return the atoms involved in forming bonds, angles, dihedrals, improper torsion.

        Args:
            level (int): 1 = Get the atoms involved in forming bonds.
                         2 = Get the atoms involved in forming angles.
                         3 = Get the atoms involved in forming dihedrals.
                         4 = Get the atoms involved in forming improper torsion.
            sort (bool, optional): Whether to sort. Defaults to True.
            forward_only (bool, optional): Whether to build the connectivity matrix only considering bonds in one
            direction (i.e. A-B = 1, B-A = 0). Defaults to True.
            linear_thr (float, optional): The threshold for considering an angle as linear. Defaults to 181.0*math.pi/180.0.

        Raises:
            CSPyException: Raised if any of the involved atoms are connected to more than 3 other atoms

        Returns:
            List[Tuple]: A list of tuples with each tuple containing the atoms involved in forming the bond, angle, dihedral or improper torsion.
        """
        bkeys = []
        akeys = []
        dkeys = []
        itkeys = []
        if self.num_atoms() == 0:
            return []

        conn = self.connectivity_matrix(forward_only=forward_only)

        from scipy.sparse.csgraph import csgraph_from_dense
        from scipy.sparse.csgraph import shortest_path

        Gmask = np.ma.masked_invalid(np.array(conn))
        G = csgraph_from_dense(Gmask, null_value=0)
        dist_matrix, predecessors = shortest_path(G, return_predecessors=True)
        if level == 1:
            for i in range(0, self.num_atoms() - 1, 1):
                for y in np.where(dist_matrix[i][i + 1 :] == 1)[0]:
                    x = y + i + 1
                    bkeys.append((i, x))
            return bkeys
        if level == 2:
            for i in range(0, self.num_atoms() - 1, 1):
                for y in np.where(dist_matrix[i][i + 1 :] == 2)[0]:
                    x = y + i + 1
                    akeys.append((i, predecessors[i][x], x))
            return akeys
        if level == 3:
            for i in range(0, self.num_atoms() - 1, 1):
                for y in np.where(dist_matrix[i][i + 1 :] == 3)[0]:
                    x = y + i + 1
                    if (
                        self.angle_value(
                            i, predecessors[i][predecessors[i][x]], predecessors[i][x]
                        )
                        < linear_thr
                        and self.angle_value(
                            predecessors[i][predecessors[i][x]], predecessors[i][x], x
                        )
                        < linear_thr
                    ):
                        dkeys.append(
                            (
                                i,
                                predecessors[i][predecessors[i][x]],
                                predecessors[i][x],
                                x,
                            )
                        )
            return dkeys
        if level == 4:
            # return itkeys
            for i in range(0, self.num_atoms() - 1, 1):
                bonds = np.where(dist_matrix[i][:] == 1)[0]
                if len(bonds) == 3:
                    itkeys.append((i, bonds[0], bonds[1], bonds[2]))
                    itkeys.append((i, bonds[1], bonds[2], bonds[0]))
                    itkeys.append((i, bonds[2], bonds[0], bonds[1]))
                elif len(bonds) > 3:
                    raise CSPyException(
                        "Cant handle atoms connected to more than 3 other atoms"
                    )
            return itkeys

    def bond_keys(self, sort: bool = True, forward_only: bool = True) -> List[Tuple]:
        """Retrieve the atoms involved in forming bonds

        Args:
            sort (bool, optional): Whether to sort the output. Defaults to True.
            forward_only (bool, optional): Whether to build the connectivity matrix only considering bonds in one
            direction. Defaults to True.

        Returns:
            List[Tuple]: A list of tuples with each tuple containing the atoms involved in forming bonds
        """
        return self.connected_keys(level=1, sort=sort, forward_only=forward_only)

    def angle_keys(self, forward_only: bool = False) -> List[Tuple]:
        """Retrieve the atoms involved in forming angles

        Args:
            forward_only (bool, optional): Whether to build the connectivity matrix only considering bonds in one
            direction. Defaults to False.

        Returns:
            List[Tuple]: A list of tuples with each tuple containing the atoms involved in forming angles
        """
        return self.connected_keys(level=2, forward_only=forward_only)

    def dihedral_keys(self, linear_thr: float = 181.0 * math.pi / 180.0) -> List[Tuple]:
        """Retrieve the atoms involved in forming dihedrals

        Args:
            linear_thr (float, optional): The threshold for considering an angle as linear. Defaults to 181.0*math.pi/180.0.

        Returns:
            List[Tuple]: A list of tuples with each tuple containing the atoms involved in forming dihedrals
        """
        return self.connected_keys(level=3, forward_only=False, linear_thr=linear_thr)

    def improper_keys(self) -> List[Tuple]:
        """Retrieve the atoms involved in forming improper torsion

        Returns:
            List[Tuple]: A list of tuples with each tuple containing the atoms involved in forming bonds
        """
        return self.connected_keys(level=4, forward_only=False)

    def set_bond_st_vectors(
        self, a: int, b: int, Bprim: np.ndarray, internal_index: int
    ) -> None:
        """Set the bond length elements involving atoms A and B in the primitive wilson B matrix

        Args:
            a (int): Index of atom A in the molecule
            b (int): Index of atom B in the molecule
            Bprim (np.ndarray): The primitive Wilson B matrix
            internal_index (int): Index of the internal coordinate to modify
        """
        axyz = self.atoms[a].xyz
        bxyz = self.atoms[b].xyz
        set_bond_st_vectors_fast(axyz, bxyz, a, b, Bprim, internal_index)

    def set_angle_st_vectors(
        self, a: int, b: int, c: int, Bprim: np.ndarray, internal_index: int
    ) -> None:
        """Set the bond angle elements involving atoms A, B and C in the primitive wilson B matrix

        Args:
            a (int): Index of atom A in the molecule
            b (int): Index of atom B in the molecule
            c (int): Index of atom C in the molecule
            Bprim (np.ndarray): The primitive Wilson B matrix
            internal_index (int): Index of the internal coordinate to modify
        """
        axyz = self.atoms[a].xyz
        bxyz = self.atoms[b].xyz
        cxyz = self.atoms[c].xyz
        set_angle_st_vectors_fast(
            axyz, cxyz, bxyz, a, b, c, Bprim, internal_index
        )  # This uses the deprecated function currently

    def set_dihedral_st_vectors(
        self,
        a: int,
        b: int,
        c: int,
        d: int,
        Bprim: np.ndarray,
        q: np.ndarray,
        internal_index: int,
    ) -> None:
        """Set the dihedral angle elements involving atoms A, B, C and D in the primitive wilson B matrix

        Args:
            a (int): Index of atom A in the molecule
            b (int): Index of atom B in the molecule
            c (int): Index of atom C in the molecule
            d (int): Index of atom D in the molecule
            Bprim (np.ndarray): The primitive Wilson B matrix
            q (np.ndarray): An array of internal coordinates
            internal_index (int): Index of the internal coordinate to modify
        """
        axyz = self.atoms[a].xyz
        bxyz = self.atoms[b].xyz
        cxyz = self.atoms[c].xyz
        dxyz = self.atoms[d].xyz
        set_dihedral_st_vectors_fast(
            axyz, bxyz, cxyz, dxyz, a, b, c, d, Bprim, q, internal_index
        )  # This uses the deprecated function currently

    def set_impropertorsion_st_vectors(
        self,
        a: int,
        b: int,
        c: int,
        d: int,
        Bprim: np.ndarray,
        q: np.ndarray,
        internal_index: int,
    ) -> None:
        """Set the improper torsion angle elements involving atoms A, B, C and D in the primitive wilson B matrix

        Args:
            a (int): Index of atom A in the molecule
            b (int): Index of atom B in the molecule
            c (int): Index of atom C in the molecule
            d (int): Index of atom D in the molecule
            Bprim (np.ndarray): The primitive Wilson B matrix
            q (np.ndarray): An array of internal coordinates
            internal_index (int): Index of the internal coordinate to modify
        """
        axyz = self.atoms[a].xyz
        bxyz = self.atoms[b].xyz
        cxyz = self.atoms[c].xyz
        dxyz = self.atoms[d].xyz
        set_impropertorsion_st_vectors_fast(
            bxyz, cxyz, dxyz, axyz, b, c, d, a, Bprim, q, internal_index
        )  # This uses the deprecated function currently

    def set_moments_inertia_eig_vals(self, ev):
        self.moments_inertia_eig_vals = ev
        return self.moments_inertia_eig_vals

    def set_moments_inertia_eig_vecs(self, ev):
        self.moments_inertia_eig_vecs = ev
        return self.moments_inertia_eig_vecs

    def calculate_moments_inertia(
        self, 
        invert_left_handed=False,
        use_last_atom=False):

        ''' Sets ordered right handed inertial vectors
        smallest value first, and farthest atom from centre
        (ambiguous for symmetric mol??)
        in +ve quadrant of 1st two axes
        If invert_left_handed- use new convention, wherein all axes inverted
        if left handed (probably better)
        '''
        import numpy
        Imat = numpy.zeros(shape=(3, 3))
        coords = [list3subtract(atom.xyz, self.centroid())
                  for atom in self.atoms]
        for i in range(0, len(coords)):
            Imat[0, 0] = Imat[0, 0] + coords[i][1] ** 2 + coords[i][2] ** 2
            Imat[1, 0] = Imat[1, 0] - coords[i][0] * coords[i][1]
            Imat[2, 0] = Imat[2, 0] - coords[i][0] * coords[i][2]
            Imat[0, 1] = Imat[0, 1] - coords[i][0] * coords[i][1]
            Imat[1, 1] = Imat[1, 1] + coords[i][0] ** 2 + coords[i][2] ** 2
            Imat[2, 1] = Imat[2, 1] - coords[i][2] * coords[i][1]
            Imat[0, 2] = Imat[0, 2] - coords[i][0] * coords[i][2]
            Imat[1, 2] = Imat[1, 2] - coords[i][2] * coords[i][1]
            Imat[2, 2] = Imat[2, 2] + coords[i][0] ** 2 + coords[i][1] ** 2
        # column eig_vecs[ijk , p] is vector with eigenvalue eigs[p]
        eigs, eig_vecs = numpy.linalg.eig(Imat)

        # Dealing with point-molecules or linear ones -- poor hack
        if eigs[1] == eigs[0] or eigs[2] == eigs[0] or eigs[2] == eigs[1]:
            self.set_moments_inertia_eig_vecs(
                numpy.array(eig_vecs).transpose())
            self.set_moments_inertia_eig_vals(numpy.array(eigs))
        else:
            # TODO Old way of doing the sort
            self.set_moments_inertia_eig_vecs(numpy.array([x for (y, x) in
                                                           sorted(zip(eigs,
                                                                      numpy.array(
                                                                          eig_vecs).transpose()))]))  # [::-1]))
            self.set_moments_inertia_eig_vals(numpy.array(sorted(eigs)))

        # TODO need some comments on what happens here
        if use_last_atom or False:
            #            print coords[-1], 'last coord'
            if list3dot(coords[-1], self.moments_inertia_eig_vecs[2]) < 0.0:
                self.set_moments_inertia_eig_vecs(numpy.array(
                    [self.moments_inertia_eig_vecs[0],
                     self.moments_inertia_eig_vecs[1],
                     list3multiply(self.moments_inertia_eig_vecs[2], -1.0)]))
            if list3dot(coords[-1], self.moments_inertia_eig_vecs[0]) < 0.0:
                self.set_moments_inertia_eig_vecs(numpy.array(
                    [list3multiply(self.moments_inertia_eig_vecs[0], -1.0),
                     self.moments_inertia_eig_vecs[1],
                     self.moments_inertia_eig_vecs[2]]))
        else:
            if list3dot(sorted(coords, key=lambda x: abs(
                    list3dot(x, self.moments_inertia_eig_vecs[2])))[-1],
                        self.moments_inertia_eig_vecs[2]) < 0.0:
                self.set_moments_inertia_eig_vecs(numpy.array(
                    [self.moments_inertia_eig_vecs[0],
                     self.moments_inertia_eig_vecs[1],
                     list3multiply(self.moments_inertia_eig_vecs[2], -1.0)]))
            if list3dot(sorted(coords, key=lambda x: abs(
                    list3dot(x, self.moments_inertia_eig_vecs[0])))[-1],
                        self.moments_inertia_eig_vecs[0]) < 0.0:
                self.set_moments_inertia_eig_vecs(numpy.array(
                    [list3multiply(self.moments_inertia_eig_vecs[0], -1.0),
                     self.moments_inertia_eig_vecs[1],
                     self.moments_inertia_eig_vecs[2]]))

        if list3dot(list3cross(self.moments_inertia_eig_vecs[0],
                               self.moments_inertia_eig_vecs[1]),
                    self.moments_inertia_eig_vecs[2]) < 0.0:
            self.set_moments_inertia_eig_vecs(numpy.array(
                [self.moments_inertia_eig_vecs[0],
                 list3multiply(self.moments_inertia_eig_vecs[1], -1.0),
                 self.moments_inertia_eig_vecs[2]]))
        return

    def set_box_lengths(self):
        ''' Set and Return the box lengths along the moments of inertia '''

        if not hasattr(self, "moments_inertia_eig_vecs"):
            debug = "Calculating moments of inertia in set_box_lengths"
          #  molecule_logger.debug(debug)
            LOG.debug(debug)
            self.calculate_moments_inertia()
        self.box_lengths = []
        com = self.centroid()
        for axis in self.moments_inertia_eig_vecs:
            bl_max = list3dot(list3subtract(self.atoms[0].xyz, com),
                              axis) / list3mag(axis)
            bl_min = list3dot(list3subtract(self.atoms[0].xyz, com),
                              axis) / list3mag(axis)
            for atom in self.atoms:
                temp_pt = list3dot(list3subtract(atom.xyz, com),
                                   axis) / list3mag(axis)
                bl_max = max(bl_max, temp_pt + atom.VdW_radius())
                bl_min = min(bl_min, temp_pt - atom.VdW_radius())
            self.box_lengths.append(bl_max - bl_min)
        return self.box_lengths
    
    def bond_value(self, a: int, b: int) -> float:
        """Get the distance between two atoms in the molecule

        Args:
            a (int): Index of first atom
            b (int): Index of the second atom

        Returns:
            float: The distance between the two atoms
        """
        value = self.atoms[a].distance_to(self.atoms[b])
        return value

    def angle_value(self, a: int, b: int, c: int) -> float:
        """Get the angle between three atoms in the molecule

        Args:
            a (int): Index of first atom
            b (int): Index of the second atom
            c (int): Index of the third atom

        Returns:
            float: The angle between the three atoms
        """
        a1 = self.atoms[a].xyz
        a2 = self.atoms[b].xyz
        a3 = self.atoms[c].xyz
        value = list3angle(a1, a2, a3)
        return value

    def improper_value(self, a: int, b: int, c: int, d: int) -> float:
        """Get the improper torsion angle between four atoms in the molecule

        Args:
            a (int): Index of first atom
            b (int): Index of the second atom
            c (int): Index of the third atom
            d (int): Index of the fourth atom

        Returns:
            float: The improper torsion angle between the four atoms
        """
        a1 = self.atoms[a].xyz
        a2 = self.atoms[b].xyz
        a3 = self.atoms[c].xyz
        a4 = self.atoms[d].xyz

        from listmathfast import (
            improper_value,
        )

        return improper_value(a2, a3, a4, a1)

    def dihedral_value(self, a: int, b: int, c: int, d: int) -> float:
        """Get the dihedral angle between four atoms in the molecule

        Args:
            a (int): Index of first atom
            b (int): Index of the second atom
            c (int): Index of the third atom
            d (int): Index of the fourth atom

        Returns:
            float: The dihedral angle between the four atoms
        """
        a1 = self.atoms[a].xyz
        a2 = self.atoms[b].xyz
        a3 = self.atoms[c].xyz
        a4 = self.atoms[d].xyz

        # project atomic positions onto a plane with a normal in 2-3 direction
        # and passes through [0,0,0]
        p_inplane = [0.0, 0.0, 0.0]
        e32 = list3normalize(list3diff(a2, a3))
        n_toplane = e32

        # Do projections
        a1_proj = list3diff(
            a1, list3multiply(n_toplane, list3dot(list3diff(a1, p_inplane), n_toplane))
        )
        a2_proj = list3diff(
            a2, list3multiply(n_toplane, list3dot(list3diff(a2, p_inplane), n_toplane))
        )
        a4_proj = list3diff(
            a4, list3multiply(n_toplane, list3dot(list3diff(a4, p_inplane), n_toplane))
        )

        # shift positions so 1 and 4 become vectors point out from [0,0,0]
        a1_projb = list3diff(a1_proj, a2_proj)
        a4_projb = list3diff(a4_proj, a2_proj)

        # Need to flatten plane so its normal is in [0,0,1] direction

        # Rotate around z-axis so normal has no component in y direction
        ang = -math.atan2(n_toplane[1], n_toplane[0])
        a1_projc = rotation_by_q(a1_projb, ang, [0, 0, 1])
        a4_projc = rotation_by_q(a4_projb, ang, [0, 0, 1])
        n_toplane_b = rotation_by_q(n_toplane, ang, [0, 0, 1])

        # Rotate around y-axis so formal points in z direction
        ang = -math.atan2(n_toplane_b[0], n_toplane_b[2])
        a1_projd = rotation_by_q(a1_projc, ang, [0, 1, 0])
        a4_projd = rotation_by_q(a4_projc, ang, [0, 1, 0])

        # Find angle to atom 1 from x-direction
        ang1 = math.atan2(a1_projd[1], a1_projd[0])

        # Find angle to rotated atom 4
        ang2 = math.atan2(a4_projd[1], a4_projd[0])

        # Final directed dihedral angle
        value = pirange(ang2 - ang1)
        return value

    def dof_value(self, x: Union[List, str], radians: bool = True) -> float:
        """Calculate the value of an internal coordinate corresponding to a set of specified atoms.
        e.g. bond length, angle, dihedral angle, e.t.c

        Args:
            x (Union[List, str]): A list or string containing the set of atoms to sample over.
            radians (bool, optional): Whether to return the value in radians (True) or degrees (False). Defaults to True.

        Returns:
            float: The value corresponding to the selected degree of freedom (bond, angle, e.t.c)
        """
        if isinstance(x, str):
            x = list(map(int, x.split("_")))
        if len(x) == 2:
            return self.bond_value(x[0], x[1])
        if len(x) == 3:
            v = self.angle_value(x[0], x[1], x[2])
        if len(x) == 4:
            v = self.dihedral_value(x[0], x[1], x[2], x[3])
        if radians:
            return v
        else:
            return 180.0 * v / math.pi

    def get_bad_keys(
        self,
        bond_keys: bool = False,
        angle_keys: bool = False,
        dihedral_keys: bool = False,
        improper_keys: bool = False,
        linear_thr: float = 181.0 * math.pi / 180.0,
    ):
        """Function to store the atoms involved in specified internal coordinate as keys in
        the bonds, angles, dihedrals and torsions dictionary.

        Args:
            bond_keys (bool, optional): Whether the molecule's bonds have been specified already. Defaults to False.
            angle_keys (bool, optional): Whether the molecule's angles have been specified already. Defaults to False.
            dihedral_keys (bool, optional): Whether the molecule's dihedrals have been specified already. Defaults to False.
            improper_keys (bool, optional): Whether the molecule's improper torsions have been specified already. Defaults to False.
            linear_thr (float, optional): The threshold for considering an angle as linear. Defaults to 181.0*math.pi/180.0.

        Returns:
            dict: A dictionary containing the atoms involved in forming bonds, angles, dihedrals, e.t.c
        """
        if not bond_keys:
            self.bad_keys["o_bond_keys"] = self.bond_keys()
        if not angle_keys:
            self.bad_keys["o_angle_keys"] = self.angle_keys()
        if not dihedral_keys:
            self.bad_keys["o_dihedral_keys"] = self.dihedral_keys(linear_thr=linear_thr)
        if not improper_keys:
            pass
            # self.bad_keys["o_improper_keys"] = self.improper_keys()
        return self.bad_keys

    def apply_mode_addition_in_internals(
        self,
        factors: List[float],
        internals_list: Union[None, List[str]] = None,
        nloops: int = 100,
        recompute_K: bool = False,
        dQnormThr: float = 1.0e-8,
        add_factors: bool = True,
        strict_dihedrals: bool = False,
        recompute_K_thr: float = 20.0,
        recompute_K_loop: int = 5,
    ) -> None:
        """Displaces a Molecule along specified internal coordinates by a set of factors

        Args:
            factors (List[float]): A list of floats describing the amount to change the internal coordinates specified by internals_list.
            internals_list (Union[None, List[str]], optional): A list of strings specifying the internal coordinates to use, e.g. ['0_1', '0_1_2', '0_1_2_3']. Defaults to None.
            nloops (int, optional): Number of loops to converge redundants. Defaults to 100.
            recompute_K (bool, optional): Whether to recompute K on each loop. Defaults to False.
            dQnormThr (float, optional): A convergence criteria for the overall change to the internal coordinates. Defaults to 1.0e-8.
            add_factors (bool, optional): Whether to add the factors on each iteration/loop. Defaults to True.
            strict_dihedrals (bool, optional): If True, will only distort atoms in the dihedral. If False, will move all atoms about the central axis of the dihedral (see mol-CSPy documentation for more info). Defaults to False.
            recompute_K_thr (float, optional): A threshold for determining if K is diverging and requires recomputing. Defaults to 20.0.
            recompute_K_loop (int, optional): The number of loops before recomputing K. Defaults to 5.

        Raises:
            CSPyException: Raised if there is a failure to converge before reaching nloops.
        """
        from scipy import linalg as la
        import time

        xyz_new = self.xyz_form(clean_labels=True)
        LOG.debug("Find all primitive internal co-ordinates")
        # Create index of bonds, angles and dihedrals at start
        if not all(
            [
                self.bad_keys["o_bond_keys"],
                self.bad_keys["o_angle_keys"],
                self.bad_keys["o_dihedral_keys"],
            ]
        ):
            bad_keys = self.get_bad_keys()
        o_bond_keys = self.bad_keys["o_bond_keys"]
        o_angle_keys = self.bad_keys["o_angle_keys"]
        o_dihedral_keys = self.bad_keys["o_dihedral_keys"]
        all_keys = o_bond_keys + o_angle_keys + o_dihedral_keys
        sz = (
            sum(map(len, [o_bond_keys, o_angle_keys, o_dihedral_keys])),
            self.num_atoms() * 3,
        )
        Bprim = np.zeros(shape=sz)
        diverging_K = False
        force_convergence_point = nloops - 20

        num_internals = len(o_bond_keys) + len(o_angle_keys) + len(o_dihedral_keys)

        q = np.empty(shape=[num_internals])

        for loop in range(nloops):
            internal_index = 0
            start = time.time()
            for n_a, n_b in o_bond_keys:
                self.set_bond_st_vectors(n_a, n_b, Bprim, internal_index)
                q[internal_index] = self.bond_value(n_a, n_b)
                internal_index += 1

            for n_a, n_b, n_c in o_angle_keys:
                self.set_angle_st_vectors(n_a, n_b, n_c, Bprim, internal_index)
                q[internal_index] = self.angle_value(n_a, n_b, n_c)
                internal_index += 1

            for n_a, n_b, n_c, n_d in o_dihedral_keys:
                self.set_dihedral_st_vectors(
                    n_a, n_b, n_c, n_d, Bprim, q, internal_index
                )
                internal_index += 1

            if loop == 0 or recompute_K or diverging_K:
                G = np.dot(Bprim, Bprim.transpose())
                w, v = la.eigh(G)
                vt = v.transpose()
                n = 0
                K = []
                L = []
                LOG.debug("Eigen values and vectors of G")
                for i in range(num_internals):
                    if w[i] > 1e-6:
                        n += 1
                        K.append(vt[i].real.tolist())
                        L.append(w[i])
                    else:
                        # Found a zero eigenvalue
                        pass
                LOG.debug("Found " + str(n) + " non-zero eigenvalues")
                K = np.array(K)

            # Create the Wilson B matrix for the non-redundant internals
            B = np.dot(K, Bprim)
            B_inv = np.dot(np.linalg.inv(np.dot(B, B.transpose())), B)
            # Get cartesian displacement vector
            if loop == 0:
                X = np.zeros(self.num_atoms() * 3)
                LOG.debug("Mode list :  " + str(internals_list))
                factors_vec = np.zeros((self.num_atoms() * 3 - 6,))
                # Set original primitive internals
                q0 = q
                # Target primitive internals
                qT = q.copy()
                internals_list = [
                    [tuple(map(int, k.split("_"))) for k in i.split(":")]
                    for i in internals_list
                ]
                for i in range(0, len(o_bond_keys)):
                    for f_index, int_set in enumerate(internals_list):
                        for internal in int_set:
                            if len(internal) != 2:
                                continue
                            if all_keys[i] == internal:
                                if not add_factors:
                                    qT[i] = 0.0
                                qT[i] = qT[i] + factors[f_index]
                            elif all_keys[i] == tuple(reversed(internal)):
                                if not add_factors:
                                    qT[i] = 0.0
                                qT[i] = qT[i] + factors[f_index]

                for i in range(len(o_bond_keys), len(o_bond_keys) + len(o_angle_keys)):
                    for f_index, int_set in enumerate(internals_list):
                        for internal in int_set:
                            if len(internal) != 3:
                                continue
                            # print internal, all_keys[i]
                            if all_keys[i] == internal:
                                # print 'Match'
                                if not add_factors:
                                    qT[i] = 0.0
                                qT[i] = qT[i] + factors[f_index]
                            elif all_keys[i] == tuple(reversed(internal)):
                                # print 'Match Reverse'
                                if not add_factors:
                                    qT[i] = 0.0
                                qT[i] = qT[i] + factors[f_index]

                if strict_dihedrals:
                    kmn = 0
                    kmx = 4
                else:
                    kmn = 1
                    kmx = 3
                for i in range(
                    len(o_bond_keys) + len(o_angle_keys),
                    len(o_bond_keys) + len(o_angle_keys) + len(o_dihedral_keys),
                ):
                    for f_index, int_set in enumerate(internals_list):
                        for internal in int_set:
                            if len(internal) != 4:
                                continue
                            if all_keys[i][kmn:kmx] == internal[kmn:kmx]:
                                if not add_factors:
                                    qT[i] = 0.0
                                qT[i] = qT[i] + factors[f_index]
                            elif (
                                all_keys[i][kmn:kmx]
                                == tuple(reversed(internal))[kmn:kmx]
                            ):
                                if not add_factors:
                                    qT[i] = 0.0
                                qT[i] = qT[i] + factors[f_index]

                dq = qT - q
                dQ = np.dot(K, dq)
            else:
                # New dQ will be how far from target primitive (dT)
                dq = qT - q
                for i in range(
                    len(o_bond_keys) + len(o_angle_keys),
                    len(o_bond_keys) + len(o_angle_keys) + len(o_dihedral_keys),
                ):
                    dq[i] = pirange(dq[i])
                dQ = np.dot(K, dq)
            # Check the magnitude of change to see if it will make a change to
            # the molecule and exit. WARNING this treats all values equally
            dQnorm = np.linalg.norm(dQ)
            if recompute_K_thr and loop > recompute_K_loop:
                diverging_K = recompute_K_thr < dQnorm or loop > force_convergence_point

            if dQnorm > dQnormThr:
                dX = np.dot(B_inv.transpose(), dQ)
                ind = 0
                for i in range(self.num_atoms()):
                    temp_list = list(self.atoms[i].xyz)
                    for j in range(3):
                        temp_list[j] += dX[ind]
                        ind += 1
                    self.atoms[i].xyz = tuple(temp_list)
                continue
            self.xyz_string_form_debug(False, "After additions")
            return

        er = "Redundants did not converge within " + str(nloops) + " iterations"
        LOG.error(er)
        raise CSPyException(er)

    def invert(self) -> None:
        """Invert the coordinates of a Molecule"""
        cent = self.centroid()
        for i in range(self.num_atoms()):
            centered = list3subtract(self.atoms[i].xyz, cent)
            self.atoms[i].xyz = list3add(list3multiply(centered, -1.0), cent)

    def xyz_array(self) -> List:
        """Convert the xyz coordinatesfrom an array to a list

        Returns:
            List: A copy of the array data as a (nested) Python list
        """
        xyz = np.zeros(shape=(self.num_atoms(), 3))
        for i, a in enumerate(self.atoms):
            xyz[i] = a.xyz
        return xyz.tolist()

    def get_fchk_gaussian_energy(self, filename: str) -> float:
        """Get the total energy from a Gaussian FCHK file

        Args:
            filename (str): A path or the name of the .fchk file

        Raises:
            CSPyException: Raised if no energy can be found

        Returns:
            float: The total energy
        """
        for line in open(filename, "r"):
            if "Total Energy" in line:
                self.energy = float(line.strip().split()[-1])
                return self.energy
        raise CSPyException("No energy found")

    def init_from_fchk(self, filename: str) -> None:
        """Create a Molecule from a Gaussian fchk file

        Args:
            filename (str): A path or the name of the .fchk file
        """
        f = open(filename, "r")
        self.atoms = []
        for line in f:
            if line.startswith("Atomic numbers"):
                sl = line.strip().split()
                natoms = int(sl[-1])
                nlines = int(math.ceil(float(natoms) / 6.0))
                vals = []
                for _ in range(nlines):
                    line = next(f)
                    vals += map(float, line.strip().split())
                for n in range(natoms):
                    self.add_atom(FlexAtom(vals[n], (0.0, 0.0, 0.0)))
            if line.startswith("Current cartesian coordinates"):
                sl = line.strip().split()
                nl = int(sl[-1])
                nlines = int(math.ceil(float(nl) / 5.0))
                vals = []
                for _ in range(nlines):
                    line = next(f)
                    vals += map(float, line.strip().split())
                for n in range(natoms):
                    self.atoms[n].xyz = tuple(
                        [
                            vals[n * 3 + 0] * 0.529177249,
                            vals[n * 3 + 1] * 0.529177249,
                            vals[n * 3 + 2] * 0.529177249,
                        ]
                    )
        f.close()
        try:
            self.get_fchk_gaussian_energy(filename)
        except CSPyException as exc:
            pass  # no energy in file is ok

    def init_from_xyz_str(self, init_str: str) -> None:
        """Create a Molecule from the file content of a .xyz file

        Args:
            init_str (str): The contents of an .xyz file as a single string
        """
        init_num_atoms = None
        for i, line in enumerate(init_str.split("\n")):
            sline = line.split()
            if line.strip() == "":
                pass
            elif i == 0:
                # if true then xyz starts with a number
                if len(sline) == 1:
                    init_num_atoms = int(sline[0])
                else:
                    self.add_atom_from_str(line)
            elif i == 1 and init_num_atoms:
                # this will be the comment line
                pass
            else:
                self.add_atom_from_str(line)

    def init_from_xyz(self, filename: str) -> None:
        """Create a Molecule from a .xyz file

        Args:
            filename (str): A path or the name of the .xyz file

        Raises:
            CSPyException: If the file cannot read
        """
        if is_file_readable(filename):
            self.init_from_xyz_str(open(filename).read())
        else:
            er = "Cannot open file [" + filename + "] for reading XYZ input"
            raise CSPyException(er)

    def make_internal_dof(
        self,
        coord: str,
        offset: float = 0.0,
        initial: Union[None, float] = None,
        n_steps: int = 0,
        step_size: float = 0.0,
    ):
        """Create an instance of the Internal class for a specified internal coordinate and its modification

        Args:
            coord (str): The internal coordinate to be explored, defined in terms of the indices of the atoms involved (e.g. 0_1_2_3 would describe the torsion between atoms 1, 2, 3 and 4)
            offset (float, optional): The offset from the initial conformation to start the scan from. Defaults to 0.0.
            initial (Union[None, float], optional): Initial value of the the internal coordinate. This is an alternative to (and mutually exclusive with) the offset o. Defaults to None.
            n_steps (int, optional): The number of steps to take during the scan. Defaults to 0.
            step_size (float, optional): The step size. Defaults to 0.0.

        Raises:
            CSPyException: Raised if offset and initial value are both set.

        Returns:
            Internal: An Internal describing the internal coordinate and the modification to be applied during the scan
        """
        current_value = self.dof_value(coord, radians=True)
        if initial is None:
            starting_value = None
        else:
            if offset != 0.0:
                raise CSPyException("Dont specify both offset!=0 and initial values")
            starting_value = initial
        i = Internal(
            coord,
            original_value=current_value,
            initial_offset=offset,
            starting_value=starting_value,
            number_of_steps=n_steps,
            step_size=step_size,
        )
        return i

    def make_internal_dof_from_dict(self, udict: dict) -> "Internal":
        """Make an Internal for each specified internal coordinate

        Args:
            udict (dict): A dictionary containing information regarding the internal coordinate and the modification to be applied

        Returns:
            Internal: An Internal describing the internal coordinate and modification that is to be applied
        """
        c = "coord"
        n = "n_steps"
        s = "step_size"
        o = "offset"
        i = "initial"
        default = {n: 0, s: 0.0, o: 0.0, i: None}
        default.update(udict)
        internal = self.make_internal_dof(
            coord=default[c],
            offset=default[o],
            initial=default[i],
            n_steps=default[n],
            step_size=default[s],
        )
        return internal

    def eval_user_dofs_str(self, scan_dofs: str, mercury_indexing: bool = True) -> List:
        """Evaluate the users input degrees of freedom string to create a series of Internals


        Args:
            scan_dofs (List[str]): A list containing the internal coordinate to modify e.g. ["{c:'12_11_5_3',n:179,s:2,o:-180}"].
            For more information please see the documentation.
            mercury_indexing (bool, optional): User input indexes atoms like Mercury. I.e. starts from 1 (not 0).

        Returns:
            List: A list of Internals
        """
        c = 'coord'
        n = 'n_steps'
        s = 'step_size'
        o = 'offset'
        i = 'initial'
        D = math.pi / 180.  # indicate degrees
        R = 1.

        # if indexed like mercury, we correct back to python indexing by subtracting one
        if mercury_indexing:
            for dof_index, dof in enumerate(scan_dofs):
                dof = dof.replace('\"', '\'')
                dof_split = dof.split('\'')
                ids_split = dof_split[1].split('_')
                ids_minusone = [str(int(id)-1) for id in ids_split]
                dof_split[1] = '_'.join(ids_minusone)
                scan_dofs[dof_index] = '\''.join(dof_split)
        
        scan_dofs_str = '[' + ','.join(scan_dofs) + ']'
        user_dofs = eval(scan_dofs_str)
        di = [self.make_internal_dof_from_dict(dof) for dof in user_dofs]

        return di

    @staticmethod
    def make_sobol_points_of_internals(
        internals: List[Internal],
        npoints: int,
        starting_seed: int = 1,
        coupled: bool = False,
    ) -> List:
        """Creates a series of scan points for internal coordinates based on the Sobol method

        Args:
            internals (List[Internals]): A list of Internals
            npoints (_type_): The number of sobol points
            starting_seed (int, optional): The initial seed number. Defaults to 1.
            coupled (bool, optional): Make every internal share one Sobol
                coordinate. Defaults to False.

        Returns:
            List: A list of scan points
        """
        ndim = 1 if coupled else len(internals)
        seed = starting_seed
        scan_points = []
        for n in range(npoints):
            vec = sobol_vector(seed, ndim)
            seed += 1
            scan_point = []
            for i, internal in enumerate(internals):
                v = vec[0] if coupled else vec[i]
                inter = internal.copy()
                # ``number_of_steps`` is the number of grid points, whose step
                # indices run from zero to number_of_steps - 1.  Scale the
                # Sobol coordinate over that same full interval rather than
                # sampling only between grid steps zero and one.
                inter.current_step = v * max(inter.number_of_steps - 1, 0)
                scan_point.append(inter)
            scan_points.append(scan_point)
        return scan_points

    @staticmethod
    def make_combination_of_internal_steps(
        internals: List[Internal], coupled: bool = False
    ) -> List:
        """Create a combination of scan steps from a list of internal coordinates

        Args:
            internals (List): A list of Internals
            coupled (bool, optional): Advance every internal using the same
                scan index instead of constructing a Cartesian product.

        Returns:
            List: A list containing the combination of scan points
        """

        def perm(alist):
            if len(alist) < 2:
                return [[item] for item in alist[0]]
            tails = perm(alist[1:])
            return [[item] + p for item in alist[0] for p in tails]

        if coupled:
            n_points = {internal.n_points for internal in internals}
            if len(n_points) != 1:
                raise ValueError(
                    "Coupled grid scan DOFs must have the same number of steps"
                )
            scans = [internal.scan_internals() for internal in internals]
            return [list(point) for point in zip(*scans)]

        scan_points = perm([x.scan_internals() for x in internals])
        return scan_points

    def cleanup(self, files: List) -> None:
        """Function for cleaning up files

        Args:
            files (List): A list of filenames or paths to files to remove
        """
        for fname in files:
            f = Path(fname)
            if f.exists():
                f.unlink()

    def gaussian_scan(
        self,
        internals: List,
        comm: Any,
        filename_prefix: str = "scan",
        gaussian_args: dict = {},
        method: str = "B3LYP",
        charges: str = "0",
        potential: str = "F",
        multiplicities: str = "1",
        basis_set: str = "6-311G**",
        foreshorten_hydrogens: Union[None, float] = None,
        sobol_points: int = 0,
        coupled_scan_dofs: bool = False,
        constraints: Union[None, List] = None,
        redundant: Union[None, str] = None,
    ) -> None:
        """Generates a series of molecular conformations by applying distortions to a set of specified internal coordinates.
        Energies of each conformer are evaluated using Gaussian and stored into a molecular SQL database.

        Args:
            internals (List): A List of internals to scan
            comm (MPI.COMM_WORLD): An MPI.COMM_WORLD communicator usually passed from cspy.apps.mol_dis
            filename_prefix (str, optional): A prefix to use for new files. Defaults to "scan".
            gaussian_args (dict, optional): A dictionary of arguments to pass to Gaussian. Defaults to {}.
            method (str, optional): The method to employ during the Gaussian single-point energy calculation. Defaults to "B3LYP".
            charges (str, optional): The charge of the molecule which can be passed to Gaussian. Defaults to "0".
            potential (str, optional): The potential type to be used in the flexible-molecule CSP. Can take values of eithe 'F' for FIT or 'W' for Williams type potentials. Defaults to "F".
            multiplicities (str, optional): The multiplicity of the molecule which can be passed to Gaussian.. Defaults to "1".
            basis_set (str, optional): The basis set to employ during the Gaussian single-point energy calculation. Defaults to "6-311G**".
            foreshorten_hydrogens (Union[None,float], optional): Amount to foreshorten hydrogen positions by. Usually set to None for FIT potentials and 0.1 for Williams potentials. Defaults to None.
            sobol_points (int, optional): The number of points to use for generating the sobol grid. Defaults to 0.
            coupled_scan_dofs (bool, optional): Advance all scan DOFs with a
                shared coordinate. Defaults to False.
            constraints (Union[None, List], optional): Constraints for a constrained gaussian calculation. ['1 2 3', '5 6 7 8', '9'] would lead to the fixing of the angle between" \
                                                    "atoms 1,2, and 3, the dihedral angle between atoms 5, 6, 7 and 8, and the atomic position of atom '9'."
            redundant (Union[None, str], optional): Constraints mandatory option for Gaussian calc; It should be opt=ModRedundant. Defaults to None.
        """
        Ha2kjmol = 2625.5002
        rank = comm.Get_rank()
        nprocs = comm.Get_size()

        if rank == 0:
            db_path = Path(filename_prefix + "_flex.db")
            exists_db = db_path.exists()
            if exists_db:
                db = CspDataStoreFlex(filename_prefix + "_flex.db")
                conformations = [
                    row[0] for row in db.select("distorted_molecules", ["id"])
                ]
                db.disconnect()
            else:
                create_flex_database(filename_prefix)

            if sobol_points > 0:
                scan_points = FlexMolecule.make_sobol_points_of_internals(
                    internals, sobol_points, coupled=coupled_scan_dofs
                )
            else:
                scan_points = FlexMolecule.make_combination_of_internal_steps(
                    internals, coupled=coupled_scan_dofs
                )

            map_dict = {}
            if exists_db:
                new_scan_points = []
                new_id = 0
                for idx, item in enumerate(scan_points):
                    element = str(idx) + "_" + filename_prefix
                    if element not in conformations:
                        new_id += 1
                        LOG.info("Conformer %i is not in the database.", int(idx))
                        new_scan_points += [item]
                        map_dict[str(new_id - 1)] = str(idx)
                scan_points = new_scan_points
            else:
                with open(filename_prefix + "-angles.dat", "w+") as file_angles:
                    for idx, point in enumerate(scan_points):
                        line = (
                            str(idx)
                            + "  "
                            + "  ".join(
                                map(
                                    str,
                                    [
                                        round(math.degrees(x.get_current_offset()), 1)
                                        for x in point
                                    ],
                                )
                            )
                            + "\n"
                        )
                        file_angles.write(line)

            ave, res = divmod(len(scan_points), nprocs)
            count = np.array([ave + 1 if p < res else ave for p in range(nprocs)])
            starts = [sum(count[:p]) for p in range(nprocs)]
            ends = [sum(count[: p + 1]) for p in range(nprocs)]
            scan_points = [scan_points[starts[p] : ends[p]] for p in range(nprocs)]
        else:
            exists_db = None
            map_dict = None
            scan_points = None
            starts = None

        exists_db = comm.bcast(exists_db, root=0)
        map_dict = comm.bcast(map_dict, root=0)
        starts = comm.bcast(starts, root=0)
        data_flex = comm.scatter(scan_points, root=0)

        data_dict = {}
        for id, point in enumerate(data_flex):
            if exists_db:
                id_label = map_dict[str(id + starts[rank])]
            else:
                id_label = str(id + starts[rank])

            LOG.info(
                "%i: Applying values %s",
                int(id_label),
                [round(math.degrees(x.get_current_offset()), 1) for x in point],
            )
            molec_name = id_label + "_" + filename_prefix
            factors = [x.get_current_offset() for x in point]
            internals_list = [str(x).split()[0] for x in point]
            try:
                mod_mol = copy.deepcopy(self)
                mod_mol.apply_mode_addition_in_internals(
                    factors=factors,
                    internals_list=internals_list,
                    dQnormThr=2.0e-6,
                    recompute_K_thr=10.0,
                    strict_dihedrals=False,
                )

            except Exception as exc:
                LOG.info(
                    f"""Molecule {molec_name} produced the folloiwing exception in the apply_mode_addition_in_internals function: 
                    {traceback.format_exc()}"""
                )
                continue

            # Calculation of the molecular axes, multipoles, single-point energy,
            # charge and res file by calling 'generate_multipoles' function in cspy/apps/dma.py
            xyz_file = mod_mol.xyz_string_to_database(comment=f"{molec_name}")
            name = f"{molec_name}.xyz"
            with open(name, "w+") as f:
                f.write(xyz_file)

            title = generate_combined_name(filenames=[name])
            combined_res_filename = f"{title}.res"
            mf_format = (
                title + "{}.dma"
            )  # The {} are filled in by fit_multipoles_with_mulfit() in cspy/apps/dma.py
            crystal = generate_combined_res(
                [name], title, combined_res_filename, sort=True
            )
            if potential == "W" and foreshorten_hydrogens is None:
                foreshorten_hydrogens = 0.1
                LOG.info(
                    "Reducing hydrogen bond lengths by: %.2f", foreshorten_hydrogens
                )
            try:
                cleanup_files = generate_multipoles(
                    crystal,
                    [0],  # ranks
                    mf_format,
                    potential_type=potential,
                    axis_file=None,
                    basis_set=basis_set,
                    method=method,
                    dma_switch=4,
                    foreshorten_hydrogens=foreshorten_hydrogens,
                    hydrogen_radius=None,
                    xyz_filenames=[name],
                    charges=charges,
                    multiplicities=multiplicities,
                    constraints=constraints,
                    redundant=redundant,
                    flex_scf=True,
                    **gaussian_args,
                )
            except Exception as e:
                LOG.info(
                    f"""Molecule {molec_name} produced the following exception:\n{traceback.format_exc()}"""
                )
                continue

            e = None
            try:
                e = mod_mol.get_fchk_gaussian_energy(molec_name + "_A.fchk")
            except CSPyException as exc:
                LOG.info(f"{molec_name}_A.fchk: No Energy found")
            except IOError as exc:
                LOG.info(f"{molec_name}_A.fchk does not exist, skipping")
            if e is None:
                continue
            else:
                mod_mol.init_from_fchk(molec_name + "_A.fchk")
                mod_mol.xyz_print_to_file(
                    f"{molec_name}-out.xyz",
                    comment=f"Output cartesian coordinates of {molec_name}",
                )

                exts = ["-out.xyz", ".dma", ".mols", "_rank0.dma", ".res"]
                conf_db_info = []
                for ext in exts:
                    with open(f"{molec_name+ext}", "r") as f:
                        conf_db_info.append("".join([line for line in f]))
                conf_db_info.insert(1, (e * Ha2kjmol))

                LOG.info(f"The energy of conformer {molec_name} is {e*Ha2kjmol}")
                data_dict[molec_name] = conf_db_info
                add_to_flex_database(data_dict, filename_prefix)
                exts += [".xyz"]
                cleanup_files += [molec_name + extension for extension in exts]
            self.cleanup(cleanup_files)
        LOG.debug(f'Rank {rank}: Completed assigned scan points. Waiting for other ranks to finish.')
        comm.Barrier()
