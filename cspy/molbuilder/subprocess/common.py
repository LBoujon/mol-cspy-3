import copy
import logging
import re
from rdkit import Chem
from rdkit.Chem import AllChem
from cspy.molbuilder.exceptions import UFFOptimizationFailure


LOG = logging.getLogger(__name__)


def coordinates_from_uff(molecule):
    """Generates a list of coordinates from a rdkit molecule object by
    running a rdkit UFF optimization. Assumes only one conformer exists.

    :param molecule: rdkit molecule object
    :return: a list of coordinates
    """
    LOG.debug('Started UFF optimization with molecule %s',
              Chem.MolToInchi(molecule, options='-SNon'))

    molecule = copy.deepcopy(molecule)
    try:
        AllChem.EmbedMolecule(molecule, useBasicKnowledge=False)
        AllChem.UFFOptimizeMolecule(molecule)
    except Exception as e:
        raise UFFOptimizationFailure('UFF Optimization failure: %s', e)

    LOG.debug('Completed UFF optimization with molecule %s',
              Chem.MolToInchi(molecule, options='-SNon'))

    return coordinates_from_rdkit_molecule(molecule)


def coordinates_from_rdkit_molecule(molecule):
    """Generates a list of coordinates from a rdkit molecule object.

    :param molecule: rdkit molecule object
    :return: a list of coordinates
    """
    coordinates = []
    for atom in molecule.GetAtoms():
        position = molecule.GetConformer().GetAtomPosition(atom.GetIdx())
        coordinates.append(
            [atom.GetAtomicNum(), position.x, position.y, position.z]
        )
    return coordinates


def coordinates_from_xyz_file(xyz_file_name):
    """From an xyz file generate a list of coordinates.

    :param xyz_file_name: xyz file name
    :return: a list of coordinates
    """
    coordinates = []

    with open(xyz_file_name, 'r') as xyz:
        lines = xyz.read().splitlines()

    for line in lines[2:]:
        if not re.match(' *([0-9]+|[A-Z][a-z]?.*)([ \t]+[-]?[0-9]+[.][0-9]+){3} *', line):
            continue

        coordinate = list(re.match(
            ' *([0-9]+|[A-Z][a-z]?.*)[ \t]+([-]?[0-9]+[.][0-9]+)[ \t]+([-]?[0-9]+[.][0-9]+)'
            '[ \t]+([-]?[0-9]+[.][0-9]+) *', line
        ).group(1, 2, 3, 4))

        coordinate[0] = coordinate[0].rstrip()
        coordinate[1] = float(coordinate[1])
        coordinate[2] = float(coordinate[2])
        coordinate[3] = float(coordinate[3])

        coordinates.append(coordinate)

    return coordinates


def coordinates_from_cspy_molecule(molecule):
    """Generates a list of coordinates from a cspy molecule object.

    :param molecule: cspy molecule object
    :return: a list of coordinates
    """
    coordinates = []
    for element, (x, y, z) in zip(molecule.elements, molecule.positions):
        coordinates.append([element, x, y, z])
    return coordinates


def coordinates_to_xyz_file(coordinates, xyz_file_name):
    """Writes a list of coordinates to an xyz file.

    :param coordinates: coordinates of the molecule
    :param xyz_file_name: xyz file name
    """
    with open(xyz_file_name, 'w') as xyz:
        xyz.write('{}\n\n'.format(len(coordinates)))
        for coordinate in coordinates:
            xyz.write('{} {:.6f} {:.6f} {:.6f}\n'.format(*coordinate))
