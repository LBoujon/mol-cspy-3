import logging
import copy
import numpy as np
from scipy.spatial.transform import Rotation as R


LOG = logging.getLogger(__name__)


class PhononException(Exception):
    """General exception of phonon functions

    """


def jmol_phonon_xyz_file(crystal, filename, **kwargs):
    """Creates an xyz file that can be loaded in jmol so that the
    vibrational animation can be viewed.

    Parameters
    ----------
    crystal : cspy.Crystal
        crystal structure that jmol vibrational animation will be
        created for, crystal object must have phonon properties
    filename : str
        filename that the jmol animation will be written to

    """
    if ("phonon_frequencies" not in crystal.properties
            or "phonon_eigenvectors" not in crystal.properties):
        raise PhononException(
            "Crystal structure is missing the phonon frequencies and/or "
            "phonon eigenvector property and is unable to generate jmol "
            "vibrational xyz file."
        )

    eigenvectors = crystal.properties["phonon_eigenvectors"]
    n, m = eigenvectors.shape
    if n != m:
        raise PhononException(
            "Phonon eigenvectors is not a square matrix, certain modes "
            "must have been removed or something wrong happened when "
            "obtaining these eigenvectors."
        )

    mols = crystal.unit_cell_molecules()
    if len(mols) * 6 == m:
        max_tran = kwargs.get("max_translation", 0.5)
        max_rota = kwargs.get("max_rotation", np.pi / 36)
        jmol_rigid_body_phonon_xyz_file(crystal, max_tran, max_rota, filename)
    elif sum([len(i.elements) for i in mols]) * 3 == m:
        raise NotImplementedError(
            "Only rigid body phonon are supported"
        )
    else:
        raise PhononException(
            "Unable to figure out if phonon eigenvectors refer to rigid "
            "body or atomistic vibrations"
        )


def jmol_rigid_body_phonon_xyz_file(crystal, max_tran, max_rota, filename):
    """Using the rigid body phonon eigenvectors obtained from dmacrys
    for example write out an xyz file that can be loaded in jmol so
    that the vibrational animation can be viewed.

    Parameters
    ----------
    crystal : cspy.Crystal
        crystal structure that jmol vibrational animation will be
        created for, crystal object must have phonon properties
    max_tran : float
        max finite translation that will be used to calculate
        vibrational displacements
    max_rota : float
        max finite rotation that will be used to calculate vibrational
        displacements
    filename : str
        filename that the jmol animation will be written to

    """
    frequencies = crystal.properties["phonon_frequencies"]
    eigenvectors = crystal.properties["phonon_eigenvectors"] \
                   @ crystal.properties["transform_cspy_global"]
    mols = crystal.unit_cell_molecules()
    num_mols = len(mols)
    num_atoms = sum([len(i.elements) for i in mols])
    a, b, c = crystal.unit_cell.lengths
    alpha, beta, gamma = crystal.unit_cell.angles

    with open(filename, "w") as jmol_file:
        for i, eigenvector in enumerate(eigenvectors):

            jmol_file.write("{}\n".format(num_atoms))
            if i == 0:
                jmol_file.write(
                    'Mode {} {} cm-1 jmolscript: load "" {{1 1 1}} '
                    'spacegroup "x,y,z" unitcell {{{} {} {} {} {} {}}}; '
                    'select all; vibration period 1; vibration scale 1; '
                    'vibration on; vector scale 5; vector on;\n'.format(
                        i + 1, frequencies[i], a, b, c, 180 * alpha / np.pi,
                        180 * beta / np.pi, 180 * gamma / np.pi
                    )
                )
            else:
                jmol_file.write(
                    'Mode {} {} cm-1\n'.format(i + 1, frequencies[i])
                )

            mol_eigenvector = np.split(eigenvector, num_mols)
            for j, mol in enumerate(mols):
                translation, rotation = np.split(mol_eigenvector[j], 2)
                mag_rota = np.linalg.norm(rotation)

                mol_copy = copy.deepcopy(mol)
                # rotate the molecule copy by finite amount, skip if
                # rotation is too small otherwise scipy can crash
                if mag_rota >= 1e-3 * np.pi / 180:
                    axis_x, axis_y, axis_z = rotation / mag_rota
                    mol_copy.translate(- mol.center_of_mass)
                    sin_rota = np.sin(max_rota * mag_rota / 2)
                    rota_matrix = R.from_quat([
                        axis_x * sin_rota, axis_y * sin_rota,
                        axis_z * sin_rota, np.cos(max_rota * mag_rota / 2)
                    ]).as_matrix()
                    mol_copy.transform(rota_matrix)
                    mol_copy.translate(mol.center_of_mass)

                # translate molecule copy by finite amount
                mol_copy.translate(max_tran * translation)

                differences = mol_copy.positions - mol.positions
                for k, difference in enumerate(differences):
                    original = mol.positions[k]
                    jmol_file.write(
                        "{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n".format(
                        mol.elements[k].symbol, original[0], original[1],
                        original[2], difference[0], difference[1], difference[2]
                    ))
