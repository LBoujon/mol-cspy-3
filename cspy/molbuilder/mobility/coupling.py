import logging
import math
import ast
import re
import os
import itertools as it
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
from scipy.spatial import cKDTree as KDTree
from cspy.molbuilder.subprocess import coordinates_from_cspy_molecule
from cspy.molbuilder.subprocess import coordinates_to_xyz_file
from cspy.pyAOMlite.pyAOM_utils import Sab
from cspy.pyAOMlite.pyAOM_utils import single_molecule
from .dimer import perturbed_molecules


LOG = logging.getLogger(__name__)


amu_to_au = 1822.888486209
wave_to_e_h = 299792458 * 100 * 6.62607015e-34 / 4.3597447222071e-18
ev_to_e_h = 1 / 27.211386245988
ang_to_bohr = 1 / 0.529177210903


class AOM:

    def __init__(self, config, mo_number, aom_coeff_path):
        """The AOM object used to run electronic coupling calculations.

        :param config: config parser object
        :param mo_number: The MO number from the cp2k output to use.
        :param aom_coeff_path: AOM coeff file path.
        """
        self.delta_tran = float(config['MOBBUILDER'].get('delta_tran', 0.01))
        self.delta_rota = float(config['MOBBUILDER'].get('delta_rota', 0.01))
        self.vdw_buffer = float(config['MOBBUILDER']['vdw_buffer'])
        self.aom_dict = ast.literal_eval(config['AOM']['AOM_dict'])
        self.c = float(config['AOM']['C'])
        self.mo_number = mo_number
        self.aom_coeff_path = aom_coeff_path

    def run(self, mol_i, mol_j):
        """Calculate the electronic coupling.

        :param mol_i: A cspy molecule.
        :param mol_j: A cspy molecule.
        :return: The electronic coupling between mol_i and mol_j.
        """
        coordinates = coordinates_from_cspy_molecule(mol_i) \
                      + coordinates_from_cspy_molecule(mol_j)
        with TemporaryDirectory(prefix="/dev/shm/") as tmpdirname:
            coordinates_to_xyz_file(coordinates, tmpdirname + '/dimer.xyz')
            results = Sab(
                tmpdirname + '/dimer.xyz', self.aom_coeff_path,
                self.aom_coeff_path, self.mo_number, self.mo_number,
                self.aom_dict
            )
        return self.c * results

    def calc_local_coupling(self, active_mols, contacts=None):
        """Calculates a matrix of local couplings.

        :param active_mols: The unit cell molecules and molecules
            surrounding it used for electronic coupling calculations.
        :param contacts: A matrix of bools state if two molecules are
            in close contact based on electronic couplings.
        :return: A matrix of local couplings.
        """
        LOG.debug('Calculating local electronic coupling tensor elements')
        local_couplings = np.zeros((len(active_mols), len(active_mols)))
        for (i, mol_i), (j, mol_j) in it.combinations(enumerate(active_mols), 2):
            if contacts is None or contacts[i, j]:
                local_couplings[i, j] = self.run(mol_i, mol_j)
        LOG.debug('Completed calculating local electronic coupling '
                  'tensor elements')
        return local_couplings + local_couplings.T

    def calc_nonlocal_coupling(self, crystal, depths, full_mol_idxs, contacts):
        """A list of matrices of nonlocal couplings, one for each
        vibrational mode.

        :param crystal: The crystal structure.
        :param depths: The expansions used to create the supercell to
            find the active molecules.
        :param full_mol_idxs: The full molecule indices of the
            active molecules.
        :param contacts: A matrix of bools state if two molecules are
            in close contact based on electronic couplings.
        :return: A list of nonlocal coupling matrices.
        """
        num_uc_mols = len(crystal.unit_cell_molecules())
        phonon_eigenvectors = crystal.properties['phonon_eigenvectors']
        transformation = crystal.properties['transform_cspy_global']
        phonon_freq = wave_to_e_h * crystal.properties['phonon_frequencies']
        mass_and_moments = amu_to_au * crystal.properties['mass_and_moments']
        LOG.debug('Calculating nonlocal electron-phonon coupling tensor '
                  'elements for crystal structure with %s unit cell '
                  'molecules', num_uc_mols)

        nonlocal_couplings = []
        for i, vector in enumerate(transformation):
            uc_mol_idx = i // 6
            axis = np.split(vector, num_uc_mols * 2)[i // 3]

            if (i // 3) % 2 == 0:
                delta = self.delta_tran
                derivative = 'translation'
            else:
                delta = self.delta_rota
                derivative = 'rotation'

            delta_mols = perturbed_molecules(
                crystal, uc_mol_idx, axis, delta, derivative,
                depths, full_mol_idxs
            )
            local_couplings = self.calc_local_coupling(delta_mols, contacts)
            delta_mols = perturbed_molecules(
                crystal, uc_mol_idx, axis, -delta, derivative,
                depths, full_mol_idxs
            )
            local_couplings -= self.calc_local_coupling(delta_mols, contacts)

            nonlocal_couplings.append(local_couplings * ev_to_e_h / (
                    2 * delta * ang_to_bohr * math.sqrt(mass_and_moments[i])
            ))

        nonlocal_couplings = np.einsum(
            'ijk,li->ljk', np.array(nonlocal_couplings), phonon_eigenvectors
        )[3:]
        phonon_freq = phonon_freq[3:]
        for i, freq in enumerate(phonon_freq):
            nonlocal_couplings[i] \
                = math.sqrt(1 / (2 * freq)) * nonlocal_couplings[i] / freq

        LOG.debug('Completed calculating nonlocal electron-phonon coupling '
                  'tensor elements')
        return nonlocal_couplings

    def molecule_close_contacts(self, mols):
        """Generates a matrix of bools stating whether the two
        molecules are in close contact.

        :param mols: A list of cspy molecule objects.
        :return: A numpy array of bools stating whether the two
            molecules are in close contact.
        """
        LOG.debug('Calculating molecular close contacts with AOM')
        trees = [KDTree(mol.positions) for mol in mols]
        vdws = [[i.vdw for i in mol.elements] for mol in mols]
        vdws = [i for j in vdws for i in j]
        max_vdw = np.max(vdws)

        close_contacts = np.zeros((len(mols), len(mols)))
        for (i, tree_i), (j, tree_j) in it.combinations(enumerate(trees), 2):
            contacts = tree_i.query_ball_tree(
                tree_j, 2 * max_vdw + self.vdw_buffer)
            if any([len(i) != 0 for i in contacts]):
                close_contacts[i, j] = 1

        return (close_contacts + close_contacts.T).astype(bool)

    @classmethod
    def load(cls, config, cp2k_out_file, xyz, carrier):
        """Create the AOM object from config, cp2k out, an xyz file
        for a specified carrier.

        :param config: Config parser object.
        :param cp2k_out_file: cp2k output file location.
        :param xyz: xyz file location of the molecule.
        :param carrier: Should be 'hole' or 'electron'.
        :return: An AOM object.
        """
        with open(cp2k_out_file, 'r') as cp2k:
            cp2k_out = cp2k.read()
        mo_number = int(
            re.findall(' *Number of electrons: +([0-9]+) *', cp2k_out)[0]
        )
        if carrier == 'electron':
            mo_number += 1
        elif not carrier == 'hole':
            raise ValueError('The carrier should be hole or electron.')

        mol = single_molecule(xyz)
        mol.get_cp2k_info(
            mo_number, cp2k_out_file, config['AOM']['cp2k_basis_file'],
            config['AOM']['basis']
        )
        sto_proj_dict = ast.literal_eval(config['AOM']['STO_proj_dict'])
        mol.initialize_STOs(sto_proj_dict)
        mol.project()
        coeff_dir = Path(os.getcwd() + '/output/{}'.format(Path(xyz).stem))
        coeff_dir.mkdir(exist_ok=True, parents=True)
        mol.save_AOM(Path(xyz).stem)
        LOG.info('STO projection completeness: %s', mol.orb_compl_dict)
        aom_coeff_path \
            = os.getcwd() + '/output/{}/AOM_COEFF.dat'.format(Path(xyz).stem)
        return cls(config, mo_number, aom_coeff_path)
