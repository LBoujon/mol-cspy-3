import math
import logging
import numpy as np
import itertools as it
from collections import defaultdict
from cspy.molbuilder.exceptions import MobilityCalculationFailure
from cspy.crystal import UnitCell
from .dimer import active_molecules


LOG = logging.getLogger(__name__)


amu_to_au = 1822.888486209
wave_to_e_h = 299792458 * 100 * 6.62607015e-34 / 4.3597447222071e-18
wave_to_ev = 299792458 * 100 * 6.62607015e-34 / 1.602176634e-19
ev_to_e_h = 1 / 27.211386245988
ang_to_bohr = 1 / 0.529177210903
k_b_ev = 8.617333262145e-5
k_b_cm = 0.695034800
h_bar_ev = 6.582119569e-16


def mol_idx(u, v, w, uc_idx, a, b, c):
    """Generates an index for the matrices in TLT for a given unit
    cell translation, molecule and TLT supercell expansion.

    :param u: A component of the unit cell translation.
    :param v: A component of the unit cell translation.
    :param w: A component of the unit cell translation.
    :param uc_idx: The unit cell index of the molecule, this is the
        index of the molecule in the list from the
        unit_cell_molecules method.
    :param a: A component of the expansion of the TLT supercell.
    :param b: A component of the expansion of the TLT supercell.
    :param c: A component of the expansion of the TLT supercell.
    :return: The index of the molecule in the matrices for the
        TLT supercell.
    """
    u = u - a * math.floor(u / a)
    v = v - b * math.floor(v / b)
    w = w - c * math.floor(w / c)
    return u + v * a + w * (a * b) + (a * b * c * uc_idx)


class TransientLocalisation:

    def __init__(self,  config, carrier):
        """Transient localisation theory object.

        :param config: Config parser object.
        :param carrier: The charge carrier should be electron or hole.
        """
        self.len_target = float(config['MOBBUILDER']['len_target'])
        self.vdw_buffer = float(config['MOBBUILDER']['vdw_buffer'])
        self.h_bar_tau = float(config['MOBBUILDER']['h_bar_over_tau'])
        self.iterations = int(config['MOBBUILDER']['iterations'])
        if carrier == 'electron':
            self.sign = -1
        elif carrier == 'hole':
            self.sign = 1
        else:
            raise ValueError('Carrier should be electron or hole.')

    def mobility(self, network, crystal, expansion, sigma, temp):
        """Calculate the mobility using transient localisation theory (TLT).

        :param network: The coupling network.
        :param crystal: The crystal structure.
        :param expansion: Tuple of the expansions used to create the
            TLT supercell.
        :param sigma: The variance of the occupation energy.
        :param temp: The temperature of the calculation.
        :return: The isotropic mobility calculated using TLT.
        """
        if crystal.space_group.symbol != 'P1':
            crystal = crystal.as_P1()

        a, b, c = expansion
        uc_mols = crystal.unit_cell_molecules()
        num_uc_mols = len(uc_mols)
        tlt_num_mols = num_uc_mols * a * b * c

        base, sigmas = self.create_hamiltonian_matrices(
            num_uc_mols, tlt_num_mols, expansion, sigma, network
        )
        r_x, r_y, r_z = self.create_distance_matrices(
            crystal.unit_cell, uc_mols, tlt_num_mols, expansion
        )

        avg_mob = 0
        for i in range(self.iterations):
            # hamiltonian and sigmas are upper triangle matrices
            # the deviations are applied symmetrically as physically
            # H_{nn'} should equal to H_{n'n}
            hamiltonian = np.random.normal(base, sigmas)
            hamiltonian = hamiltonian + hamiltonian.T \
                          - np.diag(np.diag(hamiltonian))
            energies, vectors = np.linalg.eigh(hamiltonian)

            tile = np.tile(energies, (len(energies), 1))
            delta_e_sq = np.square(tile - tile.T)
            # The TLT hamiltonian is a single particle hamiltonian
            # which is solved to determine the coefficients of the
            # linear combination of HOMO/LUMOs (LCMO). For electron
            # transport it is simply a sum weighted from the
            # lowest energies LCMOs. For hole transport we form
            # a single slater determinant from these LCMOs which are
            # filled up with N - 1 electrons so the thermal average is
            # summed over all N - 1 slater determinants with different
            # configurations. By going through the maths it will turn
            # out that this is equivalent to taking the summations over
            # LCMOs but using a plus sign in the exp.
            boltz = np.exp(self.sign * energies / (k_b_ev * temp))
            pre = (2 * boltz) / (self.h_bar_tau**2 + delta_e_sq)
            z = np.sum(boltz)

            j_x = r_x * hamiltonian
            j_y = r_y * hamiltonian
            j_z = r_z * hamiltonian

            l_xx = np.sum(pre.T * np.square(vectors.T @ j_x @ vectors)) / z
            l_yy = np.sum(pre.T * np.square(vectors.T @ j_y @ vectors)) / z
            l_zz = np.sum(pre.T * np.square(vectors.T @ j_z @ vectors)) / z

            factor = 1 / (2 * (h_bar_ev / self.h_bar_tau) * k_b_ev * temp)
            avg_mob += 1e-16 * factor * (l_xx + l_yy + l_zz) / 3
            LOG.info('Average mobility: {}'.format(avg_mob / (i + 1)))

        return avg_mob / self.iterations

    def create_hamiltonian_matrices(self, num_uc_mols, num_mols, expansion,
                                    sigma_diag, network):
        """Creates the Hamiltonian and sigma (nonlocal coupling
        deviation) maxtrices for the TLT supercell.

        :param num_uc_mols: Number of unit cell molecules.
        :param num_mols: Number of molecules in the TLT supercell.
        :param expansion: The expansion used to generate the
            TLT supercell.
        :param sigma_diag: The variance of the occupation energy.
        :param network: The coupling network of the crystal structure.
        :return: The TLT Hamiltonian and sigma.
        """
        hamiltonian = np.zeros((num_mols, num_mols))
        sigmas = np.zeros((num_mols, num_mols))

        a, b, c = expansion
        u = np.arange(a)
        v = np.arange(b)
        w = np.arange(c)
        for q, r, s in it.product(u, v, w):
            for uc_idx in range(num_uc_mols):
                i = mol_idx(q, r, s, uc_idx, a, b, c)
                for other, coupling, sigma, m, n, o in network[uc_idx]:
                    j = mol_idx(q + m, r + n, s + o, other, a, b, c)
                    # Matrix elements being assigned more then once
                    # can occur with small supercells or particularly
                    # long ranged electronic couplings, this is
                    # because of the periodic boundary conditions.
                    # This is ok except for the case when two
                    # assignments disagree on the value.
                    if hamiltonian[i, j] != 0.0 and \
                            abs(hamiltonian[i, j] - coupling) >= 0.001:
                        raise MobilityCalculationFailure(
                            f'Element in Hamiltonian has been assigned '
                            f'more than once with different '
                            f'values: {hamiltonian[i, j]} and {coupling}, '
                            f'try increasing the TLT supercell size.'
                        )
                    hamiltonian[i, j] = coupling
                    sigmas[i, j] = sigma

        if not np.allclose(hamiltonian, hamiltonian.T, atol=0.001, rtol=0):
            raise MobilityCalculationFailure(
                'Hamiltonian is not symmetric for some reason.'
            )
        # make all matrices exactly symmetric
        hamiltonian = (hamiltonian + hamiltonian.T) / 2
        sigmas = (sigmas + sigmas.T) / 2
        LOG.info('Size of the Hamiltonian: {}'.format(hamiltonian.shape))
        return np.triu(hamiltonian), \
               np.triu(sigmas) + np.diag([sigma_diag] * num_mols)

    def create_distance_matrices(self, unit_cell, uc_mols, num_mols,
                                 expansion):
        """Creates three distance matrices one for each cartesian
        component for all molecules of the TLT supercell.

        :param unit_cell: The unit cell.
        :param uc_mols: List of unit cell molecules.
        :param num_mols: Total number of molecules in the TLT supercell.
        :param expansion: The expansion used to generate the
            TLT supercell.
        :return: A tuple of the three distance matrices.
        """
        r_x = np.zeros((num_mols, num_mols))
        r_y = np.zeros((num_mols, num_mols))
        r_z = np.zeros((num_mols, num_mols))

        a, b, c = expansion
        h, k, l = unit_cell.lengths
        tlt_cell = UnitCell.from_lengths_and_angles(
            (a * h, b * k, c * l), unit_cell.angles
        )

        positions = np.zeros((num_mols, 3))
        h = np.arange(a)
        k = np.arange(b)
        l = np.arange(c)
        trans = np.array(list(it.product(h, k, l)))
        for tran in trans:
            for i, mol_i in enumerate(uc_mols):
                q, r, s = tran
                idx = mol_idx(q, r, s, i, a, b, c)
                r_i = tran @ unit_cell.direct + mol_i.centroid
                positions[idx, :] = r_i

        positions = tlt_cell.to_fractional(positions)
        for (i, s_i), (j, s_j) in it.product(
                enumerate(positions), repeat=2):
            if i > j:
                continue
            # Calculate distances between sites under periodic boundary
            # conditions using the minimum image convention - M. E.
            # Tuckerman. Statistical Mechanics: Theory and Molecular
            # Simulation Appendix B Eq. (B.9)
            s_ij = s_i - s_j
            s_ij = s_ij - np.round(s_ij)
            r_ij = tlt_cell.to_cartesian(s_ij)
            r_x[i, j] = r_ij[0]
            r_y[i, j] = r_ij[1]
            r_z[i, j] = r_ij[2]
            r_x[j, i] = -r_ij[0]
            r_y[j, i] = -r_ij[1]
            r_z[j, i] = -r_ij[2]

        return r_x, r_y, r_z

    def coupling_network(self, aom, crystal, temp, renorm):
        """Generate the coupling network of the crystal structure.

        :param aom: The AOM object used for calculating
            electronic couplings.
        :param crystal: The crystal structure.
        :param temp: The temperature of the calculation.
        :param renorm: The renormalisation factor that scales the
            mean and deviation of the electronic coupling.
        :return: A dictionary of the coupling network information.
        """
        uc_mols = crystal.unit_cell_molecules()
        act_mols, site_idxs, full_mol_idxs, depths = active_molecules(
            crystal, self.len_target, self.vdw_buffer)
        contacts = aom.molecule_close_contacts(act_mols)

        local_couplings = aom.calc_local_coupling(act_mols, contacts)
        if 'phonon_frequencies' in crystal.properties:
            nonlocal_couplings = aom.calc_nonlocal_coupling(
                crystal, depths, full_mol_idxs, contacts
            )
            freqs = crystal.properties['phonon_frequencies'][3:]

        network = defaultdict(list)
        for (uc_idx, uc_mol), (act_idx, act_mol) in it.product(
                enumerate(uc_mols), enumerate(act_mols)):

            if uc_idx == act_idx:
                continue

            if 'phonon_frequencies' in crystal.properties:
                sigma = np.sqrt(np.sum(
                    np.square(
                        wave_to_ev * freqs
                        * nonlocal_couplings[:, uc_idx, act_idx]
                    ) / np.tanh(freqs / (2 * k_b_cm * temp))
                )) * renorm
            else:
                sigma = 0

            trans = crystal.unit_cell.to_fractional(
                act_mol.centroid - uc_mols[site_idxs[act_idx]].centroid
            )

            network[uc_idx].append((
                site_idxs[act_idx], local_couplings[uc_idx, act_idx] * renorm,
                sigma, int(round(trans[0])), int(round(trans[1])),
                int(round(trans[2]))
            ))
        return network
