import logging
import itertools as it
import numpy as np
from collections import defaultdict
from .dimer import active_molecules
from .diffusion import kubo_diffusion


LOG = logging.getLogger(__name__)


k_b_ev = 8.617333262145e-5
h_bar_ev = 6.582119569e-16
wave_to_ev = 299792458 * 100 * 6.62607015e-34 / 1.602176634e-19


class Marcus:

    def __init__(self, config):
        """Mobility calculation object using Marcus theory
        hopping rates.

        :param config: config parser object
        """
        self.len_target = float(config['MOBBUILDER']['len_target'])
        self.vdw_buffer = float(config['MOBBUILDER']['vdw_buffer'])

    def marcus_theory(self, coupling, reorg, delta_e, temp):
        """Calculate the Marcus theory hopping rate.

        :param coupling: The electronic coupling.
        :param reorg: The reorganisation energy.
        :param delta_e: The energy difference between the two states if
            hopping 1 <- 2 then it should be delta_e = E_2 - E_1.
        :param temp: Temperature.
        :return: The Marcus theory hopping rate.
        """
        rate = (coupling**2 / h_bar_ev) * np.sqrt(
            np.pi / (reorg * k_b_ev * temp)
        ) * np.exp(
            -(reorg - delta_e)**2 / (4 * reorg * k_b_ev * temp)
        )
        return rate

    def marcus_with_nonlocal(self, m, n, local_couplings, nonlocal_couplings,
                             phonon_freq, reorg, temp):
        """Calculate Marcus theory hopping rates with additional terms
        due to the the nonlocal coupling. This equation was derived
        by applying the polaron transform to the spin-boson model
        + nonlocal coupling and then applying the kubo formula as
        with marcus theory and taking all the boson occupations to the
        high temperature limit. This equation is similar to to Eq. 53
        in K. Hannewald and P. A. Bobbert Phys. Rev. B 69, 075212.

        :param m: Index of a molecule m
        :param n: Index of a molecule n
        :param local_couplings: np.array of local electronic couplings
        :param nonlocal_couplings: np.array of nonlocal
            electronic couplings
        :param phonon_freq: list of phonon freqs of the
            crystals structure.
        :param reorg: The reorganisation energy of the isolated molecule
        :param temp: The temperature.
        :return: The hopping rate.
        """
        coupling = local_couplings[m, n]
        freq_evs = wave_to_ev * phonon_freq

        rate = self.marcus_theory(coupling, reorg, 0, temp)
        for nonlocal_alpha, freq_ev in zip(nonlocal_couplings, freq_evs):
            rate += ((k_b_ev * temp / freq_ev) + 0.5) * self.marcus_theory(
                freq_ev * nonlocal_alpha[m, n], reorg, -freq_ev, temp
            )
            rate += ((k_b_ev * temp / freq_ev) - 0.5) * self.marcus_theory(
                freq_ev * nonlocal_alpha[m, n], reorg, freq_ev, temp
            )

        return rate

    def mobility(self, crystal, aom, reorg, temp):
        """Calculate the mobility using the Kubo formula with U(t) = 1 and
        Marcus theory rates.

        :param crystal: The crystal structure.
        :param aom: The AOM object used for calculating
            electronic couplings.
        :param reorg: Reorganisation energy.
        :param temp: Temperature.
        :return: Eigenvalues of the mobility tensor.
        """
        if crystal.space_group.symbol != 'P1':
            crystal = crystal.as_P1()

        uc_mols = crystal.unit_cell_molecules()
        act_mols, site_idxs, full_mol_idxs, depths = active_molecules(
            crystal, self.len_target, self.vdw_buffer)
        contacts = aom.molecule_close_contacts(act_mols)

        local_couplings = aom.calc_local_coupling(act_mols, contacts)
        LOG.info(f'Max Coupling: {np.max(np.abs(local_couplings))}')
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
                rate = self.marcus_with_nonlocal(
                    uc_idx, act_idx, local_couplings, nonlocal_couplings,
                    freqs, reorg, temp
                )
            else:
                rate = self.marcus_theory(
                    local_couplings[uc_idx, act_idx], reorg, 0, temp
                )

            centroid_i_j = (act_mol.centroid - uc_mol.centroid) * 1e-8
            network[uc_idx].append((
                site_idxs[act_idx], rate, centroid_i_j[0],
                centroid_i_j[1], centroid_i_j[2]
            ))

        mob_matrix = kubo_diffusion(network) / (k_b_ev * temp)
        mob_eig = np.linalg.eigh(mob_matrix)[0]
        LOG.info('Mobility %s: %s', crystal.titl, mob_eig)
        return mob_eig, mob_matrix
