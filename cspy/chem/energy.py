import numpy as np
from cspy.potentials import PotentialData
from itertools import combinations
import copy
from cspy.util.constants import NA, KE, J2EV, EV2KCAL, EV2KJ
import logging

LOG = logging.getLogger(__name__)

kB = 8.31446 * (10 ** -3) # Boltzmann constant in kJ mol^-1 K^-1
e = 2.7182818284 # Eulers constant

# at 1000 K, a 0 kJ/mol crystal has 3.3x the weight of a 10 kJ/mol crystal
# at 1000 K, a 0 kJ/mol crystal has 11x the weight of a 20 kJ/mol crystal
# at 1000 K, a 0 kJ/mol crystal has 1.7e+05x the weight of a 100 kJ/mol crystal
def boltzmann_weighting(energies, temperature=1000):
    """Function for calculating Boltzmann weights from energies.

    Parameters
    ----------
    energies : numpy array of floats
        Energies in kJ/mol
    temperature : float
        Temperature parameter. Higher temp = more tolerance for energy difference.

    Returns
    -------
    weights : numpy array of floats
    Boltzmann weights
    """

    # this might need some mechanism to weed out energetic anomalies (e.g. buckingham catastrophes)

    energies = np.asarray(energies)
    min_energy = np.min(energies)

    kBT = kB * temperature

    relative_energies = energies - min_energy
    Boltzman_factor = np.exp(relative_energies / kBT)
    weights = 1 / Boltzman_factor

    return weights


def boltzmann_weight_to_temperature(energies, target_weight):
    """Function for calculating temperature that would return a specified Boltzmann weight
    for the highest energy in an array

    Parameters
    ----------
    energies : numpy array of floats
        Energies in kJ/mol
    target_weight : float
        target weight that boltzmann_weight function would return if given the same energies
        and the temperature returned by this function

    Returns
    -------
    temperature : float
    Temperature
    """

    energies = np.asarray(energies)

    Boltzmann_factor = 1 / target_weight
    en_kBT = np.log(Boltzmann_factor)
    energy_dif = np.max(energies) - np.min(energies)

    kBT = energy_dif / en_kBT
    temperature = kBT / kB

    return temperature


# Buckingham potential takes form: V(r) = A×exp(- r/B) - C/r6
# r is an interatomic distance
# A, B, and C are fitable parameters

# energies differ slightly from DMACRYS ~0.005 kJ/mol
# not sure why this is but it is small enough to be a non-issue
class BuckinghamPotential:
    """Storage class for Buckingham potentials.

    Parameters
    ----------
    cluster : CSPy Molecule object
        A "molecule" with more than 1 component. 
        Energies will be calculated for interations between atoms in different components.
    potential : string
        Name of potential in cspy.potentials. Ext not needed
    dmafile : string
        Path to dmafile. Needed even if electrostatics are ignored
    charges : list of array_like
        List containing one numpy array for each component. 
        Each numpy array contains point charges on each atom in units of elementary charge
        If None, ignore charges
    Zp : int
        Z': used to for calculating energy per mol.
        Should be equal to the number of formula units.
        Assumed to be 1 if not specified.
    """

    def __init__(self, cluster, potential, atom_types, charges=None, Zp=1, foreshortenH=True, **kwargs):
        self.cluster = copy.deepcopy(cluster.components)
        self.potential = PotentialData(potential)
        self.buck = self.potential.buck
        self.anis = self.potential.anis
        self.dbuc = self.potential.dbuc
        self.atom_types = atom_types
        self.charges = charges
        self.Zp = Zp
        self.foreshortenH = foreshortenH
        self.energy = {'eV' : None,
                       'eV/atom' : None,
                       'kJ' : None,
                       'kJ/mol' : None,
                       'kcal' : None,
                       'kcal/mol' : None}
        
        # need to set this up to detect Williams pot
        # atm it's down to the user to turn it off
        if self.foreshortenH:
            for component in self.cluster:
                component.positions = component.foreshortened_hydrogen_positions()

    def calc_energy(self):
        """Calculate intermolecular potential energy between all molecules in cluster.
        """
        potential_energy = 0

        if len(self.cluster) < 2:
            LOG.error("Cannot calculate intermolecular energies for fewer than 2 molecules.\
                  \n Returning None")
            return {'eV' : None,
                       'eV/atom' : None,
                       'kJ' : None,
                       'kJ/mol' : None,
                       'kcal' : None,
                       'kcal/mol' : None}
        
        molecular_pairs = combinations(np.arange(len(self.cluster)), 2)
        for pair in molecular_pairs:
            mol0 = self.cluster[pair[0]]
            mol1 = self.cluster[pair[1]]

            mol0_pos = mol0.positions
            mol1_pos = mol1.positions

            types0 = self.atom_types[pair[0]]
            types1 = self.atom_types[pair[1]]

            r = []
            atom_pairs = []
            relevant_pair_types = []

            # iterate over atoms in mol0
            for ind, j in enumerate(mol0_pos):
                j_type = types0[ind]
                diff = j - mol1_pos
                # get distances with atoms in mol1
                dist = np.sqrt(np.einsum('ij,ij->i', diff, diff))
                r.append(dist)

                t = types0[ind]

                # also get types of pairs for each combination of atoms
                j_pairs = ['-'.join([j_type, k_type]) for k_type in types1]
                for j_pair in j_pairs:
                    if not j_pair in relevant_pair_types:
                        relevant_pair_types.append(j_pair)

                atom_pairs.append(j_pairs)

            # check if all the pair types exist in the potential file
            for pair_type_ind, pair_type in enumerate(relevant_pair_types):
                j_type, k_type = pair_type.split('-')
                try_reverse = False
                if j_type in self.buck.keys():
                    if k_type in self.buck[j_type].keys():
                        pass
                    else:
                        try_reverse = True
                else:
                    try_reverse = True

                # pair might be listed in the reverse order, so check and
                # swap the order if they are
                if try_reverse:
                    if k_type in self.buck.keys():
                        if j_type in self.buck[k_type].keys():
                            new_label = '-'.join([k_type, j_type])

                            for j_ind, j_pairs in enumerate(atom_pairs):
                                j_pairs = list(map(lambda x: x.replace(pair_type, new_label), j_pairs))
                                atom_pairs[j_ind] = j_pairs

                            relevant_pair_types[pair_type_ind] = new_label

                        else:
                            LOG.error("Potential missing for %s X %s", j_type, k_type)
                    else:
                        LOG.error("Potential missing for %s X %s", j_type, k_type)
                        LOG.error(self.buck.keys())
                        #print("Error: Potential missing for " + j_type + ' ' + k_type)

            r = np.asarray(r)

            r6 = np.power(r.flatten(), 6)
            r6 = np.reshape(r6, np.shape(r))

            for pair_type in relevant_pair_types:
                pair_type_mask = np.asarray(atom_pairs) == pair_type

                j_type, k_type = pair_type.split('-')
                buck_terms = self.buck[j_type][k_type]

                A = buck_terms[0]
                B = 1/buck_terms[1] #.pots reports 1/B, not B
                C = buck_terms[2]

                buck_energy = calc_pairwise_buck_energy(r[pair_type_mask], A, B, C, r6[pair_type_mask])
                potential_energy += buck_energy

            if self.charges:
                coloumb_energy = calc_pairwise_coloumb_energy(r, [self.charges[n] for n in pair])
                potential_energy += coloumb_energy

        self.energy['eV'] = potential_energy
        n_atoms = np.sum(np.array([len(component.positions) for component in self.cluster]))
        self.energy['eV/atom'] = potential_energy / n_atoms
        self.energy['kJ'] = potential_energy * EV2KJ
        self.energy['kcal'] = potential_energy * EV2KCAL
        n_mols = self.Zp
        self.energy['kJ/mol'] = (self.energy['kJ'] * NA) / n_mols
        self.energy['kcal/mol'] = (self.energy['kcal'] * NA) / n_mols


def calc_pairwise_buck_energy(r, A, B, C, r6=[]):
    """Calculate energy for specific buckingham interaction type.
    E.g. H_w1 and O_w1.

    Parameters
    ----------
    r : array_like
        Distances between atoms in pairwise interaction
        Each value is a seperate pairwise interaction
    A : float
        A value of potential
    B : float
        A value of potential
    C : float
        A value of potential
    r6 : array_like
        Distances to the sixth power between atoms in pairwise interaction
        Each value is a seperate pairwise interaction.
        If none, this is calculated.
    """

    if not len(r6)==0:
        r6 = np.power(r.flatten(), 6)
        r6 = np.reshape(r6, np.shape(r))

    repulsion = A * np.exp(- r * B)

    attraction = -C / r6

    energies = repulsion + attraction
    energy = np.sum(energies)

    return energy


def calc_pairwise_coloumb_energy(r, charges):
    """Calculate energy for pairwise coloumb interactions between 2 molecules.

    Parameters
    ----------
    r : array_like
        Distances between every atom in molecule 1 and every atom in molecule 2
    charges : list of array_like of floats
        List containing 2 numpy arrays: one for each molecule
        Each array contains charges for each atom
    """

    q1, q2 = charges

    energies = KE * ((np.multiply.outer(q1, q2)) * r / r ** 2) # untis are Nm (J)

    energy = np.sum(energies)
    energy = energy * J2EV     # convert to eV

    return energy