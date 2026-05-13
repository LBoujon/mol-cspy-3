import logging
import math
from rdkit import Chem
from rdkit.Chem.rdqueries import AtomNumEqualsQueryAtom
from cspy.cspympi import WorkQueue
from .gaussian_jobs import GaussianTaskTag


LOG = logging.getLogger(__name__)


class GaussianWorkQueue(WorkQueue):

    def __init__(self, mpi4py_workers, config):
        """Gaussian work queue object that handles some of the logic
        when submitting the gaussian fitness evaluation.

        :param config: config parser object
        """
        super().__init__(mpi4py_workers)
        self.start_job_tag = GaussianTaskTag.STAGE_1
        self.sleep_time = 5
        self.threshold = float(config['GAUSSIAN']['electron_affinity_target'])
        self.results_dict = {}

    @property
    def results(self):
        """Yield the results from the master for a worker that has
        completed its calculations.

        Returns
        -------
        tuple
            Yields a tuple of the results.
        """
        for task, (inchi, results) in self.master.results:

            if task == GaussianTaskTag.STAGE_1:
                self.results_dict[inchi] \
                    = GaussianFitnessResults(inchi, self.threshold)
                self.results_dict[inchi].stage_1_results = results
                self.prepend_job((
                    GaussianTaskTag.STAGE_2, (inchi, results.coordinates)
                ))
                self.append_job((
                    GaussianTaskTag.STAGE_3, (inchi, results.coordinates)
                ))

            if inchi not in self.results_dict:
                continue

            if task == GaussianTaskTag.STAGE_2:
                self.results_dict[inchi].stage_2_results = results
                self.append_job((
                    GaussianTaskTag.STAGE_4, (inchi, results.coordinates)
                ))
            elif task == GaussianTaskTag.STAGE_3:
                self.results_dict[inchi].stage_3_results = results
            elif task == GaussianTaskTag.STAGE_4:
                self.results_dict[inchi].stage_4_results = results
            elif task == GaussianTaskTag.FAILED:
                del self.results_dict[inchi]
                yield inchi, None
                continue

            fitness_results = self.results_dict[inchi]
            if fitness_results.all_complete:
                LOG.info('%s, reorganisation_energy: %s, '
                         'electron_affinity: %s, number_of_nitrogens: %s, '
                         'fitness: %s', inchi,
                         fitness_results.reorganization_energy,
                         fitness_results.electron_affinity,
                         fitness_results.number_of_nitrogens,
                         fitness_results.fitness)
                del self.results_dict[inchi]
                yield inchi, fitness_results.fitness


class GaussianFitnessResults:

    def __init__(self, inchi, threshold):
        """Gaussian fitness results object, this is used as a container
        for the fitness results as workers may run different parts of
        the gaussian fitness function calculation as different times.

        :param inchi: Inchi string of the molecule that the fitness is
            evaluated for.
        :param threshold: Electron affinity of ionisation energy
            threshold used in the fitness function.
        """
        self.inchi = inchi
        self.threshold = threshold
        self.number_of_nitrogens = len(
            Chem.MolFromInchi(inchi).GetAtomsMatchingQuery(
                AtomNumEqualsQueryAtom(7)
            ))
        self.stage_1_results = None
        self.stage_2_results = None
        self.stage_3_results = None
        self.stage_4_results = None

    @property
    def all_complete(self):
        """All results have been completed and assigned to the object.

        :return: Boolean whether all parts of the results exist.
        """
        if self.stage_1_results is None or self.stage_2_results is None \
        or self.stage_3_results is None or self.stage_4_results is None:
            return False
        else:
            return True

    @property
    def reorganization_energy(self):
        """Calculated the reorganization energy from the four results
        parameters.

        :return: The reorganization energy for this molecule in eV or
            None if all four stages of the results do not exist.
        """
        if self.all_complete:
            e_0_r_0 = self.stage_1_results.energy
            e_n_r_n = self.stage_2_results.energy
            e_n_r_0 = self.stage_3_results.energy
            e_0_r_n = self.stage_4_results.energy
            return (e_n_r_0 - e_0_r_0 + e_0_r_n - e_n_r_n) * 27.2113961
        else:
            return None

    @property
    def electron_affinity(self):
        """Calculated the electron affinity from the results parameters.

        :return: The electron affinity for this molecule in eV or
            None if all four stages of the results do not exist.
        """
        if self.all_complete:
            e_0_r_0 = self.stage_1_results.energy
            e_n_r_n = self.stage_2_results.energy
            return (e_0_r_0 - e_n_r_n) * 27.2113961
        else:
            return None

    @property
    def fitness(self):
        """Calculate the fitness of the molecule for the minimization
        of the root mean squared average of its reorganization energy
        and Schottky barrier with the solid-state electron affinity
        approximated with the gas phase isolated molecule values.

        :return: fitness value of the molecule
        """
        return math.sqrt(
            self.reorganization_energy**2
            + (self.threshold - self.electron_affinity)**2
        )
