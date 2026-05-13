import ast
import logging
import re
import time
from random import choice
from random import uniform
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from cspy.molbuilder.addition import Generator
from cspy.molbuilder.crossover import Crossover
from cspy.molbuilder.exceptions import MoleculeModificationFailure
from cspy.molbuilder.mutation import Mutator


LOG = logging.getLogger(__name__)


class Generational:

    def __init__(self, work_queue, config):
        """Population class for the generational type evolutionary
        algorithm.

        :param work_queue: the type of work queue for the population
        :param config: config parser object
        """
        self.work_queue = work_queue

        mol_smiles = ast.literal_eval(config['MOLBUILDER']['molecules'])
        frag_smiles = ast.literal_eval(config['MOLBUILDER']['fragments'])
        type0_frag_smarts \
            = ast.literal_eval(config['MOLBUILDER']['mutations_1'])
        type1_frag_smarts \
            = ast.literal_eval(config['MOLBUILDER']['mutations_2'])
        min_size, max_size \
            = ast.literal_eval(config['MOLBUILDER']['molsize'])

        if 'protected' in config['MOLBUILDER']:
            protected_smarts \
                = ast.literal_eval(config['MOLBUILDER']['protected'])
        else:
            protected_smarts = []

        self.generator = Generator(mol_smiles, frag_smiles, min_size, max_size)
        self.mutator = Mutator(
            type0_frag_smarts, type1_frag_smarts, protected_smarts)
        self.crossover = Crossover(min_size, max_size)

        self.population_size \
            = int(config['MOLBUILDER']['population_size'])
        self.initial_number_mutations \
            = int(config['MOLBUILDER']['initial_number_mutations'])
        self.elitism_population_size \
            = int(config['MOLBUILDER']['elitism_population_size'])
        self.mutation_rate \
            = float(config['MOLBUILDER']['mutation_rate'])
        self.recombination_rate \
            = float(config['MOLBUILDER']['recombination_rate'])
        self.tournament_win_rate \
            = float(config['MOLBUILDER']['tournament_win_rate'])
        self.maximize \
            = ast.literal_eval(config['MOLBUILDER']['maximize'])
        self.total_generations \
            = int(config['MOLBUILDER']['total_generations'])
        self.calculation_type = config['MOLBUILDER']['calculation_type']

        if (self.population_size - self.elitism_population_size) % 2 != 0:
            self.work_queue.terminate_workers()
            raise ValueError('MolBuilder population size minus elitism '
                             'population size needs to be even.')

        if self.maximize:
            self.reject_value = -np.inf
        else:
            self.reject_value = np.inf

        self.current_generation = 1
        self.generations = []
        self.population = []
        self.sampled_molecules = set()
        self.failed_molecules = set()
        self.max_fitness = []
        self.avg_fitness = []
        self.min_fitness = []
        self.calculated_properties = {}

        self.config = config
        if self.config.has_section('RESTART'):
            LOG.info('restarting calculations from config file')
            self.setup_from_restart()
        else:
            self.config.add_section('RESTART')

    def optimize(self):
        """Run the evolutionary algorithm.

        """
        self.init_population()
        self.draw_all_images()
        while self.current_generation < self.total_generations:
            self.set_next_population()
            self.draw_all_images()
        self.work_queue.terminate_workers()

    def init_population(self):
        """Initialize the population by first generating a list of
        random molecules until the desired population size is reached
        and evaluate the fitness of the population. If starting from a
        restart point the current population is reevaluated to calculate
        the remaining molecule fitness'.

        """
        while len(self.population) < self.population_size:

            try:
                molecule = self.generator.create_molecule()
                molecule = self.mutator.multiple_mutations(
                    molecule, self.initial_number_mutations)
            except MoleculeModificationFailure as e:
                LOG.error('%s exception caught while randomly generating a '
                          'molecule, retrying', e)
                continue

            LOG.debug('generated molecule: %s', Chem.MolToSmiles(molecule))
            self.population.append(Chem.MolToInchi(molecule, options='-SNon'))

        self.generations.append(self.population)
        self.write_restart_config()
        self.calculate_fitness()

    def calculate_fitness(self):
        """Calculate the fitness of the current population and stored
        the data into the calculated_properties dictionary.

        """
        # add calculation to be done to the work_queue
        for inchi in self.population:
            job = (self.work_queue.start_job_tag, (inchi, None))
            if inchi in self.calculated_properties or job in self.work_queue:
                continue
            self.work_queue.append_job(job)

        # keeping running if there are jobs to do or if there are
        # completed results to gather
        while not self.work_queue.done:

            self.work_queue.do_work()
            LOG.debug('MPI idle: %s', sorted(self.work_queue.idle))
            LOG.debug('MPI running: %s', sorted(self.work_queue.running))
            LOG.debug('jobs in queue: %s', self.work_queue.num_jobs)
            time.sleep(self.work_queue.sleep_time)

            for inchi, result in self.work_queue.results:
                if result is None:
                    self.calculated_properties[inchi] = self.reject_value
                else:
                    self.calculated_properties[inchi] = result

                self.update_molecule_data(inchi)
                self.write_restart_config()

        self.update_fitness_data(self.population)

        LOG.info('failed molecules: %s', self.failed_molecules)
        LOG.info('generation-%s: %s', self.current_generation,
                 [Chem.MolToSmiles(Chem.MolFromInchi(i))
                  for i in self.sort_population(self.population)])

    def set_next_population(self):
        """Creates the next generation and evaluates the populations
        fitness.

        """
        child_population = self.sort_population(
            self.population)[:self.elitism_population_size]

        while len(child_population) < self.population_size:
            parent_0 = self.tournament_selection(
                choice(self.population), choice(self.population)
            )
            parent_1 = self.tournament_selection(
                choice(self.population), choice(self.population)
            )

            child_0, child_1 = self.create_children(parent_0, parent_1)
            child_population.append(child_0)
            child_population.append(child_1)

        self.population = child_population
        self.current_generation += 1
        self.generations.append(self.population)
        self.write_restart_config()
        self.calculate_fitness()

    def tournament_selection(self, inchi_i, inchi_j):
        """Implementation of the two way tournament selection.

        :param inchi_i: inchi of one of the molecule to undergo
            tournament selection
        :param inchi_j: inchi of one of the molecule to undergo
            tournament selection
        :return: the winning inchi
        """
        property_i = self.calculated_properties[inchi_i]
        property_j = self.calculated_properties[inchi_j]

        if property_i == property_j:
            return choice([inchi_i, inchi_j])

        select_fit = self.tournament_win_rate >= uniform(0, 1)

        if self.maximize:
            if (property_i > property_j and select_fit
            or property_j > property_i and not select_fit):
                return inchi_i
            if (property_j > property_i and select_fit
            or property_i > property_j and not select_fit):
                return inchi_j
        else:
            if (property_i < property_j and select_fit
            or property_j < property_i and not select_fit):
                return inchi_i
            if (property_j < property_i and select_fit
            or property_i < property_j and not select_fit):
                return inchi_j

    def create_children(self, parent_inchi_i, parent_inchi_j):
        """Using two parent molecule inchis create two child molecules
        and return their inchis.

        :param parent_inchi_i: inchi of one of the parent molecules
        :param parent_inchi_j: inchi of one of the parent molecules
        :return: a tuple of the inchis of the two child molecules
        """
        child_i = Chem.MolFromInchi(parent_inchi_i)
        child_j = Chem.MolFromInchi(parent_inchi_j)

        try:
            child_i, child_j = self.crossover.crossover(child_i, child_j)
        except MoleculeModificationFailure as e:
            LOG.error('%s exception caught while generating children from '
                      'parents %s and %s, using parents for the '
                      'child molecules', e, Chem.MolToSmiles(child_i),
                      Chem.MolToSmiles(child_j))

        if self.mutation_rate >= uniform(0, 1):
            try:
                child_i = self.mutator.mutation(child_i)
            except MoleculeModificationFailure as e:
                LOG.error('%s exception caught while mutating child molecules '
                          '%s, leaving child molecule unmutated', e,
                          Chem.MolToSmiles(child_i))

        if self.mutation_rate >= uniform(0, 1):
            try:
                child_j = self.mutator.mutation(child_j)
            except MoleculeModificationFailure as e:
                LOG.error('%s exception caught while mutating child molecules '
                          '%s, leaving child molecule unmutated', e,
                          Chem.MolToSmiles(child_j))

        if self.recombination_rate >= uniform(0, 1):
            try:
                child_i = self.mutator.recombination(child_i)
            except MoleculeModificationFailure as e:
                LOG.error('%s exception caught while recombining child '
                          'molecules %s, leaving child molecule unmodified',
                          e, Chem.MolToSmiles(child_i))

        if self.recombination_rate >= uniform(0, 1):
            try:
                child_j = self.mutator.recombination(child_j)
            except MoleculeModificationFailure as e:
                LOG.error('%s exception caught while recombining child '
                          'molecules %s, leaving child molecule unmodified',
                          e, Chem.MolToSmiles(child_j))

        return (Chem.MolToInchi(child_i, options='-SNon'),
                Chem.MolToInchi(child_j, options='-SNon'))

    def setup_from_restart(self):
        """Sets up the population object from restart options in the
        config.ini.

        """
        self.calculated_properties = ast.literal_eval(re.sub(
            r'-?inf', 'None', self.config['RESTART']['calculated_properties']
        ))

        for key, value in self.calculated_properties.items():
            if value is None:
                self.calculated_properties[key] = self.reject_value

        self.current_generation \
            = int(self.config['RESTART']['current_generation'])
        self.generations \
            = ast.literal_eval(self.config['RESTART']['generations'])
        self.population = self.generations.pop()

        for i, generation in enumerate(self.generations):

            self.update_fitness_data(generation)
            for inchi in generation:
                self.update_molecule_data(inchi)

            self.draw_fittest(generation, 'generation-' + str(i + 1))
            self.draw_fittest(self.sampled_molecules, 'sampled-' + str(i + 1))

    def write_restart_config(self):
        """Writes a restart.ini file which is a config.ini with a
        additional restart section for restarting the calculation for
        the current point.

        """
        self.config['RESTART']['current_generation'] \
            = str(self.current_generation)
        self.config['RESTART']['generations'] \
            = str(self.generations)
        self.config['RESTART']['calculated_properties'] \
            = str(self.calculated_properties)

        with open('restart.ini', 'w') as restart:
            self.config.write(restart)

    def sort_population(self, molecules):
        """Convenience function that sorts and returns a list of
        molecules from best to worst.

        :param molecules: a list of molecules

        """
        return sorted(molecules, key=lambda i: self.calculated_properties[i],
                      reverse=self.maximize)

    def update_molecule_data(self, inchi):
        """Add the inchi to the self.sampled_molecules and
        self.failed_molecules set.

        :param inchi: inchi of the molecules to add to the

        """
        if self.calculated_properties[inchi] == self.reject_value:
            self.failed_molecules.add(inchi)
        else:
            self.sampled_molecules.add(inchi)

    def update_fitness_data(self, molecules):
        """Updates the self.max_fitness, self.avg_fitness and
        self.min_fitness using the fitness of the input list of
        molecules.

        :param molecules: a list of molecules

        """
        fitness_list = []
        for fitness in [self.calculated_properties[i] for i in molecules]:
            if fitness != np.inf and fitness != -np.inf:
                fitness_list.append(fitness)

        self.max_fitness.append(max(fitness_list))
        self.avg_fitness.append(sum(fitness_list) / len(fitness_list))
        self.min_fitness.append(min(fitness_list))

        LOG.info('max fitness: %s', self.max_fitness)
        LOG.info('avg fitness: %s', self.avg_fitness)
        LOG.info('min fitness: %s', self.min_fitness)

    def draw_all_images(self):
        """Draw the fittest molecules of the sampled set and population
        list and plot the fitness over time.

        """
        self.plot_fitness()
        self.draw_fittest(
            self.population,
            'generation-' + str(self.current_generation)
        )
        self.draw_fittest(
            self.sampled_molecules,
            'sampled-' + str(self.current_generation)
        )

    def draw_fittest(self, molecules, name, num=20):
        """Draws a chemical diagram of the best molecules in the
        molecules list.

        :param molecules: list of molecules to draw
        :param name: name of the file
        :param num: number of molecules to draw

        """
        molecules = [
            Chem.MolFromInchi(i) for i in self.sort_population(molecules)[:num]
        ]
        [AllChem.Compute2DCoords(i) for i in molecules]

        LOG.info('fittest %s: %s', name,
                 [Chem.MolToSmiles(i) for i in molecules])

        img = Draw.MolsToGridImage(
            molecules, molsPerRow=5, subImgSize=(300, 300), legends=[str(
                self.calculated_properties[Chem.MolToInchi(i, options='-SNon')]
            ) for i in molecules]
        )

        img.save('molbuilderimages/' + name + '.png')

    def plot_fitness(self):
        """Plots a graph of fitness over generation using
        self.max_fitness, self.avg_fitness and self.min_fitness.

        """
        plt.plot(np.array(self.max_fitness), label="max fitness")
        plt.plot(np.array(self.avg_fitness), label="avg fitness")
        plt.plot(np.array(self.min_fitness), label="min fitness")
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend()
        plt.tight_layout()
        plt.savefig('molbuilderimages/fitness.png')
        plt.close()
