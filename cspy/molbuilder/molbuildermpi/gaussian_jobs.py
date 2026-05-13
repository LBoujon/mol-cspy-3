import os
import logging
from enum import IntEnum
from pathlib import Path
from rdkit import Chem
from cspy.cspympi import Worker
from cspy.molbuilder.subprocess import coordinates_from_uff
from cspy.molbuilder.subprocess import GaussianSub
from cspy.molbuilder.subprocess import GaussianResults
from cspy.molbuilder.subprocess import output_ended_with_normal_termination
from cspy.molbuilder.exceptions import PropertyCalculationFailure
from cspy.molbuilder.exceptions import CarrierNotCorrectlyDefined


LOG = logging.getLogger(__name__)


class GaussianTaskTag(IntEnum):
    STAGE_1 = 1
    STAGE_2 = 2
    STAGE_3 = 3
    STAGE_4 = 4
    FAILED = 5


class GaussianWorker(Worker):

    def __init__(self, config):
        super().__init__()

        self.carrier = config['MOLBUILDER']['carrier']
        if self.carrier == 'hole':
            self.charge = 1
        elif self.carrier == 'electron':
            self.charge = -1
        else:
            raise CarrierNotCorrectlyDefined(
                'Carrier not correctly defined for reorganization '
                'energy calculation.'
            )
        self.gaussian_sub = GaussianSub(config)

    def calculate(self, job):
        """Calculate part of the fitness function. The part of the
        fitness function this method calculates is specified by a
        GaussianTaskTag enum contained in the job tuple. Additionally
        this method checks if any previous gaussian calculation were
        run and grabs the result from its output file. This useful for
        cases when the calculation had been restarted. This function
        expects a certain directory structure when restarting a
        calculation.

        :param job: A tuple containing task id and a tuple of the
            molecules inchi and coordinates. Coordinates are usually
            None if its the first step of the calculation.
        :return: A tuple containing the inchi of the molecule and
            the fitness, fitness is None if any part of the fitness
            evaluation had failed.
        """
        task, (inchi, coordinates) = job
        inchi_key = Chem.InchiToInchiKey(inchi)
        logs_dir = 'molbuilderlogs/' + inchi_key + '/reorganisation_energy'

        try:
            if task == GaussianTaskTag.STAGE_1:
                Path(logs_dir).mkdir(exist_ok=True, parents=True)

                molecule = Chem.AddHs(Chem.MolFromInchi(inchi))
                coordinates = coordinates_from_uff(molecule)

                results = self.get_gaussian_results(
                    coordinates, 0, 1, 'opt',
                    logs_dir + '/' + inchi_key + '_STAGE_1.com',
                    logs_dir + '/' + inchi_key + '_STAGE_1.log'
                )
                return GaussianTaskTag.STAGE_1, (inchi, results)

            if task == GaussianTaskTag.STAGE_2:
                results = self.get_gaussian_results(
                    coordinates, self.charge, 2, 'opt',
                    logs_dir + '/' + inchi_key + '_STAGE_2.com',
                    logs_dir + '/' + inchi_key + '_STAGE_2.log'
                )
                return GaussianTaskTag.STAGE_2, (inchi, results)

            if task == GaussianTaskTag.STAGE_3:
                results = self.get_gaussian_results(
                    coordinates, self.charge, 2, 'sp',
                    logs_dir + '/' + inchi_key + '_STAGE_3.com',
                    logs_dir + '/' + inchi_key + '_STAGE_3.log'
                )
                return GaussianTaskTag.STAGE_3, (inchi, results)

            if task == GaussianTaskTag.STAGE_4:
                results = self.get_gaussian_results(
                    coordinates, 0, 1, 'sp',
                    logs_dir + '/' + inchi_key + '_STAGE_4.com',
                    logs_dir + '/' + inchi_key + '_STAGE_4.log'
                )
                return GaussianTaskTag.STAGE_4, (inchi, results)

        except PropertyCalculationFailure as e:
            LOG.warning('Gaussian property calculation failed with '
                        'exception (%s) for molecule %s', e, inchi)
            return GaussianTaskTag.FAILED, (inchi, None)

    def get_gaussian_results(self, coordinates, charge, multiplicity,
                             calculation_type, input_file_path,
                             output_file_path):
        """Convenience function for running gaussian but with an
        additional step that checks if log files already exist from
        a previous evolutionary algorithm run.

        :param coordinates: Coordinates of the molecule.
        :param charge: Charge of the molecule.
        :param multiplicity: Multiplicity of the molecule.
        :param calculation_type: String that specifies the
            calculation type.
        :param input_file_path: The file name of the input file
            to create.
        :param output_file_path: The file name of the output file
            to create.
        :return: A GaussianResults object.
        """
        if os.path.exists(output_file_path) \
                and output_ended_with_normal_termination(output_file_path):
            return GaussianResults(output_file_path)

        if os.path.exists(input_file_path):
            os.remove(input_file_path)
        if os.path.exists(output_file_path):
            os.remove(output_file_path)

        return self.gaussian_sub.run_gaussian(
            coordinates, charge, multiplicity, calculation_type,
            input_file_path, output_file_path
        )
