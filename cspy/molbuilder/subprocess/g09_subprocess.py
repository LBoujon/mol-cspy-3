import logging
import time
import re
import subprocess
from pathlib import Path
import psutil
import numpy as np
from cspy.molbuilder.exceptions import FormchkFailure
from cspy.molbuilder.exceptions import GaussianCalculationFailure


LOG = logging.getLogger(__name__)


class GaussianSub:

    def __init__(self, config):
        """Subprocess for calling Gaussian.

        :param config: config parser object
        """
        self.g09 = config['GAUSSIAN']['g09']
        self.formchk = config['GAUSSIAN']['formchk']
        self.memory = config['GAUSSIAN']['memory']
        self.cores = config['GAUSSIAN']['cores']
        self.method = config['GAUSSIAN']['method']
        self.basis_set = config['GAUSSIAN']['basis_set']

        self.g09_timeout = int(config['GAUSSIAN']['g09_timeout'])
        self.formchk_timeout = int(config['GAUSSIAN']['formchk_timeout'])

    def run_gaussian(self, coordinates, charge, multiplicity,
                     calculation_type, input_file_path, output_file_path,
                     chk_file_path=None, fchk_file_path=None):
        """Runs a Gaussian calculation.

        :param coordinates: coordinates of the molecule
        :param charge: charge of the molecule
        :param multiplicity: multiplicity of the molecule
        :param calculation_type: string that specifies the
            calculation type
        :param input_file_path: the file name of the input file to
            create
        :param output_file_path: the file name of the output file to
            create
        :param chk_file_path: the file name of the chk file to create
        :param fchk_file_path: the file name of the fchk file to create
        :return: a GaussianResults object
        """
        self.write_input_file(
            coordinates, charge, multiplicity, calculation_type,
            input_file_path, chk_file_path
        )

        # run the g09 calculation
        g09 = psutil.Popen([self.g09, input_file_path, output_file_path])

        start = time.time()
        LOG.debug('Started Gaussian calculation with input file %s',
                  input_file_path)

        try:
            return_code = g09.wait(timeout=self.g09_timeout)

        except psutil.TimeoutExpired as e:
            # pass the NoSuchProcess exception since we're going to
            # raise GaussianCalculationFailure later anyway
            for child in g09.children(recursive=True):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass

            try:
                g09.kill()
            except psutil.NoSuchProcess:
                pass

            raise GaussianCalculationFailure('Gaussian failure: %s', e)

        if return_code != 0:
            raise GaussianCalculationFailure(
                'Gaussian failure: exited with a non-zero return '
                'code {}'.format(return_code)
            )

        if not output_ended_with_normal_termination(output_file_path):
            raise GaussianCalculationFailure(
                'Gaussian failure: logs did not end with "Normal termination '
                'of Gaussian", check {} for details'.format(output_file_path)
            )

        # create the fchk file
        if chk_file_path is not None and fchk_file_path is not None:
            try:
                subprocess.run(
                    [self.formchk, chk_file_path, fchk_file_path],
                    check=True,
                    timeout=self.formchk_timeout
                )
            except Exception as e:
                raise FormchkFailure('Gaussian failure: %s', e)

        LOG.debug('Completed Gaussian calculation for input file %s'
                  ' in %ss', input_file_path, time.time() - start)

        return GaussianResults(output_file_path, fchk_file_path)

    def write_input_file(self, coordinates, charge, multiplicity,
                         calculation_type, input_file_path, chk_file_path):
        """Writes the input file for the Gaussian calculation.

        :param coordinates: coordinates of the molecule
        :param charge: charge of the molecule
        :param multiplicity: multiplicity of the molecule
        :param calculation_type: string that specifies the
            calculation type
        :param input_file_path: the file name of the input file to
            create
        :param chk_file_path: the file name of the chk file to create
        """

        calculation_types = ['sp', 'opt', 'freq', 'force']
        if calculation_type not in calculation_types:
            raise ValueError(
                'calculation_type should be one of the options: '
                '%s ' % calculation_types
            )

        with open(input_file_path, 'w') as com:

            if chk_file_path is not None:
                com.write('%chk={}\n'.format(chk_file_path))

            com.write('%NProcShared={}\n'.format(self.cores))
            com.write('%mem={}\n'.format(self.memory))

            if calculation_type == 'opt':
                com.write('# {}/{} NoSymm Opt Geom=PrintInputOrient'
                          '\n\n'.format(self.method, self.basis_set))
            elif calculation_type == 'sp':
                com.write('# {}/{} NoSymm Geom=PrintInputOrient'
                          '\n\n'.format(self.method, self.basis_set))
            elif calculation_type == 'freq':
                com.write('#P {}/{} NoSymm Opt=Tight Int=UltraFine '
                          'Freq(HPModes) Geom=PrintInputOrient'
                          '\n\n'.format(self.method, self.basis_set))
            elif calculation_type == 'force':
                com.write('#P {}/{} NoSymm Int=UltraFine Force '
                          'Geom=PrintInputOrient'
                          '\n\n'.format(self.method, self.basis_set))

            com.write('{}\n\n'.format(Path(input_file_path).stem))
            com.write(str(charge) + ' ' + str(multiplicity) + '\n')
            for coordinate in coordinates:
                com.write('{} {:.10f} {:.10f} {:.10f}\n'.format(*coordinate))
            com.write('\n\n')


class GaussianResults:

    def __init__(self, output_file_path, fchk_file_path=None):
        """GaussianResults object containing the output file path, fchk
        file path and the coordinates and energy from the gaussian
        calculation.

        :param output_file_path: output file path
        :param fchk_file_path: fchk file path
        """
        self.output_file_path = output_file_path
        self.fchk_file_path = fchk_file_path

        with open(self.output_file_path, 'r') as out:
            self.file = out.read()

        # set the coordinates of the molecule using the output
        self.coordinates = []
        search = re.findall(
            ' +([0-9]+) +([0-9]+) +0 +(-?[0-9]+[.][0-9]+) '
            '+(-?[0-9]+[.][0-9]+) +(-?[0-9]+[.][0-9]+)', self.file
        )
        for coordinate in search[-int(search[-1][0]):]:
            self.coordinates.append((
                int(coordinate[1]), float(coordinate[2]),
                float(coordinate[3]), float(coordinate[4])
            ))

        # set the energy using the values from the output
        self.energy = float(
            re.findall('SCF Done: +E[(].+[)] = (.+) +A[.]U[.]', self.file)[-1]
        )

    @property
    def vibrational_frequencies(self):
        """Returns vibrational frequencies from the gaussian output
        file in cm^-1.

        :return: list of vibrational frequencies ordered from low to
            high frequencies
        """
        freq_section = re.findall(
            ' *and normal coordinates(.*)Harmonic frequencies',
            self.file, flags=re.DOTALL
        )[0]

        freqs = []
        for line in freq_section.splitlines():
            if re.match(' +Frequencies --- ( +[0-9]+[.][0-9]+)+', line):
                num = len(re.findall('[0-9]+[.][0-9]+', line))
                if num != 1:
                    freqs += [float(i) for i in re.match(
                        ' +Frequencies ---' + ' +([0-9]+[.][0-9]+)' * num, line
                    ).group(*tuple(range(1, num + 1)))]
                else:
                    freqs.append(float(re.match(
                        ' +Frequencies ---' + ' +([0-9]+[.][0-9]+)' * num, line
                    ).group(1)))

        return np.array(freqs)

    @property
    def normal_modes(self):
        """Returns the normal modes for each vibrational mode.

        :return: np.array of normal modes ordered from low frequencies
            to high
        """
        num_atoms = int(re.findall('NAtoms= +([0-9]+)', self.file)[0])
        num_coord = num_atoms * 3

        freq_section = re.findall(
            ' *and normal coordinates(.*)Harmonic frequencies',
            self.file, flags=re.DOTALL
        )[0]

        normal_modes = []
        for line in freq_section.splitlines():

            if re.match(' +[1-3] +[0-9]+ +[0-9]+ '
                        '( +[-]?[0-9]+[.][0-9]+)+', line):
                num_columns = len(re.findall('[-]?[0-9]+[.][0-9]+', line))

                if num_columns != 1:
                    normal_modes.append([float(i) for i in re.match(
                        ' +[1-3] +[0-9]+ +[0-9]+' + ' +([-]?[0-9]+[.][0-9]+)'
                        * num_columns, line
                    ).group(*tuple(range(1, num_columns + 1)))])
                else:
                    normal_modes.append([float(re.match(
                        ' +[1-3] +[0-9]+ +[0-9]+' + ' +([-]?[0-9]+[.][0-9]+)'
                        * num_columns, line
                    ).group(1))])

        matrix = np.array(normal_modes[:num_coord])
        for i in range(1, (len(normal_modes) // num_coord)):
            matrix = np.concatenate((matrix, np.array(
                normal_modes[i * num_coord: (i + 1) * num_coord]
            )), axis=1)

        return matrix.T

    @property
    def reduced_masses(self):
        """Returns reduced masses from the gaussian output file.

        :return: list of reduced masses for each vibrational mode
            ordered from lowest frequency to high
        """
        freq_section = re.findall(
            ' *and normal coordinates(.*)Harmonic frequencies',
            self.file, flags=re.DOTALL
        )[0]

        masses = []
        for line in freq_section.splitlines():
            if re.match(' +Reduced masses --- ( +[0-9]+[.][0-9]+)+', line):
                num = len(re.findall('[0-9]+[.][0-9]+', line))
                if num != 1:
                    masses += [float(i) for i in re.match(
                        ' +Reduced masses ---' + ' +([0-9]+[.][0-9]+)'
                        * num, line
                    ).group(*tuple(range(1, num + 1)))]
                else:
                    masses.append(float(re.match(
                        ' +Reduced masses ---' + ' +([0-9]+[.][0-9]+)'
                        * num, line
                    ).group(1)))

        return np.array(masses)

    @property
    def forces(self):
        """Returns the forces on each atom in each cartesian coordinates.

        :return: np.array of forces on each atom
        """
        force_section = re.findall(
            ' *Center +Atomic +Forces [(]Hartrees/Bohr[)](.*)Cartesian Forces',
            self.file, flags=re.DOTALL
        )[0]
        force_section = re.findall(
            ' +[0-9]+ +[0-9]+' + ' +(-?[0-9]+[.][0-9]+)' * 3 + ' *',
            force_section
        )
        return np.array(force_section).astype(float)


def output_ended_with_normal_termination(output_file_path):
    """Check if the gaussian log file ended with normal termination.

    :param output_file_path: path of the gaussian output file
    :return: bool wheere the log files ended with normal termination
    """
    with open(output_file_path, 'r') as out:
        lines = out.read().splitlines()

    if ' '.join(lines[-1].split()[:4]) == 'Normal termination of Gaussian':
        return True
    else:
        return False
