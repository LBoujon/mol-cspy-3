import os

from read_n2p2 import *
from pathlib import Path
import numpy as np
from read_n2p2 import input_data_reader, get_ids_dict
import subprocess
import time
from vasp_to_n2p2 import Structure
from atom import atoms_dict
from random import sample
from subprocess import call


def mean_std(data):
    return np.mean(np.std(data, axis=0))


def update_datasets(candidates_data, training_set, calculated_snapshots):
    '''
    candidates_data: input.data file of candidates pool.
    training_set: input.data file of current training set.
    calculated_snapshots: a dictionary containing the Snapshots of the selected data points
    '''

    candidate_snapshots = input_data_reader(candidates_data)
    current_training_snapshots = input_data_reader(training_set)
    os.rename(
        candidates_data,
        f'{os.path.dirname(candidates_data)}/old_candidates.data'
    )
    os.rename(
        training_set,
        f'{os.path.dirname(training_set)}/old_training_set.data'
    )

    for id_ in current_training_snapshots.keys():
        current_training_snapshots[id_].write(
            f'{os.path.dirname(training_set)}/training_set.data'
        )

    for id_ in calculated_snapshots:
        calculated_snapshots[id_].write(
            f'{os.path.dirname(training_set)}/training_set.data'
        )

    for id_ in candidate_snapshots.keys():
        if id_ in calculated_snapshots.keys():
            continue
        candidate_snapshots[id_].write(
            f'{os.path.dirname(candidates_data)}/candidates.data'
        )


def from_file(fname, binary=False):
    """
    Read the contents of a file into a variable.
    By default, the file will be read as a text file, resulting in a string.
    If `binary` is true, it will be read as a binary file, resulting in bytes.
    """

    mode = 'r'
    if binary:
        mode += 'b'
    with open(fname, mode) as f_in:
        data = f_in.read()
    return data


def to_file(data, output_fname, binary=False, verbose=False):
    """
    Write a variable to a file.
    The provided `data` would typically be a string or bytes, if `binary` is true.
    The output file name is protected against overwriting and if `verbose is true,
    backup file creation will be reported.
    """

    mode = 'w'
    if binary:
        mode += 'b'
    with open(output_fname, mode) as f_out:
        f_out.write(data)


def next_iteration(iteration, candidates_data, training_data, n_committee_members=8, n_epoch=20,
                   output_folder='.', input_template='input.nn', ncpu=2):
    homedir = os.path.realpath(output_folder)
    training_data = os.path.realpath(training_data)
    candidates_data = os.path.realpath(candidates_data)
    input_template = os.path.realpath(input_template)

    for i in range(n_committee_members):

        train_dir = f'{homedir}/iteration-{iteration:03d}/train-{i:02d}'
        predict_dir = f'{homedir}/iteration-{iteration:03d}/predict-{i:02d}'
        valid_dir = f'{homedir}/iteration-{iteration:03d}/valid-{i:02d}'

        os.makedirs(train_dir, exist_ok=True)
        snapshots = input_data_reader(training_data)
        s_temp = list(snapshots.values())[0]
        elements = [element for element in s_temp.elements_group.keys()]
        str_elements = ' '.join(elements)
        n_elements = len(elements)

        template_data = from_file(input_template)
        str_input = template_data.format(
            n_elements=n_elements,
            elements=str_elements,
            seed=i,
            n_epoch=n_epoch,
        )
        to_file(str_input, f'{train_dir}/input.nn')
        os.system(f'cp {training_data} {train_dir}/input.data')
        os.chdir(train_dir)
        os.system(f'nnp-scaling 100')
        os.system(f'mpiexec -n {ncpu} nnp-train > train.log 2>&1')

        # Prediction on candidates
        best_epoch = n_epoch  # replace it with the fancier function
        os.makedirs(predict_dir, exist_ok=True)
        for element in elements:
            os.system(
                f'cp {train_dir}/weights.{atoms_dict[element]:03d}.{best_epoch:06d}.out {predict_dir}/weights.{atoms_dict[element]:03d}.data')
        os.system(f'cp {train_dir}/input.nn {train_dir}/scaling.data {predict_dir}')
        os.system(f'cp {candidates_data} {predict_dir}/input.data')
        os.chdir(predict_dir)
        os.system(f'mpiexec -n {ncpu} nnp-dataset 0')
        os.system('rm -rf input.data output.data nnp-dataset.log.*')

        # Prediction on training_set
        os.makedirs(valid_dir, exist_ok=True)
        for element in elements:
            os.system(
                f'cp {train_dir}/weights.{atoms_dict[element]:03d}.{best_epoch:06d}.out {valid_dir}/weights.{atoms_dict[element]:03d}.data')
        os.system(f'cp {train_dir}/input.nn {train_dir}/scaling.data {valid_dir}')
        os.system(f'cp {training_data} {valid_dir}/input.data')
        os.chdir(valid_dir)
        os.system(f'mpiexec -n {ncpu} nnp-dataset 0')
        os.system('rm -rf input.data output.data nnp-dataset.log.*')
        os.chdir(homedir)


def reference_calc_on_new_datapoints(selected_fname, candidates_data_fname, submission_folder='.', kspacing=0.05,
                                     ncpu=40, submit=False):
    '''
    selected_fname: A file containing the ID of the selected structures in the first column.
    candidates_data_fname: input.data file that contain the structure of candidate structures in n2p2 format.
    submission_folder: The folder to which VASP calculations will be done in.
    kspacing:  kspacing for sampling kpoints, default=0.05
    ncpu: number of CPUs for determining parallelization parameters in INCAR and to submit job.
    submit: Boolean. Want to submit the jobfile?
    '''
    selected_ids = []
    with open(selected_fname, 'r') as f:
        for line in f.readlines():
            if line[0] == '#':
                continue
            parts = line.split()
            selected_ids.append(parts[0])

    candidate_snapshots = input_data_reader(candidates_data_fname)

    being_calculated_snapshots = []
    for id_ in selected_ids:
        s = candidate_snapshots[id_]
        s.generate_vasp_inputs(
            output_folder=f'{submission_folder}/{s.name}',
            kspacing=kspacing,
            ncpu=ncpu,
            submit=submit
        )

        being_calculated_snapshots.append(s)

    calculated_snapshots = []
    for s in being_calculated_snapshots:
        check_command = f'qstat | grep {s.job_id} | wc -l'
        while subprocess.check_output(check_command, shell=True):
            time.sleep(60)
        temp_s = Structure()

        temp_s.read_structure(
            poscar=f'{s.vasp_folder}/POSCAR',
            outcar=f'{s.vasp_folder}/OUTCAR'
        )
        calculated_snapshots.append(temp_s)


def qbc(candidates_data, training_set, iteration_folder, score='force'):
    '''
    candidates: candidates input.data file which will be used to generate candidates_ids_dict.
    training_set: input.data file which will be used to generate training_ids_dict
    iteration_folder: The folder containing train-xx, valid-xx and predict-xx folders.
    stats_fname: The file in which statistics of QbC are written into.
    '''
    prediction_dirs = Path(iteration_folder).glob('predict-*')
    validation_dirs = Path(iteration_folder).glob('valid-*')
    candidates_ids_dict = get_ids_dict(candidates_data)
    training_ids_dict = get_ids_dict(training_set)

    if score == 'energy':
        energy_predictions_dict = {}
        for i_model, dir in enumerate(sorted(prediction_dirs)):
            print(i_model, dir)
            energy_predictions_dict[i_model] = read_energy_prediction(f'{dir}/energy.comp')
            # This part of the code is incomplete...

    if score == 'force':
        # Candidates
        prediction_force_predictions_dict = {}
        for i_model, dir in enumerate(sorted(prediction_dirs)):
            print(i_model, dir)
            prediction_force_predictions_dict[i_model] = read_force_predictions(f'{dir}/forces.comp')

        predictions_mean_stds_list = []
        for id_ in candidates_ids_dict.keys():
            data = []
            for i_model in prediction_force_predictions_dict.keys():
                data.append(prediction_force_predictions_dict[i_model][id_]['nnp'])
            data = np.array(data)
            predictions_mean_stds_list.append([id_, mean_std(data)])

        # Training Set
        valid_force_predictions_dict = {}
        for i_model, dir in enumerate(sorted(validation_dirs)):
            print(i_model, dir)
            valid_force_predictions_dict[i_model] = read_force_predictions(f'{dir}/forces.comp')

        validation_mean_stds_list = []
        for id_ in training_ids_dict.keys():
            data = []
            for i_model in valid_force_predictions_dict.keys():
                data.append(valid_force_predictions_dict[i_model][id_]['nnp'])
            data = np.array(data)
            validation_mean_stds_list.append([id_, mean_std(data)])

    predictions_mean_stds = np.array(predictions_mean_stds_list)
    validation_mean_stds = np.array(validation_mean_stds_list)

    sorted_by_mean_std = predictions_mean_stds[predictions_mean_stds[:, 1].argsort()][::-1]

    candidates_mean_std = np.mean(predictions_mean_stds[:, 1])
    candidates_max_std = np.max(predictions_mean_stds[:, 1])
    trainingset_mean_std = np.mean(validation_mean_stds[:, 1])
    trainingset_max_std = np.max(validation_mean_stds[:, 1])
    selected_mean_std = np.mean(sorted_by_mean_std[:20, 1])

    finished = False
    if candidates_max_std < trainingset_mean_std:
        finished = True

    selected_ids = []
    selected_output_fname = f'{iteration_folder}/selected_ids.txt'
    with open(selected_output_fname, 'w') as f:
        f.write('# 1.ID  2.STD\n')
        for id_, std in sorted_by_mean_std[:20]:
            selected_ids.append(candidates_ids_dict[id_])
            f.write(f'{candidates_ids_dict[id_]}  {std}\n')

    stats_fname = f'{iteration_folder}/stats.txt'
    with open(stats_fname, 'w') as f:
        f.write(f'MEAN STD over candidates pool: {candidates_mean_std}\n')
        f.write(f'MAX STD over candidates pool: {candidates_max_std}\n')
        f.write(f'MEAN STD over training set: {trainingset_mean_std}\n')
        f.write(f'MAX STD over training set: {trainingset_max_std}\n')
        f.write(f'MEAN STD over chosen structures: {selected_mean_std}\n')

    return selected_ids, finished


class QbC:
    def __init__(self, candidates_data, training_set, n_initial, n_model, n_add, n_epoch=20,
                 n_cpu=1, input_nn_template='input.nn', restart_iter=None, working_dir='.'):
        """
        Parameters
        ----------
        candidates_data: str
            Filename of candidate structures in n2p2 (input.data) format
        training_set: str
            Filename of training set structures in n2p2 (input.data) format
        n_initial: int
            Number of initial structures to start QbC with.
        n_model: int
            Number of members of committee
        n_add: int
            Number of structures added to the training set at each iteration
        restart_iter: int
            Iteration at which QbC should be restarted.
        n_cpu: int
            Number of CPUs that will be used for training and reference data calculations.
        input_nn_template: str
            An input.nn template file that will be used to make input.nn files needed for training of committee members.
            Architecture and other "fixed" NN parameters should be adjusted in the template file.
        n_epoch: int
            Number of epochs for training the neural network
        """
        self.n_initial = n_initial
        self.n_model = n_model
        self.n_add = n_add
        self.n_epoch = n_epoch
        self.n_cpu = n_cpu
        self.working_dir = os.path.realpath(working_dir)
        self.dataset_dir = f'{self.working_dir}/datasets'
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
        self.input_nn_template = os.path.realpath(input_nn_template)
        self.candidates_data = os.path.realpath(candidates_data)
        if not os.path.exists(self.candidates_data):
            raise Exception(f'{self.candidates_data} doesn\'t exit.')

        self.iteration = 1

        try:
            temp_training_set = os.path.realpath(training_set)
        except NameError:
            temp_training_set = None

        if restart_iter:
            self.iteration = restart_iter
            temp_training_set = f'{self.working_dir}/iteration-{restart_iter:03d}/train-00/input.data'

        # candidates.data and training_set.data are to be in the same directory.
        self.training_set = f'{self.dataset_dir}/training_set.data'

        if not os.path.exists(temp_training_set) or not temp_training_set:
            print(f'{training_set} cannot be found!')
            print('Random sampling from candidates will be done...')
            self.iteration = 1
            self.sample_initial_training_set(output_fname=self.training_set)
        else:
            call(f"cp {temp_training_set} {self.training_set}", shell=True)

    def sample_initial_training_set(self, output_fname):
        candidate_snapshots = input_data_reader(self.candidates_data)
        sampled_ids = sample(list(candidate_snapshots.keys()), k=self.n_initial)
        for id_ in sampled_ids:
            s = candidate_snapshots[id_]
            s.write(output_fname)

    def update_datasets(self, calculated_snapshots):
        """
        Takes snapshots of calculated new (selected) data points and updates candidates.data and training_set.data

        Parameters
        ----------
        calculated_snapshots: dict
        A dictionary containing the Snapshot objects of the vasp calculated selected data points.
        """
        candidate_snapshots = input_data_reader(self.candidates_data)
        current_training_snapshots = input_data_reader(self.training_set)
        os.rename(
            self.candidates_data,
            f'{self.dataset_dir}/old_candidates.data'
        )
        os.rename(
            self.training_set,
            f'{self.dataset_dir}/old_training_set.data'
        )

        for id_ in current_training_snapshots.keys():
            current_training_snapshots[id_].write(
                f'{self.dataset_dir}/training_set.data'
            )

        for id_ in calculated_snapshots:
            calculated_snapshots[id_].write(
                f'{self.dataset_dir}/training_set.data'
            )

        for id_ in candidate_snapshots.keys():
            if id_ in calculated_snapshots.keys():
                continue
            candidate_snapshots[id_].write(
                f'{self.dataset_dir}/candidates.data'
            )

    def train_member(self, elements, i_member, train_dir):
        os.makedirs(train_dir, exist_ok=True)
        str_elements = ' '.join(elements)
        n_elements = len(elements)
        template_data = from_file(self.input_nn_template)
        str_input = template_data.format(
            n_elements=n_elements,
            elements=str_elements,
            seed=i_member,
            n_epoch=self.n_epoch,
        )
        to_file(str_input, f'{train_dir}/input.nn')
        os.system(f'cp {self.training_set} {train_dir}/input.data')
        os.chdir(train_dir)
        os.system(f'nnp-scaling 100')
        os.system(f'mpiexec -n {self.ncpu} nnp-train > train.log 2>&1')

    @property
    def elements_list(self):
        snapshots = input_data_reader(self.training_set)
        # reading elements information from the first snapshot of training_set
        s_temp = list(snapshots.values())[0]
        elements = [element for element in s_temp.elements_group.keys()]
        return elements

    def prediction_by_member(self, elements, train_dir, best_epoch, predict_dir, prediction_type):
        """

        Parameters
        ----------
        elements: list
        train_dir:str
        best_epoch: int
        predict_dir: str
        prediction_type: str
            'validation' for validation purposes: uses training_set.data to make predictions.
            'candidates' for sampling purposes: uses candidates.data to make predictions.
        """
        os.makedirs(predict_dir, exist_ok=True)
        for element in elements:
            os.system(
                f'cp {train_dir}/weights.{atoms_dict[element]:03d}.{best_epoch:06d}.out {predict_dir}/weights.{atoms_dict[element]:03d}.data')
        os.system(f'cp {train_dir}/input.nn {train_dir}/scaling.data {predict_dir}')
        if prediction_type == 'candidates':
            os.system(f'cp {self.candidates_data} {predict_dir}/input.data')
        elif prediction_type == 'validation':
            os.system(f'cp {self.training_set} {predict_dir}/input.data')
        os.chdir(predict_dir)
        os.system(f'mpiexec -n {self.ncpu} nnp-dataset 0')
        os.system('rm -rf input.data output.data nnp-dataset.log.*')

    def next_iteration(self):
        self.iteration += 1

        # elements = self.elements_list
        for i_member in range(self.n_model):
            train_dir = f'{self.working_dir}/iteration-{self.iteration:03d}/train-{i_member:02d}'
            predict_dir = f'{self.working_dir}/iteration-{self.iteration:03d}/predict-{i_member:02d}'
            valid_dir = f'{self.working_dir}/iteration-{self.iteration:03d}/valid-{i_member:02d}'

            self.train_member(
                elements=self.elements,
                i_member=i_member,
                train_dir=train_dir
            )

            best_epoch = self.n_epoch  # replace it with the fancier function

            # Prediction on candidates
            self.prediction_by_member(
                elements=self.elements,
                train_dir=train_dir,
                best_epoch=best_epoch,
                predict_dir=predict_dir,
                prediction_type='candidates'
            )

            # Prediction on training_set
            self.prediction_by_member(
                elements=self.elements,
                train_dir=train_dir,
                best_epoch=best_epoch,
                predict_dir=valid_dir,
                prediction_type='validation'
            )
        os.chdir(self.working_dir)

    def query_by_committee(self, score='force'):
        '''
        candidates: candidates input.data file which will be used to generate candidates_ids_dict.
        training_set: input.data file which will be used to generate training_ids_dict
        iteration_folder: The folder containing train-xx, valid-xx and predict-xx folders.
        stats_fname: The file in which statistics of QbC are written into.
        '''
        prediction_dirs = Path(f'{self.working_dir}/iteration-{self.iteration:03d}').glob('predict-*')
        validation_dirs = Path(f'{self.working_dir}/iteration-{self.iteration:03d}').glob('valid-*')
        candidates_ids_dict = get_ids_dict(self.candidates_data)
        training_ids_dict = get_ids_dict(self.training_set)

        if score == 'energy':
            energy_predictions_dict = {}
            for i_model, dir in enumerate(sorted(prediction_dirs)):
                print(i_model, dir)
                energy_predictions_dict[i_model] = read_energy_prediction(f'{dir}/energy.comp')
                # This part of the code is incomplete...

        if score == 'force':
            # Candidates
            prediction_force_predictions_dict = {}
            for i_model, dir in enumerate(sorted(prediction_dirs)):
                print(i_model, dir)
                prediction_force_predictions_dict[i_model] = read_force_predictions(f'{dir}/forces.comp')

            predictions_mean_stds_list = []
            for id_ in candidates_ids_dict.keys():
                data = []
                for i_model in prediction_force_predictions_dict.keys():
                    data.append(prediction_force_predictions_dict[i_model][id_]['nnp'])
                data = np.array(data)
                predictions_mean_stds_list.append([id_, mean_std(data)])

            # Training Set
            valid_force_predictions_dict = {}
            for i_model, dir in enumerate(sorted(validation_dirs)):
                print(i_model, dir)
                valid_force_predictions_dict[i_model] = read_force_predictions(f'{dir}/forces.comp')

            validation_mean_stds_list = []
            for id_ in training_ids_dict.keys():
                data = []
                for i_model in valid_force_predictions_dict.keys():
                    data.append(valid_force_predictions_dict[i_model][id_]['nnp'])
                data = np.array(data)
                validation_mean_stds_list.append([id_, mean_std(data)])

        predictions_mean_stds = np.array(predictions_mean_stds_list)
        validation_mean_stds = np.array(validation_mean_stds_list)

        sorted_by_mean_std = predictions_mean_stds[predictions_mean_stds[:, 1].argsort()][::-1]

        candidates_mean_std = np.mean(predictions_mean_stds[:, 1])
        candidates_max_std = np.max(predictions_mean_stds[:, 1])
        trainingset_mean_std = np.mean(validation_mean_stds[:, 1])
        trainingset_max_std = np.max(validation_mean_stds[:, 1])
        selected_mean_std = np.mean(sorted_by_mean_std[:20, 1])

        finished = False
        if candidates_max_std < trainingset_mean_std:
            finished = True

        selected_ids = []
        selected_output_fname = f'{iteration_folder}/selected_ids.txt'
        with open(selected_output_fname, 'w') as f:
            f.write('# 1.ID  2.STD\n')
            for id_, std in sorted_by_mean_std[:20]:
                selected_ids.append(candidates_ids_dict[id_])
                f.write(f'{candidates_ids_dict[id_]}  {std}\n')

        stats_fname = f'{iteration_folder}/stats.txt'
        with open(stats_fname, 'w') as f:
            f.write(f'MEAN STD over candidates pool: {candidates_mean_std}\n')
            f.write(f'MAX STD over candidates pool: {candidates_max_std}\n')
            f.write(f'MEAN STD over training set: {trainingset_mean_std}\n')
            f.write(f'MAX STD over training set: {trainingset_max_std}\n')
            f.write(f'MEAN STD over chosen structures: {selected_mean_std}\n')

        return selected_ids, finished

    def read_selected_id(self, iteration):
        selected_ids_file = f'iteration-{iteration:03d}/selected_ids.txt'
        if os.path.exists(selected_ids_file):
            selected_ids = []
            with open(selected_ids_file, 'r') as f:
                for line in f.readlines():
                    if line[0] == '#':
                        continue
                    parts = line.split()
                    selected_ids.append(parts[0])
            return selected_ids
        else:
            return None

    def run_iterative_qbc(self):
        return


def iterative_qbc(candidates_data, current_training_set, restart_iter):
    '''
    candidates_data: input.data of candidates that are to be sampled for training the model.
    current_training-set: input.data of current_training_set. It can be empty, which is equivalent to
        starting QbC from scratch.
    restart_iter: Restart from this iteration and use committee to QbC data points for the next generation.
    '''

    if current_training_set:
        pass
    if restart_iter:
        selected_ids_file = f'iteration-{restart_iter:03d}/selected_ids.txt'
        if os.path.exists(selected_ids_file):
            selected_ids = []
            with open(selected_ids_file, 'r') as f:
                for line in f.readlines():
                    if line[0] == '#':
                        continue
                    parts = line.split()
                    selected_ids.append(parts[0])

        candidate_snapshots = input_data_reader(candidates_file)

        for id_ in selected_ids:
            s = candidate_snapshots[id_]
            s.generate_vasp_inputs(output_folder=f'{calc_folder}/{s.name}', ncpu=40)
            s.submitjob(whichHPC='young')
