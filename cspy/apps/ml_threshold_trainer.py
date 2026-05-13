import json
import logging
import os
import time
from collections import defaultdict, deque
import subprocess

import numpy as np

from cspy.configuration import CONFIG, CspyConfiguration
from cspy.util.logging_config import DATEFMT, FORMATS
from cspy.util.path import Path
from cspy import Crystal
import sqlite3
from typing import List, Tuple, Literal, Union

LOG = logging.getLogger(__name__)

def generate_n2p2_input_data_str(
    comment: str,
    lattice_vectors: np.ndarray,
    atoms: List[Tuple[str, float, float, float, float, float, float]],
    energy: float,
    charge: float,
) -> str:
    """Generate a string formatted for n2p2 input.data file.
    
    Parameters:
        comment (str): Comment for the structure, typically the structure ID.
        lattice_vectors (np.ndarray): 3x3 array of lattice vectors.
        atoms (List[Tuple[str, float, float, float, float, float, float]]): List of tuples for each atom, where each tuple contains:
            - Element symbol (str)
            - x coordinate (float)
            - y coordinate (float)
            - z coordinate (float)
            - fx (float): Force x component
            - fy (float): Force y component
            - fz (float): Force z component
        energy (float): Total energy of the structure.
        charge (float): Charge of the structure.

    Returns:
        str: Formatted string for n2p2 input.data file.
    """
    content = ['begin']
    content.append(f'comment   {comment}')
    for vector in lattice_vectors:
        content.append(
            "lattice   {:.16E} {:.16E} {:.16E}".format(*vector)
        )
    for atom_data in atoms:
        content.append(
            # x, y, z, elem, c, n, fx, fy, fz (in direct/cartesian)
            "atom   {:.16E} {:.16E} {:.16E} {} {} {} {:.16E} {:.16E} {:.16E}".format(
                float(atom_data[1]),  # x
                float(atom_data[2]),  # y
                float(atom_data[3]),  # z
                str(atom_data[0]),    # element
                0.0,                  # atom charge (not implemented in n2p2 as of July 2025)
                0.0,                  # a placeholder, not implemented in n2p2 as of July 2025
                float(atom_data[4]),  # fx
                float(atom_data[5]),  # fy
                float(atom_data[6]),  # fz
            )
        )
    content.append(f"energy   {energy:.16E}")
    content.append(f"charge   {charge:.16E}")
    content.append('end\n')
    return "\n".join(content)


class ML_Threshold_Trainer:
    """
    Class for training neural network potentials from CSPy DBs
    """
    def __init__(self, 
                 dbs_dir: str, 
                 network_type: Literal["single", "committee"], 
                 train_interval: float, 
                 ncores: int, 
                 nbins: int, 
                 **kwargs):
        """
        Args:
            dbs_dir (str): Directory containing reference data DBs.
            network_type (str): Type of neural network to train, either 'single' or 'committee'.
            train_interval (float): Number of new data points to pick at each active learning iteration.
            ncores (int): Number of cores to use for parallel processes.
            nbins (int): Number of bins for symmetry function histogram.
            **kwargs: Additional keyword arguments for configuration.
        """

        self.dbs_dir = dbs_dir
        self.train_interval = train_interval
        self.prev_training_size = 0
        self.dbs = []
        self.sleep_duration = kwargs.get("sleep_duration", 15.0)
        self.finished = False
        self.ncores = ncores
        self.nbins = nbins
        self.eval_metric: str = kwargs.get("eval_metric", "MAE")
        self.eval_quantity: str = kwargs.get("eval_quantity", "energy")
        self.eval_per_atom: bool = not kwargs.get("eval_total_energy")
        self.eval_dataset: str = kwargs.get("eval_dataset")
        self.random_seed: Union[int, None] = kwargs.get("random_seed")
        self.normalize_dataset: bool = kwargs.get('normalize_dataset', False)

        training_methods = {
            'single': self.train_single_network,
            'committee': self.train_committee_NNP,
        }
        self.train_network = training_methods[network_type]

        config = CspyConfiguration()

        n2p2_template_path: str = config.get('ml_threshold.inputnn_template')
        with open(n2p2_template_path, 'r') as f:
            self.inputnn_template_str = f.read()

        if network_type == "committee":
            n2p2_committee_dir: str = config.get('ml_threshold.committee_dir')
            self.committee_network_dirs = sorted([
                os.path.join(n2p2_committee_dir, f)
                for f in os.listdir(n2p2_committee_dir)
                if os.path.isdir(os.path.join(n2p2_committee_dir, f))
            ])

            # TODO: use numpy seed generator as n2p2 uses MT which may not work well
            # similar seeds
            self.seeds = [
                x for x in range(1234567, 1234567 + len(self.committee_network_dirs))
            ]

        self.reference_dataset = ReferenceDataset(self.dbs_dir)

    def run_normalization(self) -> None:
        """Run nnp-norm to normalize the dataset."""
        LOG.info("Normalizing dataset")
        log_file = "nnp-norm.log.0000"
        try:
            subprocess.run(
                ["nnp-norm"],
                capture_output=True,
                check=True,
                universal_newlines=True,
            )
        except subprocess.CalledProcessError as e:
            LOG.error("Normalization failed")
            LOG.error(f"nnp-norm stderr: {e.stderr}")
            if os.path.exists(log_file):
                LOG.error(
                    "Log file of nnp-norm:\n{}".format(
                        "\n".join(Path(log_file).read_text().splitlines()[-10:])
                    )
                )
            raise e
        
    def run_scaling(self, ncores: int) -> None:
        """Run nnp-scaling to calculate symmetry functions of dataset."""
        LOG.info("Calculating symmetry functions of dataset")
        log_file = "nnp-scaling.log.0000"
        try:
            subprocess.run(
                [
                    "mpirun",
                    "-np",
                    str(ncores),
                    "nnp-scaling",
                    str(self.nbins),
                ],
                capture_output=True,
                check=True,
                universal_newlines=True,
            )
        except subprocess.CalledProcessError as e:
            LOG.error("nnp-scaling failed")
            LOG.error(f"nnp-scaling stderr: {e.stderr}")
            if os.path.exists(log_file):
                LOG.error(
                    "Log file of nnp-scaling:\n{}".format(
                        "\n".join(Path(log_file).read_text().splitlines()[-10:])
                    )
                )
            raise e

    def run_training(self, ncores: int) -> None:
        """Run nnp-train using mpirun."""
        log_file = "nnp-train.log.0000"
        try:
            subprocess.run(
                ["mpirun", "-np", str(ncores), "nnp-train"],
                capture_output=True,
                universal_newlines=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            LOG.error("Training failed")
            LOG.error(f"nnp-train stderr: {e.stderr}")
            if os.path.exists(log_file):
                LOG.error(
                    "Log file of nnp-train:\n{}".format(
                        "\n".join(Path(log_file).read_text().splitlines()[-100:])
                    )
                )
            raise e


    def write_training_data(self) -> None:
        """Write training structures to n2p2 input.data file."""
        self.reference_dataset.to_n2p2_data_file()

    def write_input_file(self) -> None:
        """Write input.nn file to current directory."""
        with open("input.nn", "w") as f:
            f.write(self.input_nn_file_contents)

    def write_inputnn_from_template(self, *args) -> None:
        """Write input.nn file to current directory from template."""
        file_content = self.inputnn_template_str.format(*args)
        with open("input.nn", "w") as f:
            f.write(file_content)

    def select_weights(self) -> List[Tuple[str, str]]:
        """
        Choose which epoch's weights to use from current directory.

        Returns:
            List[Tuple[str, str]]: List of (filename, file contents) for the weights chosen for each element.
        """
        from glob import glob
        from pathlib import Path
        from cspy.ml.nnp.format.parse_n2p2_learning_curve import select_best_epoch_from_file

        try:
            best_epoch, epoch_values = select_best_epoch_from_file(
                quantity=self.eval_quantity,
                metric=self.eval_metric,
                per_atom=self.eval_per_atom,
                dataset=self.eval_dataset,
            )
        except Exception:
            LOG.warning("Unable to parse training output")
            return []

        LOG.info(
            f"Best epoch {best_epoch} ({epoch_values[best_epoch]} {self.eval_quantity} "
            f"{self.eval_metric}"
            f"{' pa' if self.eval_per_atom and self.eval_quantity == 'energy' else ''} "
            f"on {self.eval_dataset})"
        )

        final_weights_files = glob(f"weights.*.{best_epoch:06}.out")

        return [
            (
                f'weights.{Path(w).stem.split(".")[1]}.data',
                Path(w).read_text()
            )
            for w in final_weights_files
        ]

    def train_single_network(self) -> None:
        """
        Scale dataset and train neural network weights using n2p2.
        Copy resulting weights and scaling.data files to specified locations.
        """
        from tempfile import TemporaryDirectory
        import shutil
        import os
        from glob import glob

        working_dir = os.getcwd()
        ncores = int(min(len(self.reference_dataset) / 4, self.ncores))

        assert self.random_seed is not None, "Must set random seed for single network"

        config = CspyConfiguration()
        n2p2_inputnn_filepath = config.get('ml_threshold.inputnn_path')
        n2p2_scaling_filepath = config.get('ml_threshold.scaling_path')
        n2p2_weights_dir = config.get('ml_threshold.weights_dir')
        train_from_prev = config.get("ml_threshold.train_from_prev", False)

        with TemporaryDirectory(prefix=f'{os.getcwd()}/') as tmpdirname:
            os.chdir(tmpdirname)

            # Write input.nn file using template and random seed
            self.write_inputnn_from_template(self.random_seed)
            # Write training data to input.data
            self.write_training_data()

            t1 = time.time()
            # Normalize data if requested
            if self.normalize_dataset:
                self.run_normalization()

            # Calculate symmetry functions and scaling
            self.run_scaling(ncores)
            LOG.info(f'Finished scaling in {time.time() - t1:.2f} seconds')

            # Optionally copy previous weights for training
            if train_from_prev:
                weights = glob(os.path.join(n2p2_weights_dir, "weights.*.data"))
                if weights:
                    LOG.info("Training from previous weights")
                for w in weights:
                    shutil.copy(w, tmpdirname)

            # Run training
            t1 = time.time()
            self.run_training(ncores)
            LOG.info(f'Finished training in {time.time() - t1:.2f} seconds')

            # Copy output files to working directory
            shutil.copy('input.data', working_dir)
            shutil.copy('input.nn', working_dir)
            shutil.copy('scaling.data', working_dir)
            shutil.copy('learning-curve.out', working_dir)

            final_weights = self.select_weights()

            # Copy input.nn and scaling.data to specified locations
            shutil.copy('input.nn', n2p2_inputnn_filepath)
            shutil.copy('scaling.data', n2p2_scaling_filepath)
            for w in final_weights:
                with open(os.path.join(n2p2_weights_dir, w[0]), 'w') as f:
                    f.write(w[1])

            os.chdir(working_dir)

    def train_committee_NNP(self) -> None:
        """
        Scale dataset and train neural network weights using n2p2.
        Copy resulting weights and scaling.data files to specified locations.
        """
        from tempfile import TemporaryDirectory
        import shutil
        import os
        import re
        from glob import glob

        working_dir = os.getcwd()
        ncores = int(min(len(self.reference_dataset) / 4, self.ncores))

        config = CspyConfiguration()
        n2p2_inputnn_filepath = config.get('ml_threshold.inputnn_path')
        n2p2_scaling_filepath = config.get('ml_threshold.scaling_path')
        train_from_prev = config.get("ml_threshold.train_from_prev", False)

        selected_weights = {}

        with TemporaryDirectory(prefix=f'{os.getcwd()}/') as tmpdirname:
            os.chdir(tmpdirname)

            # Write input.nn file
            self.write_inputnn_from_template(self.seeds[0])
            # Create input.data
            self.write_training_data()

            # Run scaling
            t1 = time.time()
            # Normalize data and copy generated input.nn to n2p2_inputnn_filepath
            if self.normalize_dataset:
                self.run_normalization()

            # Calculate symmetry functions and scaling
            self.run_scaling(ncores)
            LOG.info(f'Finished scaling in {time.time() - t1:.2f} seconds')

            t1 = time.time()
            inputfile_contents = Path("input.nn").read_text()

            # Train each network
            for i, network_dir in enumerate(self.committee_network_dirs):
                if train_from_prev:
                    weights = glob(os.path.join(network_dir, "weights.*.data"))
                    if weights:
                        LOG.info("Training from previous weights")
                        for w in weights:
                            shutil.copy(w, tmpdirname)
                    else:
                        LOG.info("No previous weights found, training from scratch")

                LOG.info(f"Training network in {network_dir}")
                if i > 0:
                    tmp = re.sub(
                        r"random_seed\s*[0-9]*",
                        f"random_seed {self.seeds[i]}",
                        inputfile_contents,
                    )
                    with open("input.nn", 'w') as f:
                        f.write(tmp)

                self.run_training(ncores)
                shutil.copy(
                    "learning-curve.out",
                    os.path.join(working_dir, f"learning-curve-{i}.out"),
                )

                # Select final epoch weight for each atomic number and copy to location
                selected_weights[network_dir] = self.select_weights()

            LOG.info(f'Finished training in {time.time() - t1:.2f} seconds')
            shutil.copy('input.data', working_dir)
            shutil.copy('input.nn', working_dir)
            shutil.copy('scaling.data', working_dir)

            shutil.copy('input.nn', n2p2_inputnn_filepath)
            shutil.copy('scaling.data', n2p2_scaling_filepath)
            for network_dir, weights in selected_weights.items():
                for w in weights:
                    with open(os.path.join(network_dir, w[0]), 'w') as f:
                        f.write(w[1])

            os.chdir(working_dir)
    
    def run(self) -> None:
        """
        Run training program loop. Once the specified number of new data points
        is added, trains new weights and writes them. Continues until the
        'finished' variable in the cspy.toml file is set to True.
        """
        while True:
            # Read reference dbs and update training dataset
            self.reference_dataset.update_dataset()
            num_data = len(self.reference_dataset)
            LOG.info(f'Number of data points = {num_data}')

            # Train network if enough new data points
            if num_data > self.prev_training_size + self.train_interval:
                LOG.info('Training network')
                self.train_network()
                self.write_network_weights()
                self.prev_training_size = num_data
            else:
                time.sleep(self.sleep_duration)

            # Check if training should finish
            if CspyConfiguration().get('ml_threshold.finished', False):
                break

        self.shutdown()
            
    def shutdown(self) -> None:
        LOG.info("Waiting for threads to finish...")
        if hasattr(self, "dbs_updater_thread") and self.dbs_updater_thread is not None:
            self.dbs_updater.finish = True
            self.dbs_updater_thread.join()
        if hasattr(self, "dbs_reader_thread") and self.dbs_reader_thread is not None:
            self.dbs_reader.finish = True
            self.dbs_reader_thread.join()
        LOG.info("ML Threshold Trainer complete! :)")


class ReferenceDataset():
    def __init__(self, dbs_dir: str):
        """
        Args:
            dbs_dir (str): Directory containing reference data DBs.
        """
        self.dbs_dir = dbs_dir
        self.ref_data = {}
        self.prev_rows = defaultdict(int)

    def __iter__(self):
        return iter(self.ref_data)
    
    def __len__(self):
        return len(self.ref_data)

    @property
    def ref_dbs(self):
        from glob import glob
        return glob(os.path.join(self.dbs_dir, "*.db"))
    
    @property
    def reference_data(self):
        self.update_dataset()
        return self.ref_data

    def to_n2p2_data_file(self) -> None:
        """Convert training structures to n2p2 input.data file."""
        content = ""
        for id, data in self.ref_data.items():
            content += generate_n2p2_input_data_str(
                id,
                data['lattice_vectors'],
                data['atom_records'],
                data['energy'],
                data['charge'],
            )
        
        with open('input.data', 'w') as f:
            f.write(content)
    
    def update_dataset(self) -> None:
        dbs = deque(self.ref_dbs)

        while dbs:
            db = dbs.popleft()
            dbname = os.path.basename(db)
            LOG.debug(f'Checking {dbname} for new structures')

            # Use immutable mode to avoid lock issues
            conn = sqlite3.connect(f'file:{db}?immutable=1', uri=True)
            cur = conn.cursor()

            # Check new rows added since previously checked
            try:
                num_rows = cur.execute('SELECT COUNT(id) FROM crystal').fetchone()[0]
            except Exception as e:
                LOG.info(f"Failed to open {dbname}: {e}")
                continue

            LOG.debug(f'{dbname}: rows = {num_rows}; previously = {self.prev_rows[db]}')

            # Add new rows to data list
            if num_rows > self.prev_rows[db]:
                for structure_data in cur.execute(
                    "SELECT id, trial_number, energy, density, file_content, metadata "
                    "FROM crystal JOIN trial_structure USING(id)"
                ):
                    if structure_data[0] in self.ref_data:
                        continue

                    id, trial_number, energy, density, file_content, metadata = structure_data
                    metadata = json.loads(metadata)
                    try:
                        crys = Crystal.from_shelx_string(file_content)
                    except Exception as e:
                        LOG.error(f'Unable to read crystal {id} from {dbname} due to {e}')
                        continue

                    labels = crys.asymmetric_unit.labels
                    elements = crys.asymmetric_unit.elements
                    positions = crys.unit_cell_atoms()['cart_pos']
                    lattice_vectors = crys.unit_cell.lattice
                    charge = 0.0
                    forces = metadata.get('forces', [[0.0, 0.0, 0.0] for _ in labels])

                    atom_records = [
                        (element, *pos, *force)
                        for element, pos, force in zip(elements, positions, forces)
                    ]

                    self.ref_data[id] = {
                        'trial_number': trial_number,
                        'energy': energy,   # assuming units same as ref_method
                        'charge': charge,
                        'density': density,
                        'res_content': file_content,
                        'labels': labels,
                        'elements': elements,
                        'positions': positions,
                        'forces': forces,
                        'lattice_vectors': lattice_vectors,
                        'atom_records': atom_records,
                    }

            self.prev_rows[db] = num_rows

            conn.close()
            cur = None


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Train BP NN potential using CSPy DBs"
    )
    
    parser.add_argument(
        "dbs_dir",
        type=str,
        help='Path to directory containing reference data (CSPy) DBs.',
    )

    parser.add_argument(
        "network_type",
        choices=["single", "committee"],
        default="single",
        type=str,
        help="Type of neural network to train",
    )
    
    parser.add_argument(
        "-j",
        "--ncores",
        default=4,
        type=int,
        help="Number of cores for parallel processes",
    )

    parser.add_argument(
        "-b",
        "--nbins",
        default=500,
        type=int,
        help="Number of bins for symmetry function histogram",
    )

    parser.add_argument(
        "--eval-quantity",
        choices=["energy", "forces", "combined"],
        default="energy",
        help="Criterion for evaluating best epoch during training single network.",
    )

    parser.add_argument(
        "--eval-metric",
        choices=["MAE", "RMSE"],
        default="MAE",
        help="Metric for evaluating best epoch during training single network.",
    )

    parser.add_argument(
        "--eval-total-energy",
        action="store_true",
        default=False,
        help="Evaluate best weights based on total energy error rather than per atom",
    )

    parser.add_argument(
        "--eval-dataset",
        type=str,
        choices=["train", "test"],
        default="test",
        help="Dataset to evaluate best epoch from",
    )

    parser.add_argument(
        "-n",
        "--normalize_dataset",
        action="store_true",
        default=False,
        help="Run nnp-norm on dataset before training",
    )

    parser.add_argument(
        "-rs",
        "--random-seed",
        type=int,
        default=None,
        help="Random seed",
    )
    
    parser.add_argument(
        "--log-level",
        default=CONFIG.get("csp.log_level"),
        help="Log level",
    )

    args = parser.parse_args()
    logging.basicConfig(
        format=FORMATS[args.log_level],
        datefmt=DATEFMT,
        level=args.log_level,
    )
    
    config = CspyConfiguration()

    dbs_dir = os.path.abspath(args.dbs_dir)
    train_interval = config.get('ml_threshold.train_interval', 20)

    trainer = ML_Threshold_Trainer(
        dbs_dir=dbs_dir,
        network_type=args.network_type,
        train_interval=train_interval,
        ncores=args.ncores,
        nbins=args.nbins,
        eval_metric=args.eval_metric,
        eval_quantity=args.eval_quantity,
        eval_total_energy=args.eval_total_energy,
        eval_dataset=args.eval_dataset,
        normalize_dataset=args.normalize_dataset,
        random_seed=args.random_seed,
    )
    LOG.info(f"Training BP NN potential using dbs in {dbs_dir}")

    trainer.run()

if __name__ == "__main__":
    main()
