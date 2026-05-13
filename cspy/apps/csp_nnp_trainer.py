import logging
import os
import random
from collections import defaultdict
import numpy as np
from typing import Iterator, Union, Literal, Dict, List
from cspy import Crystal
from cspy.configuration import CONFIG, CspyConfiguration
from cspy.cspympi.ml_threshold_tasks import MLThresholdWorker, ML_ThresholdStructure
from cspy.cspympi.threshold_tasks import PerturbedStructure
from cspy.db import CspDataStore
from cspy.similarity.unique_structures import UniqueStructures
from cspy.util.logging_config import DATEFMT, FORMATS
from cspy.apps.ml_threshold_trainer import (
    ML_Threshold_Trainer,
    generate_n2p2_input_data_str,
)
LOG = logging.getLogger(__name__)
BIG_UNCERTAINTY = 1e6

def plot_uncertainties(uncertainties_file: str) -> None:
    """ Plot uncertainties from the relevant file with the following columns:
    - n_candidates
    - mean_uncertainty
    - std_uncertainty
    - max_uncertainty
    - ratio

    Args:
        uncertainties_file (str): Path to the file containing uncertainties data.

    Returns:
        None: The function saves the plots as images.
    """
    import matplotlib.pyplot as plt

    with open(uncertainties_file, 'r') as f:
        lines = f.readlines()

    data = [
        [float(x) for x in line.split()] 
        for line in lines 
        if not line.startswith('#')
    ]
    data = np.array(data)

    iterations = np.arange(1, data.shape[0] + 1)
    mean_uncertainty, std_uncertainty, max_uncertainty, ratio = data[:, 1:5].T

    # Plot mean_uncertainty with error bars
    plt.figure(figsize=(12, 8))
    plt.errorbar(
        iterations,
        mean_uncertainty,
        yerr=std_uncertainty,
        fmt='o',
        ecolor='black',
        capsize=5,
        color='grey',
        markersize=8,
        markerfacecolor='blue',
        markeredgewidth=1.5,
        linestyle='-',
        linewidth=1,
        elinewidth=1.5,
        capthick=1.5,
        label='Mean uncertainty'
    )

    plt.xlabel('AL iteration', fontsize=14, fontweight='bold')
    plt.ylabel('Mean uncertainty (kJ/mol)', fontsize=14, fontweight='bold')
    plt.title('Mean Uncertainty', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig('mean_uncertainty.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot max_uncertainty and ratio of candidates above uncertainty threshold
    fig, ax1 = plt.subplots(figsize=(12, 8))
    color1 = 'tab:blue'
    ax1.set_xlabel('AL Iteration', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Max Uncertainty (kJ/mol)', color=color1, fontsize=14, fontweight='bold')
    ax1.plot(
        iterations,
        max_uncertainty,
        'o-',
        color=color1,
        markersize=8,
        markerfacecolor=color1,
        markeredgewidth=1.5,
        linewidth=2,
        label='Max uncertainty'
    )
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.tick_params(axis='both', which='major', labelsize=12)

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel(
        'Ratio of candidates above uncertainty threshold',
        color=color2,
        fontsize=14,
        fontweight='bold'
    )
    ax2.plot(
        iterations,
        ratio,
        's-',
        color=color2,
        markersize=8,
        markerfacecolor=color2,
        markeredgewidth=1.5,
        linewidth=2,
        label='Ratio of candidates above threshold'
    )
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.tick_params(axis='both', which='major', labelsize=12)

    plt.title(
        'Max Uncertainty / Ratio of Candidates Above Uncertainty Threshold',
        fontsize=16,
        fontweight='bold'
    )
    fig.tight_layout()
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper left', fontsize=12)
    ax2.legend(loc='upper right', fontsize=12)
    plt.savefig('max_uncertainty_and_ratio.png', dpi=300, bbox_inches='tight')
    plt.close()


class CSP_NNP_Trainer(ML_Threshold_Trainer):
    """
    Class for training neural network potentials on the fly from csp dbs
    """
    def __init__(
        self,
        dbs_dir: str,
        network_type: Literal["single", "committee"],
        training_method: Literal["sequential", "highest_uncertainty", "highest_uncertainty_FPS"],
        train_interval: int,
        target: int,
        ncores: int,
        **kwargs
    ) -> None:
        """
        Args:
            dbs_dir (str): path to directory with csp databases
            network_type (str): type of neural network to train, either 'single' or 'committee'
            training_method (str): method for selecting structures to add to training set
            train_interval (int): number of structures to add to training set before re-training
            target (int): target size of training set after which training stops
            ncores (int): number of cores to use for parallel processing
            **kwargs: additional keyword arguments for training settings
        """
        self.dbs_dir = dbs_dir
        self.train_interval = train_interval
        self.sleep_duration = 5.0
        self.reference_dataset = {}
        self.prev_training_size = 0
        self.prev_nrows = defaultdict(int)
        self.ncores = ncores
        self.trained_weights = False
        self.finished = False
        self.random_seed: int = kwargs.get("random_seed", 1)
        random.seed(self.random_seed)

        # NNP settings
        self.target_dataset_size = target
        self.normalize_dataset: bool = kwargs.get("normalize_dataset", False)
        self.nbins: int = kwargs.get("nbins", 500)
        self.eval_metric: str = kwargs.get("eval_metric", "MAE")
        self.eval_quantity: str = kwargs.get("eval_quantity", "energy")
        self.eval_per_atom: bool = not kwargs.get("eval_total_energy")
        self.eval_dataset: str = kwargs.get("eval_dataset", "test")
        self.uncertainty_threshold: float = kwargs.get("uncertainty_threshold")
        self.randomize_dataset: bool = kwargs.get("randomize_dataset", False)
        self.energy_cutoff: Union[None, float] = kwargs.get("energy_cutoff", None)
        self.delta_learning: bool = kwargs.get("delta_learning", False)
        self.energy_window: Union[None, float] = kwargs.get("energy_window", None)

        self.nnp_settings = {}
        self.nnp_settings.update(kwargs)

        training_methods = {
            "sequential": self.add_ref_data_sequential,
            "highest_uncertainty": self.add_ref_data_highest_uncertainty,
            "highest_uncertainty_FPS": self.add_ref_data_highest_uncertainty_FPS,
        }
        self.training_method = training_method
        self.add_ref_data = training_methods[training_method]

        self.unique_structures = UniqueStructures()
        self.candidates = {}
        self.unique_index = 0

        self.singlepoint_method = "cNNP" if network_type == "committee" else "mlp"
        network_methods = {
            "single": self.train_single_network,
            "committee": self.train_committee_NNP,
        }
        self.train_network = network_methods[network_type]

        config = CspyConfiguration()
        self.reference_method: str = config.get("ml_threshold.reference_method")
        self.remove_conformational_energy: bool = config.get(
            "ml_threshold.remove_conformational_energies", False
        )
        self.conformational_energies: Dict = config.get("conformational_energies", {})
        self.override_asym_nmols: Union[None, int] = config.get("ml_threshold.override_asym_nmols")

        n2p2_template_path: str = config.get("ml_threshold.inputnn_template")
        with open(n2p2_template_path, "r") as f:
            self.inputnn_template_str = f.read()

        if network_type == "committee":
            n2p2_committee_dir: str = config.get("ml_threshold.committee_dir")
            self.uncertainty_metric: str = config.get(
                "ml_threshold.uncertainty_metric", "kjmol_per_molecule"
            )
            self.committee_network_dirs = sorted(
                [
                    os.path.join(n2p2_committee_dir, f)
                    for f in os.listdir(n2p2_committee_dir)
                    if os.path.isdir(os.path.join(n2p2_committee_dir, f))
                ]
            )
            self.seeds = [
                x
                for x in range(
                    1234567, 1234567 + len(self.committee_network_dirs)
                )
            ]

        worker_data = {}
        self.worker = MLThresholdWorker(worker_data)

        if os.path.exists("input.data"):
            self.restart_from_input_data()

    @property
    def csp_dbs(self) -> List[str]:
        from glob import glob

        return glob(os.path.join(self.dbs_dir, "*.db"))

    def write_training_data(self) -> None:
        """Convert training structures to n2p2 input.data file."""
        content = ""
        for structure_id, data in self.reference_dataset.items():
            content += generate_n2p2_input_data_str(
                structure_id,
                data['lattice_vectors'],
                data['atom_records'],
                data['energy'],
                data['charge'],
            )

        with open('input.data', 'w') as f:
            f.write(content)

    def restart_from_input_data(self) -> None:
        """
        Restart the training using previous input data
        """
        from cspy.ml.nnp.format.parse_n2p2_data import parse_n2p2_data_content
        from pathlib import Path

        structure_data = parse_n2p2_data_content(Path('input.data').read_text())

        for structure in structure_data:
            comment, lattice_vectors, atom_records, energy, charge = structure
            structure_id = comment.split()[-1]

            atoms = [
                (x['element'], *x['position'], *x['forces'])
                for x in atom_records
            ]

            self.reference_dataset[structure_id] = {
                'energy': energy,
                'charge': charge,
                'lattice_vectors': lattice_vectors,
                'atom_records': atoms,
            }

        self.train_network()
        self.trained_weights = True  # assumes weights in correct location

    def read_in_unique_structures_from_db(self, low_mem: bool = False) -> None:
        """
        Read in unique structures from db. Ignore structures without descriptors.

        Args:
            low_mem (bool): if True, do not read file_content from db, only energy and molecule_id.
                            Useful for large datasets to save memory.
        """
        query_str = (
            "select energy, molecule_id"
            + (", file_content " if not low_mem else " ")
            + "from crystal where id='{}' "
        )

        for dbpath in self.csp_dbs:
            LOG.info(f"Reading unique structures from {dbpath}")
            ds = CspDataStore(dbpath)
            if "molecule_id" not in ds.get_table_columns("crystal"):
                LOG.error(
                    f"molecule_id column not found in crystal table of {dbpath}. "
                    "Please ensure the database has the latest schema."
                )
                LOG.info(
                    "Run cspy-db convert to update the database schema if needed."
                )
                LOG.info("Skipping the database %s", dbpath)
                continue

            max_energy = ds.query("select max(energy) from crystal").fetchone()[0]
            if self.energy_window is not None:
                min_energy = ds.query("select min(energy) from crystal").fetchone()[0]
                max_energy = min_energy + float(self.energy_window)

            candidate_ids_of_dbpath = [
                x[0]
                for x in ds.query(
                    "select distinct(id) from crystal "
                    "join descriptor using(id) "
                    "join equivalent_to on crystal.id = equivalent_to.unique_id "
                    f"where energy <= {max_energy}"
                ).fetchall()
            ]

            if len(candidate_ids_of_dbpath) == 0:
                n_crystal = len(ds.query("select id from crystal").fetchall())
                n_et = len(ds.query("select * from equivalent_to").fetchall())
                LOG.warning(f"0 structures were extracted in {dbpath}.")
                if n_crystal > 0 and n_et == 0:
                    LOG.warning(f"There are {n_crystal} IDs in crystal table of {dbpath}.")
                    LOG.warning("The database has not been clustered. Run cspy-db cluster.")

            # If using clustered db, randomize to avoid grouping structures by spacegroup
            if self.randomize_dataset:
                random.shuffle(candidate_ids_of_dbpath)

            for id in candidate_ids_of_dbpath:
                try:
                    energy, molecule_id, *data = ds.query(query_str.format(id)).fetchone()
                    if molecule_id.startswith("Empty") or molecule_id.startswith("no mol"):
                        molecule_id = None
                except Exception:
                    LOG.info(f"Failed to read ID {id} from {dbpath} database. Skipping this ID.")
                    continue
                if (
                    id not in self.reference_dataset
                    and (self.energy_cutoff is None or energy < self.energy_cutoff)
                ):
                    self.candidates[id] = CandidateStructure(
                        name=id,
                        dbpath=dbpath,
                        csp_energy=energy,
                        data=data,
                        molecule_id=molecule_id,
                    )
            ds.close()
            LOG.debug("read_in_unique_structures_from_db() finished successfully.")
            
    def calculate_candidate_descriptors(self) -> None:
        """
        Calculate descriptors for candidate structures.
        """
        from cspy.ml.nnp.pynnp_symmetry_functions import calculate_descriptors

        config = CspyConfiguration()
        inputnn_fpath = config.get('ml_threshold.inputnn_template')

        symm_funcs = calculate_descriptors(
            jobs=[c.file_content for c in self.candidates.values()],
            inputnn_fpath=inputnn_fpath,
            scaling_fpath=None,
            nprocs=self.ncores,
        )
        for i, (name, G) in enumerate(zip(self.candidates, symm_funcs)):
            self.candidates[name].index = i
            self.candidates[name].descriptor = G

    def calculate_distance_matrix(self, candidates: List[str], metric: str = "euclidean") -> np.ndarray:
        """
        Calculate distance matrix between structures for FPS sampling.

        Args:
            candidates (list[str]): candidate names to construct distance matrix with
            metric (str): distance metric to use. Defaults to "euclidean"

        Returns:
            np.ndarray: distance matrix between candidates
        """
        from cspy.ml.nnp.pynnp_symmetry_functions import calculate_distance
        
        # calculate distance matrix
        return calculate_distance(
            descriptors=[self.candidates[name].descriptor for name in candidates],
            nprocs=self.ncores,
            method=metric,
        )

    def evaluate_candidate_uncertainty(self, structure: PerturbedStructure) -> Union[float, None]:
        """Evaluate candidate using cNNP.

        Args:
            structure (PerturbedStructure): Structure to evaluate.

        Returns:
            float | None: A float containing the uncertainty of the prediction or None if calculation fails.
        """
        result = self.worker.energy_methods[self.singlepoint_method](structure)
        try:
            uncertainty = self.calculate_uncertainty(result)
            LOG.info(
                "cNNP prediction for %s: energy = %s uncertainty = %s (threshold = %s)",
                structure.name,
                result.metadata.get("raw_energy", result.energy),
                uncertainty,
                self.uncertainty_threshold,
            )
        except Exception:
            LOG.info("cNNP prediction failed for %s", structure.name)
            return None

        return uncertainty
            
    def add_ref_data_sequential(self) -> None:
        """
        Calculate reference data sequentially by iterating through the candidate list.
        """
        num_calculated = 0
        candidates_iter = iter(self.candidates.copy())

        while self.candidates and num_calculated < self.train_interval:
            name = next(candidates_iter)
            candidate = self.candidates.pop(name)

            crys = Crystal.from_shelx_string(candidate.file_content, titl=name)

            # Check centering to see if can find smaller P1 cell
            crys = crys.as_primitive_P1()

            structure = PerturbedStructure(
                name=name,
                trial_number=0,
                spacegroup=1,
                file_content=crys.to_shelx_string(),
                mc_step=0,
                molecule_id=candidate.molecule_id,
                Zp="calculate"
            )

            # Check if structure is poorly described by model
            if self.trained_weights and self.singlepoint_method == "cNNP":
                uncertainty = self.evaluate_candidate_uncertainty(structure)
                if uncertainty is not None and uncertainty < self.uncertainty_threshold:
                    continue

            ref_energy = self.calculate_ref_datapoint(structure, candidate.csp_energy)

            if ref_energy is None:
                continue

            num_calculated += 1

        if num_calculated == 0:
            self.finished = True

    def add_ref_data_highest_uncertainty(self) -> None:
        """
        calculate ref data by highest uncertainty from cNNP
        """
        from multiprocessing import Pool
        from functools import partial
        from cspy.ml.nnp.executable.nnp_singlepoint import n2p2_pynnp_committee_predict
        
        def to_structures(candidates: dict) -> Iterator[Crystal]:
            # TODO: create as seperate function for single candidate, then change this to gen comprehension
            for name, candidate in candidates.items():
                crys = Crystal.from_shelx_string(candidate.file_content, titl=name)
                if self.remove_conformational_energy:
                    try:
                        crys.properties['conformational_energy'] = (
                            self.conformational_energies[candidate.molecule_id]
                        )
                    except KeyError:
                        raise Exception(
                            f"Key {candidate.molecule_id} was not found in conformational_energies. "
                            "Please set it under [conformational_energies] in cspy.toml file."
                        )
                    # TODO Vasp calculation can be called here to calculate the missing values.
                else:
                    crys.properties['conformational_energy'] = 0

                # Check centering to see if can find smaller P1 cell
                crys = crys.as_primitive_P1()
                
                yield crys
        
        # evaluate candidates
        if self.trained_weights:
            num_candidates = len(self.candidates)
            LOG.info(f"evaluating %d candidate structures", num_candidates)
            with Pool(self.ncores) as pool:
                uncertainties = [
                    self.calculate_uncertainty(x) for x in pool.map(
                        partial(n2p2_pynnp_committee_predict, silent=True),
                        to_structures(self.candidates)
                    )
                ]
            
            for i, name in enumerate(self.candidates):
                self.candidates[name].uncertainty = uncertainties[i]

            uncertainties = np.array(uncertainties)
            mean_uncertainty = uncertainties.mean()
            std_uncertainty = uncertainties.std()
            max_uncertainty = uncertainties.max()
            
            LOG.info(
                "max, mean, std of uncertainties in candidates: "
                "%.4f, %.4f, %.4f",
                max_uncertainty,
                mean_uncertainty,
                std_uncertainty,
            )
            
            # check percent of candidates above uncertainty threshold, finish if < 5%
            num_uncertain = len(
                [x for x in uncertainties if x > self.uncertainty_threshold]
            )
            ratio = num_uncertain / num_candidates
            LOG.info("ratio of candidates above uncertainty threshold: %.4f (%d/%d)", 
                ratio,
                num_uncertain,
                num_candidates,
            )

            if not os.path.exists('uncertainties.dat'):
                with open('uncertainties.dat', 'w') as f:
                    f.write("# n_candidates mean_uncertainty std_uncertainty max_uncertainty ratio\n")
            with open('uncertainties.dat', 'a') as f:
                f.write(
                    f"{int(uncertainties.shape[0]):>12d}  "
                    f"{mean_uncertainty:>14.4f}  "
                    f"{std_uncertainty:>14.4f}  "
                    f"{max_uncertainty:>14.4f}  "
                    f"{ratio:>7.4f}\n"
                )
            try:
                plot_uncertainties('uncertainties.dat')
            except Exception:
                LOG.warning("Failed to plot uncertainties.")

            if ratio < 0.05:
                self.finished = True
                return
            
            # sort by uncertainty
            candidates_iter = iter(
                sorted(
                    self.candidates,
                    key=lambda x: self.candidates[x].uncertainty,
                    reverse=True,
                )
            )
        else:
            candidates_iter = iter(self.candidates.copy())
        
        num_calculated = 0
        
        # calculate ref data up to training interval
        while self.candidates and num_calculated < self.train_interval:
            name = next(candidates_iter)
            candidate = self.candidates.pop(name)
            uncertainty = getattr(candidate, "uncertainty", None)
            
            if uncertainty is not None and uncertainty < self.uncertainty_threshold:
                break
            
            crys = Crystal.from_shelx_string(candidate.file_content, titl=name)

            # check centering to see if can find smaller P1 cell
            crys = crys.as_primitive_P1()
            
            structure = PerturbedStructure(
                name=name,
                trial_number=0,
                spacegroup=1,
                file_content=crys.to_shelx_string(),
                mc_step=0,
                molecule_id = candidate.molecule_id,
                Zp="calculate"
            )
            
            ref_energy = self.calculate_ref_datapoint(
                structure, candidate.csp_energy
            )
            
            if ref_energy is None:
                continue
            
            num_calculated += 1
        
        if num_calculated == 0:
            self.finished = True
    
    def add_ref_data_highest_uncertainty_FPS(self) -> None:
        """
        add reference data by farthest point sampling structures with uncertainty above
        threshold, starting from structure with highest uncertainty
        """
        from multiprocessing import Pool
        from functools import partial
        from cspy.ml.nnp.executable.nnp_singlepoint import n2p2_pynnp_committee_predict
        # function below seems to be a copy from add_ref_data_highest_uncertainty -> consider creating one source  
        def to_structures(candidates: dict) -> Iterator[Crystal]:
            for name, candidate in candidates.items():
                crys = Crystal.from_shelx_string(
                    candidate.file_content, titl=name
                )
                if self.remove_conformational_energy:
                    try:
                        crys.properties['conformational_energy'] = (
                            self.conformational_energies[candidate.molecule_id]
                        )
                    except KeyError:
                        raise Exception(
                            f"Key {candidate.molecule_id} was not found in conformational_energies. "
                            "Please set it under [conformational_energies] in cspy.toml file."
                        )
                        # TODO Vasp calculation here
                else:
                    crys.properties['conformational_energy'] = 0
                # check centering to see if can find smaller P1 cell
                crys = crys.as_primitive_P1()
                
                yield crys
        
        # evaluate candidates
        if self.trained_weights:
            num_candidates = len(self.candidates)
            LOG.info(f"evaluating %d candidate structures", num_candidates)
            with Pool(self.ncores) as pool:
                uncertainties = [
                    self.calculate_uncertainty(x) for x in pool.map(
                        partial(n2p2_pynnp_committee_predict, silent=True),
                        to_structures(self.candidates)
                    )
                ]
            
            for i, name in enumerate(self.candidates):
                self.candidates[name].uncertainty = uncertainties[i]
            
            LOG.info(
                "max uncertainty in candidates %.4f",
                max(uncertainties)
            )
            
            # check percent of candidates above uncertainty threshold, finish if <5%
            num_uncertain = len(
                [x for x in uncertainties if x > self.uncertainty_threshold]
            )
            ratio = num_uncertain / num_candidates
            LOG.info("ratio of candidates above uncertainty threshold: %.4f (%d/%d)", 
                ratio,
                num_uncertain,
                num_candidates,
            )
            if ratio < 0.05:
                self.finished = True
                return
            
            # sort by uncertainty
            candidates_iter = iter(
                sorted(
                    self.candidates,
                    key=lambda x: self.candidates[x].uncertainty,
                    reverse=True,
                )
            )
            uncertain_candidates = [
                name
                for name in candidates_iter
                if self.candidates[name].uncertainty > self.uncertainty_threshold
            ]
        else:
            # if not trained, do FPS on entire candidate set
            uncertain_candidates = list(self.candidates)
        
        num_calculated = 0
        LOG.info("Calculating distance matrix between high uncertainty candidates")
        distance_matrix = self.calculate_distance_matrix(uncertain_candidates)
        uncertain_candidates = {i: name for i, name in enumerate(uncertain_candidates)}
        
        # calculate ref data upto training interval
        while uncertain_candidates and num_calculated < self.train_interval:
            # begin FPS from highest uncertainty structure
            if num_calculated == 0 and 0 in uncertain_candidates:
                current_idx = 0
                shortest_distances = distance_matrix[current_idx]
            else:
                current_idx = np.argmax(shortest_distances)
                shortest_distances = np.minimum(
                    shortest_distances,
                    distance_matrix[current_idx],
                )
                
            LOG.debug("current_idx=%d", current_idx)
            LOG.debug("uncertain_candidates=%s", uncertain_candidates)
            LOG.debug("shortest_distances=%s", shortest_distances)
                
            name = uncertain_candidates.pop(current_idx)
            candidate = self.candidates.pop(name)
            uncertainty = getattr(candidate, "uncertainty", None)
            
            if uncertainty is not None and uncertainty < self.uncertainty_threshold:
                break
            
            crys = Crystal.from_shelx_string(candidate.file_content, titl=name)

            # check centering to see if can find smaller P1 cell
            crys = crys.as_primitive_P1()
            
            structure = PerturbedStructure(
                name=name,
                trial_number=0,
                spacegroup=1,
                file_content=crys.to_shelx_string(),
                mc_step=0,
                molecule_id = candidate.molecule_id,
                Zp="calculate"
            )
            
            ref_energy = self.calculate_ref_datapoint(
                structure, candidate.csp_energy
            )
            
            if ref_energy is None:
                continue
            
            num_calculated += 1
        
        if num_calculated == 0:
            self.finished = True

    def calculate_ref_datapoint(self, structure: PerturbedStructure, csp_energy: float) -> Union[float, None]:
        """calculate reference datapoint

        Args:
            structure (PerturbedStructure): structure to calculate reference energy for
            csp_energy (float): CSP energy to compare against

        Returns:
            float | None: reference energy for the structure as a float, or None if calculation failed
        """
        from cspy.util.constants import EV2KJ_PER_MOL

        result = self.worker.energy_methods[self.reference_method](
            structure,
            save_forces=True,
            reference=True,
        )
        
        if result.energy is None:
            LOG.warning("calculation failed for %s", structure.name)
            return None
        
        crys = Crystal.from_shelx_string(structure.file_content)
        if self.reference_method == "vasp":
            crys = crys.standardized()
            # This won't change anything, as VaspMinimizer uses as_P1 anyway.
            crys = crys.as_P1()

        nmols = len(crys.unit_cell_molecules())
        
        metadata = result.metadata
        ref_energy = metadata.get("ref_energy", result.energy)
        labels = crys.asymmetric_unit.labels  # atom labels, e.g., H1
        elements = crys.asymmetric_unit.elements  # elements of each atom in order
        positions = crys.unit_cell_atoms()["cart_pos"]  # atom positions in cartesian/direct coords
        lattice_vectors = crys.unit_cell.lattice  # lattice vectors as 3x3 array
        charge = metadata.get("charge", 0.0)
        forces = metadata.get("forces", [[0.0, 0.0, 0.0] for _ in labels])
        
        # if learning lattice energy, remove conformational energy
        if self.remove_conformational_energy:
            asym_nmols = (
                self.override_asym_nmols
                if self.override_asym_nmols is not None
                else len(crys.asym_mols())
            )
            try:
                ref_energy -= (
                    self.conformational_energies[structure.molecule_id]
                    * (nmols / asym_nmols)
                )
            except KeyError:
                raise Exception(
                    f"remove_conformational_energy = True but conformational_energy for "
                    f"{structure.molecule_id} is not set. Please set it under "
                    f"[conformational_energies] in cspy.toml file."
                )
            # TODO remove intramolecular forces, if needed

        # delta learning difference between csp and ref energies
        # if using dmacrys for csp energy, the ref energy must be lattice energy
        # i.e. provide conformational energy
        if self.delta_learning:
            # calculate difference in energy
            csp_total_energy = (csp_energy / EV2KJ_PER_MOL) * nmols
            LOG.info(
                f"csp energy = {csp_total_energy}; "
                f"ref energy = {ref_energy} "
            )
            ref_energy -= csp_total_energy
            # TODO: calculate difference in forces, 
            # may require redoing the CSP calculation, since it is not stored in CSP db

        atom_records = [
            (element, *position, *force)
            for element, position, force in zip(elements, positions, forces)
        ]

        LOG.info(f"adding structure {structure.name} (energy = {ref_energy}) to ref data")
        self.reference_dataset[structure.name] = {
            'energy': ref_energy,
            'charge': charge,
            'lattice_vectors': lattice_vectors,
            'atom_records': atom_records,
        }
        self.write_training_data()  # save ref data each step so don't have to recalculate
        
        return ref_energy
        

    def calculate_uncertainty(self, result: Union[Crystal, ML_ThresholdStructure]) -> float:
        """
        calculate uncertainty to compare against uncertainty threshold for ref calcs

        Args:
            result (Crystal | ML_ThresholdStructure): result of the cNNP calculation

        Returns:
            float: uncertainty value based on the specified metric
        """
        from cspy.util.constants import EV2KJ_PER_MOL
        
        if isinstance(result, Crystal):
            crystal = result
            nmols = len(crystal.unit_cell_molecules())
            natoms = len(crystal.unit_cell_atoms()['label'])
            en_std_dev_kjmol_per_molecule = crystal.properties.get(
            "energy_std_dev", None
            )
            forces_std_dev = crystal.properties.get("forces_std_dev", None)
        elif isinstance(result, ML_ThresholdStructure):
            crystal = Crystal.from_shelx_string(result.file_content)
            en_std_dev_kjmol_per_molecule = result.metadata.get(
            "energy_std_dev", None
            )
            forces_std_dev = result.metadata.get("forces_std_dev", None)
            nmols = len(crystal.unit_cell_molecules())
            natoms = len(crystal.unit_cell_atoms()['label'])
            
        if en_std_dev_kjmol_per_molecule is None:
            return BIG_UNCERTAINTY
        if self.uncertainty_metric == "kjmol_per_molecule":
            return en_std_dev_kjmol_per_molecule
        elif self.uncertainty_metric == "total_energy":
            return en_std_dev_kjmol_per_molecule * nmols
        elif self.uncertainty_metric == "eV_per_atom":
            total_energy_eV = (
            en_std_dev_kjmol_per_molecule * nmols / EV2KJ_PER_MOL
            )
            return total_energy_eV / natoms
        elif self.uncertainty_metric == "forces":
            return forces_std_dev
        else:
            raise NotImplementedError(
            f"Uncertainty metric not supported: {self.uncertainty_metric}"
            )

    def run(self) -> None:
        """
        Run training program loop. Once added the specified new number of data points
        trains new weights and writes them.
        """
        self.read_in_unique_structures_from_db()
        if self.training_method == "highest_uncertainty_FPS":
            self.calculate_candidate_descriptors()
        num_datapoints = 0
        LOG.info(
            f"The size of candidate pool at the beginning of run: {len(self.candidates)}"
        )
        while len(self.candidates) > 0 and num_datapoints < self.target_dataset_size:
            self.add_ref_data()
            if self.finished:
                LOG.info("No further datapoints to train on, finishing run.")
                break
            num_datapoints = len(self.reference_dataset)
            LOG.info(f"Training network with {num_datapoints} datapoints")
            self.train_network()
            if not self.trained_weights:
                self.trained_weights = True


class CandidateStructure():
    """
    class for cadidate structure data for training MLP from CSP landscapes
    """
    def __init__(
        self,
        name: str,
        dbpath: str,
        csp_energy: float,
        data: str,
        molecule_id: str = None,
        low_mem: bool = False,
    ) -> None:
        """
        Args:
            name (str): name of the structure, e.g. 'CSP-1234'
            dbpath (str): path to the database where the structure is stored
            csp_energy (float): energy of the structure from CSP calculation
            data (str): file content of the structure, if low_mem is False (default)
            molecule_id (str): molecule ID of the structure, if not provided, it is derived
                               from the name by splitting at '-' and replacing '.' with '_'
            low_mem (bool): if True, do not read file_content from db, only energy and molecule_id.
                            Useful for large datasets to save memory.
        """
        self.dbpath = dbpath
        self.name = name
        if molecule_id:
            self.molecule_id = molecule_id
        else:
            self.molecule_id = name.split('-')[0].replace('.', '_')
        self.csp_energy = csp_energy
        if data is not None:
            self._file_content = data[0]
        elif not low_mem:
            self._file_content = self._get_file_content_from_db()
        
    @property
    def file_content(self) -> str:
        """Get file content of the structure."""
        if self._file_content:
            return self._file_content 
            
        return self._get_file_content_from_db()
    
    def _get_file_content_from_db(self) -> str:
        """get SHELX content from datastore"""
        ds = CspDataStore(self.dbpath)
        file_content = ds.query(
            "select file_content from crystal where id='{}'".format(self.name)
        ).fetchone()[0]
        ds.close()
        
        return file_content


def main():
    import argparse

    parser = argparse.ArgumentParser(help="Train a neural network potential using CSP data.")

    parser.add_argument(
        "dbs_dir",
        type=str,
        help="path to the directory containing reference data CSP dbs",
    )

    parser.add_argument(
        "network_type",
        choices=["single", "committee"],
        default="single",
        type=str,
        help="type of neural network to train",
    )
    parser.add_argument(
        "-m",
        "--training-method",
        default="sequential",
        choices=["sequential", "highest_uncertainty", "highest_uncertainty_FPS"],
        help="active learning method",
    )
    parser.add_argument(
        "-j",
        "--ncores",
        default=4,
        type=int,
        help="number of cores for parallel processes",
    )

    parser.add_argument(
        "-b",
        "--nbins",
        default=500,
        type=int,
        help="number of bins for symmetry function histogram",
    )

    parser.add_argument(
        "--eval-quantity",
        choices=["energy", "forces", "combined"],
        default="energy",
        help="Criterion for choosing best epoch during training each neural network",
    )

    parser.add_argument(
        "--eval-metric",
        choices=["MAE", "RMSE"],
        default="MAE",
        help="metric for evaluating best epoch during training each neural network",
    )

    parser.add_argument(
        "-n",
        "--normalize_dataset",
        action="store_true",
        default=False,
        help="run nnp-norm on dataset before training",
    )
    parser.add_argument(
        "--energy-cutoff",
        type=float,
        default=None,
        help="only train on structures with energy less than cutoff energy",
    )

    parser.add_argument(
        "--energy-window",
        type=float,
        default=None,
        help=(
            "Only train on structures within energy window from the global minimum "
            "of the corresponding database."
        ),
    )

    parser.add_argument(
        "--delta-learning",
        action="store_true",
        default=False,
        help="train NNP by delta learning, i.e. diff between CSP energy and ref method",
    )
    parser.add_argument(
        "--randomize-dataset",
        action="store_true",
        default=False,
        help="randomize order of structures from database",
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
    
    kwargs = vars(args)
    dbs_dir = os.path.abspath(kwargs.pop("dbs_dir"))
    network_type = kwargs.pop("network_type")
    train_interval = config.get("ml_threshold.train_interval", 20)
    target_dataset_size = config.get("ml_threshold.ref_data_target", 2000)
    uncertainty_threshold = config.get(
        "ml_threshold.cNNP_uncertainty_threshold", 0.3
    )
    on_the_fly = config.get("ml_threshold.on_the_fly", False)

    trainer = CSP_NNP_Trainer(
        dbs_dir=dbs_dir,
        network_type=network_type,
        training_method=args.training_method,
        train_interval=train_interval,
        target=target_dataset_size,
        uncertainty_threshold=uncertainty_threshold,
        **kwargs
    )

    LOG.info(f'training Behler-Parrinello NN potential using dbs in {dbs_dir}')
    LOG.info(f'training method: {trainer.training_method}')
    LOG.info(f'train interval: {trainer.train_interval}')
    LOG.info(f'target dataset size: {trainer.target_dataset_size}')
    LOG.info(f'uncertainty threshold: {trainer.uncertainty_threshold}')
    LOG.info(f'network type: {trainer.network_type}')
    LOG.info(f'ncores: {trainer.ncores}')
    LOG.info(f'normalize dataset: {trainer.normalize_dataset}')
    if trainer.energy_cutoff is not None:
        LOG.info(f'energy cutoff: {trainer.energy_cutoff}')
    if trainer.energy_window is not None:
        LOG.info(f'energy window: {trainer.energy_window}')
    LOG.info(f'delta learning: {trainer.delta_learning}')
    LOG.info(f'randomize dataset: {trainer.randomize_dataset}')
    LOG.info(f'csp dbs {trainer.csp_dbs}')

    if on_the_fly:
        LOG.info("Running on-the-fly training mode.")
        trainer.run_on_the_fly()
    else:
        LOG.info("Running standard training mode.")
        trainer.run()

if __name__ == "__main__":
    main()
