import logging
import os
import time
from copy import deepcopy
from typing import List, Union, Generator

import numpy as np

from cspy.crystal import Crystal
from cspy.db.key import CspDatabaseId
from cspy.db import CspDataStore
from cspy.cspympi.ml_threshold_tasks import ML_ThresholdStructure
from cspy.ml.nnp.format.parse_n2p2_data import *
from cspy.util.constants import EV2KJ_PER_MOL, HA2KJ_PER_MOL

LOG = logging.getLogger(__name__)


class SuppressStdoutStderr(object):
    """
    A context manager for doing a "deep suppression" of stdout and stderr in 
    Python, i.e. will suppress all print, even if the print originates in a 
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).      
    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


def ev_to_kjmol(energy: float) -> float:
    """convert energy in eV to kJ/mol
    
    Args:
        energy (float): energy in eV

    Returns:
        float: energy in kJ/mol
    """
    return (energy * EV2KJ_PER_MOL)


def kjmol_to_ev(energy: float) -> float:
    """convert energy in kJ/mol to eV
    
    Args:
        energy (float): energy in kJ/mol

    Returns:
        float: energy in eV
    """
    return energy / EV2KJ_PER_MOL


def Ha_to_kjmol(energy: float) -> float:
    """convert energy in Ha to kJ/mol

    Args:
        energy (float): energy in Ha
        
    Returns:
        float: energy in kJ/mol
    """
    return (energy * HA2KJ_PER_MOL)

energy_to_kjmol = {
    'eV': ev_to_kjmol,
    'Ha': Ha_to_kjmol,
    'kjmol': lambda en: en, # assuming cspy kjmol per molecule
}


def n2p2_binary_committee_predict(crystal: Crystal, **kwargs) -> Crystal:
    """Predict energy and forces using N2P2 committee network.

    Args:
        crystal (Crystal): Crystal object to predict energy and forces for.
        **kwargs: Additional keyword arguments, such as:
            - save_forces (bool): If True, save forces in the crystal properties.

    Returns:
        Crystal: New Crystal object with predicted energy and forces and 
        corresponding standard deviations saved in properties.
    """
    from tempfile import TemporaryDirectory
    from cspy.configuration import CspyConfiguration
    import os
    import shutil

    working_dir = os.getcwd()
    config = CspyConfiguration()

    # Get committee directory
    n2p2_committee_dir = config.get('ml_threshold.committee_dir')
    n2p2_inputnn_filepath = config.get('ml_threshold.inputnn_path')
    n2p2_scaling_filepath = config.get('ml_threshold.scaling_path')
    energy_units = config.get('ml_threshold.nnp_energy_units', 'eV')

    save_forces = kwargs.get('save_forces', False)

    committee_network_dirs = sorted([
        os.path.join(n2p2_committee_dir, f)
        for f in os.listdir(n2p2_committee_dir)
        if os.path.isdir(os.path.join(n2p2_committee_dir, f))
    ])

    committee_energies = []
    committee_forces = []

    with TemporaryDirectory(prefix="/dev/shm/") as tmpdirname:
        os.chdir(tmpdirname)

        # Convert structure to input.data
        to_n2p2_input_data_file(crystal)
        # Write/copy input.nn, weights and scaling
        shutil.copy(n2p2_inputnn_filepath, tmpdirname)
        shutil.copy(n2p2_scaling_filepath, tmpdirname)

        try:
            for member_dir in committee_network_dirs:
                shutil.copytree(member_dir, tmpdirname, dirs_exist_ok=True)

                # Run nnp-predict
                os.system('nnp-predict 0 &> predict.log')

                new_crystal = from_n2p2_input_data_file('output.data')[0]
                committee_energies.append(new_crystal.properties['energy'])
                committee_forces.append(new_crystal.properties['atom_forces'])
        except Exception:
            LOG.debug('N2P2 prediction failed')
            return None
        finally:
            os.chdir(working_dir)

    energy = np.mean(committee_energies)
    energy_std_dev = np.std(committee_energies)

    forces = None
    forces_std_dev = None
    if save_forces:
        forces = np.mean(committee_forces, axis=0)
        forces_std_dev = np.std(committee_forces, axis=0)

    new_crystal = deepcopy(crystal)
    new_crystal.properties['raw_energy'] = energy
    new_crystal.properties['energy'] = energy_to_kjmol[energy_units](
        energy
    ) / len(crystal.unit_cell_molecules())
    new_crystal.properties['energy_std_dev'] = energy_to_kjmol[energy_units](
        energy_std_dev
    ) / len(crystal.unit_cell_molecules())
    new_crystal.properties['raw_energy_std_dev'] = energy_std_dev
    new_crystal.properties['atom_forces'] = forces
    new_crystal.properties['forces_std_dev'] = forces_std_dev

    return new_crystal


def n2p2_pynnp_committee_predict(
    crystal: Crystal, results_dict=None, **kwargs
) -> Crystal:
    """Predict energy and forces using N2P2 committee network with pynnp.

    Args:
        crystal (Crystal): Crystal object to predict energy and forces for.
        results_dict (dict): Dictionary to store the results.
        **kwargs: Additional keyword arguments, such as:
            - save_forces (bool): If True, save forces in the crystal properties.

    Returns:
        Crystal: New Crystal object with predicted energy and forces and
        corresponding standard deviations saved in properties.
    """
    from tempfile import TemporaryDirectory
    from cspy.configuration import CspyConfiguration
    import os
    from cspy.ml.n2p2_ase_calculator.n2p2_tools import (
        CNN,
        n2p2_committee_calculator_string,
    )

    if results_dict is None:
        results_dict = {}

    working_dir = os.getcwd()
    config = CspyConfiguration()

    # get committee directory
    n2p2_committee_dir = config.get('ml_threshold.committee_dir')
    n2p2_inputnn_filepath = config.get('ml_threshold.inputnn_path')
    n2p2_scaling_filepath = config.get('ml_threshold.scaling_path')
    energy_units = config.get('ml_threshold.nnp_energy_units', 'eV')

    save_forces = kwargs.get('save_forces', False)

    # create temporary directory
    with TemporaryDirectory(prefix="/dev/shm/") as tmpdirname:
        os.chdir(tmpdirname)
        # convert structure to input.data
        inp_string = to_n2p2_input_data_str(crystal)

        # create CNN object
        try:
            cnn = CNN(
                input_nn=n2p2_inputnn_filepath,
                scaling_data=n2p2_scaling_filepath,
                top_folder=n2p2_committee_dir,
                silent=kwargs.get("silent", False),
            )
            energy, forces, energy_std_dev, forces_std_dev = n2p2_committee_calculator_string(
                cnn,
                inp_string,
                calculate_forces=save_forces,
            )
        except Exception as e:
            LOG.warning(f"committee N2P2 prediction failed for {crystal}: {e}")
            results_dict['new_crystal'] = crystal
            return
        finally:
            del cnn
            os.chdir(working_dir)

    new_crystal = deepcopy(crystal)
    new_crystal.properties['raw_energy'] = energy
    new_crystal.properties['energy'] = energy_to_kjmol[energy_units](
        energy
    ) / len(crystal.unit_cell_molecules())

    if crystal.properties.get('baseline_energy') is not None:
        new_crystal.properties['energy'] += crystal.properties.get('baseline_energy')
    if crystal.properties.get('conformational_energy') is not None:
        new_crystal.properties['energy'] += crystal.properties.get('conformational_energy')
    new_crystal.properties['energy_std_dev'] = energy_to_kjmol[energy_units](
        energy_std_dev
    ) / len(crystal.unit_cell_molecules())
    new_crystal.properties['raw_energy_std_dev'] = energy_std_dev
    new_crystal.properties['atom_forces'] = forces
    new_crystal.properties['forces_std_dev'] = forces_std_dev

    results_dict['new_crystal'] = new_crystal

    return new_crystal


def n2p2_committee_predict(crystal: Crystal, **kwargs) -> Crystal:
    """Predict energy and forces using N2P2 committee network with multiprocessing.

    Args:
        crystal (Crystal): Crystal object to predict energy and forces for.
        **kwargs: Additional keyword arguments, such as:
            - save_forces (bool): If True, save forces in the crystal properties.

    Returns:
        Crystal: New Crystal object with predicted energy and forces and
        corresponding standard deviations saved in properties.
    """
    import multiprocessing

    manager = multiprocessing.Manager()
    results = manager.dict()

    process = multiprocessing.Process(
        target=n2p2_pynnp_committee_predict,
        args=(crystal, results),
        kwargs=kwargs,
    )

    process.start()
    process.join()

    return results['new_crystal']


def n2p2_predict(crystal: Crystal, **kwargs) -> Crystal:
    """Predict energy and forces using N2P2 single network.
    Args:
        crystal (Crystal): Crystal object to predict energy and forces for.
        **kwargs: Additional keyword arguments, such as:
            - save_forces (bool): If True, save forces in the crystal properties.

    Returns:
        Crystal: New Crystal object with predicted energy and forces saved in properties.
    """
    from tempfile import TemporaryDirectory
    import shutil
    from cspy.configuration import CspyConfiguration
    import os
    working_dir = os.getcwd()
    config = CspyConfiguration()
    n2p2_inputnn_filepath = config.get('ml_threshold.inputnn_path')
    n2p2_scaling_filepath = config.get('ml_threshold.scaling_path')
    n2p2_weights_dir = config.get('ml_threshold.weights_dir')
    energy_units = config.get('ml_threshold.nnp_energy_units', 'eV')

    # create temporary directory
    with TemporaryDirectory(prefix="/dev/shm/") as tmpdirname:
        os.chdir(tmpdirname)
        # convert structure to input.data
        to_n2p2_input_data_file(crystal)

        # write/copy input.nn, weights and scaling
        # (maybe just put these in cspy.toml to begin)
        shutil.copy(n2p2_inputnn_filepath, tmpdirname)
        shutil.copy(n2p2_scaling_filepath, tmpdirname)
        shutil.copytree(n2p2_weights_dir, tmpdirname, dirs_exist_ok=True)

        # run nnp-predict
        os.system('nnp-predict 0 &> predict.log')

        # parse output for energy and forces
        try:
            new_crystal = from_n2p2_input_data_file('output.data')[0]
            energy = new_crystal.properties['energy']
            forces = new_crystal.properties['atom_forces']
        except Exception:
            LOG.debug('N2P2 prediction failed')
            return None
        finally:
            os.chdir(working_dir)
    new_crystal.properties['raw_energy'] = energy
    new_crystal.properties['energy'] = energy_to_kjmol[energy_units](
        energy
    ) / len(crystal.unit_cell_molecules())
    if kwargs.get("save_forces", False):
        new_crystal.properties["atom_forces"] = forces
    # forces parser returns single list for all atoms in fx, fy, fz order
    # have to split into groups of three to get by atom (in order of input.data)
    # the next line works because all the same iter and so each time it gets a
    # value it increments. nifty, right
    # new_crystal.properties['atom_forces'] = list(zip(*(iter(forces),) * 3))

    return new_crystal  # energy, forces # units depend on what trained on
 

def energy_evaluation_MLP(
    structure: ML_ThresholdStructure, **kwargs
) -> ML_ThresholdStructure:
    """
    Evaluate single point energy by machine learned potential using n2p2.

    Args:
        structure (ML_ThresholdStructure): ML_ThresholdStructure object with file_content as shelx
            string of the structure.
        **kwargs: Additional keyword arguments, such as:
            - save_forces (bool): If True, save forces in the structure properties.

    Returns:
        ML_ThresholdStructure: New ML_ThresholdStructure object with predicted energy.
    """
    t1 = time.time()
    if isinstance(structure, list):
        structure = structure[0]

    name = structure.name
    sg = structure.spacegroup
    trial_number = structure.trial_number
    mc_step = structure.mc_step
    ini_res = structure.file_content
    u_index = float("nan")

    # Convert to Crystal object for prediction
    crystal = Crystal.from_shelx_string(structure.file_content)

    try:
        new_crystal = n2p2_predict(crystal, **kwargs)
        energy = new_crystal.properties['energy']
        crystal.properties.update(new_crystal.properties)
    except Exception:
        LOG.info(f'NNP energy evaluation failed for step {mc_step}')
        energy = None

    density = crystal.density
    structure_id = CspDatabaseId.from_components(
        name, "thre", sg, trial_number, 0, mc_step
    )

    mtime = time.time() - t1
    # If changed atom ordering, maybe not needed here
    res = crystal.to_shelx_string(
        titl=f"{structure_id} {energy} {density}"
    )
    xrd = None
    metadata = {
        'raw_energy': crystal.properties.get('raw_energy', float("nan"))
    }
    if kwargs.get('save_forces', False):
        metadata['forces'] = crystal.properties.get("atom_forces", [])

    return ML_ThresholdStructure(
        name=name,
        id=structure_id,
        spacegroup=sg,
        trial_number=trial_number,
        minimization_step=0,
        mc_step=mc_step,
        unique_index=u_index,
        energy=energy,
        density=density,
        file_content=res,
        initial_res=ini_res,
        xrd=xrd,
        time=mtime,
        accept=False,
        metadata=metadata,
    )


def main():
    import argparse
    from multiprocessing import Pool
    from cspy.configuration import CONFIG

    parser = argparse.ArgumentParser(
        description="Predict single point energy for .res, .cif, or .db files."
    )

    parser.add_argument(
        "files",
        type=str,
        nargs="+",
        help=".res, .cif or .db file to predict single point energy",
    )

    parser.add_argument(
        "-s",
        "--save-predictions",
        action="store_true",
        default=False,
        help="Save predictions to CSV.",
    )

    parser.add_argument(
        "-f",
        "--save-forces",
        action="store_true",
        default=False,
        help="Save forces.",
    )

    parser.add_argument(
        "-c",
        "--committee-prediction",
        action="store_true",
        default=False,
        help="Do committee NNP prediction.",
    )

    parser.add_argument(
        "--silent",
        action="store_true",
        default=False,
        help="Silence pynnp.",
    )

    parser.add_argument(
        "--delta-learning",
        action="store_true",
        default=False,
        help="Delta-learning prediction: Add reference energy when making predictions.",
    )

    parser.add_argument(
        "--add-conformational-energy",
        action="store_true",
        default=False,
        help="Add conformational energies to predictions (values are set in cspy.toml file).",
    )

    parser.add_argument(
        "--energy-window",
        default=None,
        help="Energy window from the global minimum if a database is passed for predictions.",
    )

    parser.add_argument(
        "-j",
        "--nprocs",
        type=int,
        default=1,
        help="Number of CPUs.",
    )

    parser.add_argument(
        "--log-level",
        default=CONFIG.get("csp.log_level"),
        help="Log level.",
    )

    args = parser.parse_args()
    kwargs = vars(args)

    logging.basicConfig(
        level=args.log_level,
        format='%(asctime)s - %(levelname)s - %(module)s %(lineno)d - '
               '%(message)s'
    ) 

    if args.save_predictions:
        with open("predictions.csv", "w") as f:
            f.write("structure,zprime,energy,uncertainty,raw_energy,\n")

    def to_primitive(
        fpaths: List[str], energy_window: Union[float, None] = None
    ) -> Generator[Crystal, None, None]:
        """Convert files to primitive crystals.

        Args:
            fpaths (List[str]): List of file paths to .res, .cif or .db files.
            energy_window (Union[float, None]): Energy window for database files.
                If None, no energy window is applied.

        Yields:
            Crystal: Primitive Crystal objects with properties set.
        """
        for fpath in fpaths:
            extension = fpath[fpath.rfind('.'):]
            if extension in ['.res', '.cif']:
                try:
                    crys = Crystal.load(fpath)
                except Exception:
                    LOG.info(f"Couldn't load {fpath}")
                    continue

                if crys.space_group.centering == "primitive":
                    crys = crys.as_P1()
                else:
                    crys = crys.as_primitive_P1()
                yield crys

            elif extension == '.db':
                try:
                    ds = CspDataStore(fpath)
                    max_energy = ds.query(
                        'select max(energy) from crystal'
                    ).fetchone()[0]
                    if energy_window:
                        min_energy = ds.query(
                            'select min(energy) from crystal'
                        ).fetchone()[0]
                        max_energy = min_energy + float(energy_window)
                    rows = ds.query(
                        f'select id, file_content, molecule_id, energy '
                        f'from crystal where energy < {max_energy} '
                        f'order by energy'
                    ).fetchall()
                except Exception:
                    LOG.info(f"Couldn't load {fpath} database.")
                    continue

                for id_, file_content, molecule_id_, energy_ in rows:
                    try:
                        crys = Crystal.from_shelx_string(file_content)
                    except Exception:
                        LOG.info(
                            f"Couldn't load {id_} from {fpath} database."
                        )
                        continue

                    if molecule_id_.startswith('Empty '):
                        molecule_id_ = id_.split('-')[0]

                    if crys.space_group.centering == "primitive":
                        crys = crys.as_P1()
                    else:
                        crys = crys.as_primitive_P1()

                    if args.delta_learning:
                        crys.properties['baseline_energy'] = energy_
                    else:
                        crys.properties['baseline_energy'] = 0

                    if args.add_conformational_energy:
                        try:
                            conf_energy = CONFIG.get(
                                'conformational_energies'
                            )[molecule_id_]
                        except KeyError:
                            raise Exception(
                                f'{molecule_id_} was not found in '
                                'conformational_energies in cspy.toml file.'
                            )
                    else:
                        conf_energy = 0
                    crys.properties['conformational_energy'] = conf_energy

                    yield crys

            else:
                LOG.info(
                    f"File {fpath} with {extension} extension is not supported. Skipping it."
                )


    with Pool(args.nprocs) as pool:
        results = pool.map(
            n2p2_pynnp_committee_predict,
            to_primitive(args.files, args.energy_window)
        )


    names = []
    for fpath in args.files:
        extension = fpath[fpath.rfind('.'):]
        if extension in ['.res', '.cif']:
            base_name = os.path.basename(fpath)
            name, _ = os.path.splitext(base_name)
            names.append(name)
        elif extension == '.db':
            ds = CspDataStore(fpath)
            max_energy = ds.query(
                'select max(energy) from crystal'
            ).fetchone()[0]
            if args.energy_window:
                min_energy = ds.query(
                    'select min(energy) from crystal'
                ).fetchone()[0]
                max_energy = min_energy + float(args.energy_window)
            ids = ds.query(
                f'select id from crystal where energy < {max_energy} order by energy'
            ).fetchall()
            for id_tuple in ids:
                names.append(id_tuple[0])
        

    for name, result in zip(names, results):
        try:
            energy = result.properties.get('energy', '')
            energy_std_dev = result.properties.get('energy_std_dev', '')
            raw_energy = result.properties.get('raw_energy', '')
            zprime = len(result.unit_cell_molecules())
        except Exception:
            LOG.warning(f"Prediction failed for {name}")
            continue

        print(
            f"crystal {name} energy {energy} uncertainty {energy_std_dev}"
        )

        if args.save_predictions:
            with open("predictions.csv", "a") as f:
                f.write(
                    f"{name},{zprime},{energy},{energy_std_dev},{raw_energy},\n"
                )   

if __name__ == "__main__":
    main()
