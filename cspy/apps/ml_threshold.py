import logging
import sys
from pprint import pformat

import numpy as np

from cspy.apps.setup_app import check_multipoles_valid
from cspy.apps.dma import generate_combined_name
from cspy.chem import Molecule
from cspy.configuration import CONFIG, CspyConfiguration
from cspy.distributed.ml_threshold_manager import ML_Threshold
from cspy.potentials import available_potentials
from cspy.util.logging_config import DATEFMT, FORMATS
from cspy.util.path import Path
from cspy import Molecule

LOG = logging.getLogger(__name__)


def get_mc_setting() -> dict:
    """
    Parse cspy.toml Monte Carlo settings and set defaults.

    Returns:
        dict: Dictionary containing Monte Carlo parameters
    """
    config = CspyConfiguration()
    mc_para = {
        "trial_step": config.get(
            "mc.num_steps"
        ),  # Number of maximum steps for one threshold trial
        "sat_expand": config.get(
            "mc.sat_expand"
        ),  # Whether to sat-expand after perturbation
        "move_all": config.get(
            "mc.move_all"
        ),  # Whether to apply all available types of move at one perturbation, don't use unless specific interest
        "auto_prob": config.get(
            "mc.auto_prob"
        ),  # Calculate probability of choosing move type according to degrees of freedom
        "auto_cutoff": config.get(
            "mc.auto_cutoff"
        ),  # Calculate cutoff of move type, currently only applied to volume expansion and contraction based on number of molecules in unit cell
        "on_the_fly": config.get(
            "mc.on_the_fly"
        ),  # Whether to use on-the-fly clustering, if True, continue_running will always be False
        "cluster_s": config.get("mc.cluster_s"),
        "move": config.get("mc.move"),  # Specified move details
        "move_scale": config.get(
            "mc.move_scale"
        ),  # Scale of move step size with lid states increasing
        "dump_accept": config.get(
            "basin_hopping.dump_accept"
        ),  # Only dump accepted structures if True
        "interval_para": config.get(
            "threshold.interval_para"
        ),  # Parameters for deciding intervals (number of steps) under each lid state
        "increase_para": config.get(
            "threshold.increase_para"
        ),  # Parameters for deciding increase energy at each lid
        "minimize_s": config.get(
            "threshold.minimize"
        ),  # Minimize after perturbation for basin information
        "min_energy": config.get(
            "threshold.min_energy"
        ),  # Set minimal energy for threshold, would use minimized energy of initial structure if not
        "distributed": config.get(
            "mc.distributed", False
        ),  # Run Monte-Carlo Trajectories distributed over cores
        "singlepoint_method": config.get(
            "mc.singlepoint_method", "dmacrys"
        ),  # The method to use for calculating singlepoints of perturbed structures
    }
    return mc_para


def get_ml_threshold_settings() -> dict:
    """
    Parse cspy.toml ml_threshold settings and set defaults.

    Returns:
        dict: Dictionary containing machine learning threshold parameters
    """
    config = CspyConfiguration()
    ml_para = {
        "dbs_dir": config.get(
            "ml_threshold.dbs_dir", None
        ),  # directory for storing reference data dbs
        "reference_method": config.get(
            "ml_threshold.reference_method", None
        ),  # reference energy method to use
        "nnp_energy_units": config.get(
            "ml_threshold.nnp_energy_units", "eV"
        ),  # units the NNP predicts energies in
        "cNNP_uncertainty_threshold": config.get(
            "ml_threshold.cNNP_uncertainty_threshold", 0.3
        ),  # threshold of std dev for running ref calculation when using cNNP
        "conformational_energy": config.get(
            "ml_threshold.conformational_energy", None
        ),  # if rigid, conformational_energy to be subtracted from ref total energy
    }
    return ml_para


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run ML Threshold.")
    parser.add_argument(
        "xyz_files",
        type=str,
        nargs="+",
        help="XYZ files containing molecules for generation",
    )
    parser.add_argument(
        "-r",
        "--res_files",
        type=str,
        nargs="+",
        help="Res files containing molecules for generation",
    )
    parser.add_argument(
        "-c",
        "--charges",
        type=str,
        default=CONFIG.get("csp.charges_file"),
        help="Rank0 multipole file",
    )
    parser.add_argument(
        "-m",
        "--multipoles",
        type=str,
        default=CONFIG.get("csp.multipoles_file"),
        help="RankN multipole file",
    )
    parser.add_argument(
        "-g",
        "--spacegroups",
        type=str,
        default=CONFIG.get("csp.spacegroups"),
        help="Spacegroup for structure generation",
    )
    parser.add_argument(
        "-a",
        "--axis",
        type=str,
        default=None,
        help="Axis filename for structure minimization",
    )
    parser.add_argument(
        "-t",
        "--nthreads",
        type=int,
        default=1,
        help="Number of dask worker threads per process to use for local scheduler",
    )
    parser.add_argument(
        "-rs",
        "--random-seed",
        type=int,
        default=None,
        help="Specify seed for random generator",
    )
    parser.add_argument(
        "-p",
        "--potential",
        default="fit",
        choices=available_potentials.keys(),
        help="Intermolecular potential name",
    )
    parser.add_argument(
        "--cutoff",
        default=CONFIG.get("csp.cutoff"),
        help="Dmacrys real space/repulsion-dispersion cutoff",
    )
    parser.add_argument(
        "--status-file",
        default=CONFIG.get("csp.status_file"),
        type=str,
        help="Specify output status file",
    )
    parser.add_argument(
        "--trials-per-res",
        type=int,
        default=1,
        help="Number of trials/trajectories initiated for each res provided",
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

    default_name = generate_combined_name(args.xyz_files)

    mc_para = get_mc_setting()
    ml_para = get_ml_threshold_settings()

    worker_data = {
        "minimization": {},
    }

    config = CspyConfiguration()

    if (
        mc_para["singlepoint_method"] == "dmacrys"
        or ml_para["reference_method"] == "dmacrys"
    ):
        if args.axis is None:
            args.axis = f"{default_name}.mols"
            LOG.info("No axis file provided, trying %s", args.axis)
        if args.charges is None:
            args.charges = f"{default_name}_rank0.dma"
            LOG.info("No charges file provided, trying %s", args.charges)
        if args.multipoles is None:
            args.multipoles = f"{default_name}.dma"
            LOG.info("No multipole file provided, trying %s", args.multipoles)
        if not check_multipoles_valid(
            args.xyz_files,
            args.axis,
            args.charges,
            args.multipoles,
            args.potential,
        ):
            LOG.error("No good matches for given multipoles! Exiting...")
            sys.exit(1)

        if args.cutoff == "calculate":
            cutoff = 15.0
            from scipy.spatial.distance import pdist

            for x in args.xyz_files:
                mol = Molecule.load(x)
                cutoff = max(cutoff, 1.5 * np.max(pdist(mol.positions)))
        else:
            cutoff = float(args.cutoff)

        config.set("neighcrys.potential", args.potential)

        elec_option = config.get("csp_minimization_step")[-1]["electrostatics"]
        if elec_option == "charges":
            electrostatics = Path(args.charges).read_text()
        elif elec_option == "multipoles":
            electrostatics = Path(args.multipoles).read_text()
        else:
            raise ValueError(
                f"Unknown electrostatics option: {elec_option}. "
                "Choose 'charges' or 'multipoles'."
            )

        worker_data.update(
            {
                "charges": Path(args.charges).read_text(),
                "multipoles": Path(args.multipoles).read_text(),
                "electrostatics": electrostatics,
                "axis": Path(args.axis).read_text(),
                "minimization": {
                    "neighcrys": {
                        "vdw_cutoff": cutoff,
                        "potential": args.potential,
                    },
                    "pmin": {"timeout": config.get("pmin.timeout")},
                    "dmacrys": {"timeout": config.get("dmacrys.timeout")},
                },
            }
        )

    bonds = []
    bondlength_cutoffs = {}

    for xyz_file in args.xyz_files:
        mol = Molecule.load(xyz_file)
        mol.guess_bonds()
        bonds.append(mol.bonds.todense())
        for atom_pair, cutoff in mol.neighcrys_bond_cutoffs().items():
            if atom_pair not in bondlength_cutoffs or cutoff > bondlength_cutoffs[atom_pair]:
                bondlength_cutoffs[atom_pair] = cutoff

    mc_para["bonds"] = bonds

    worker_data.update({
        "check_single_point_energy": config.get("csp.check_single_point_energy"),
        "bondlength_cutoffs": bondlength_cutoffs,
        "mc": mc_para,
    })
    
    worker_data.update(ml_para)

    worker_data["minimization"]["minimization_steps"] = config.get(
        "csp_minimization_step"
    )
    LOG.info(
        "Minimization settings: %s",
        pformat(worker_data["minimization"])
    )

    res_files = []
    for res in args.res_files:
        res_files.extend([res for _ in range(args.trials_per_res)])

    dbs_dir = worker_data["dbs_dir"]
    if dbs_dir is None:
        raise ValueError(
            "Directory for DFT databases is required (toml: ml_threshold.dbs_dir)"
        )

    manager = ML_Threshold(
        name=default_name,
        spacegroups=args.spacegroups,
        res_files=res_files,
        worker_data=worker_data,
        status_file=args.status_file,
        random_seed=args.random_seed,
        dbs_dir=dbs_dir,
    )

    manager.run()


if __name__ == "__main__":
    main()
