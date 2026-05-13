import argparse
import logging
import zipfile
import pandas as pd
from cspy.chem import Molecule
from cspy.crystal.generate_crystal import CrystalGenerator
import time
from cspy.apps.setup_app import CspyApp, add_common_arguments, add_CLG_arguments
from cspy.configuration import COMMON_SAMPLING_SETTINGS, CONFIG

LOG = logging.getLogger(__name__)

def initialize_spacegroups(app):
    spacegroups = app.spacegroups
    number_structures = app.number_structures
    spacegroup_sampling = dict()

    if CONFIG.get('sampling_settings', None):
        LOG.debug('Getting sampling settings from TOML file')
        sampling_settings = CONFIG.get('sampling_settings')
        for x, num in sampling_settings.items():
            spacegroup_sampling[int(x)] = num
        
    elif spacegroups in COMMON_SAMPLING_SETTINGS:
        sg = COMMON_SAMPLING_SETTINGS[spacegroups]
        for x in sg["space_group"]:
            spacegroup_sampling[x] = sg["number_structures"][x]
    else:
        if number_structures is None:
            raise ValueError(
                "Must set number of structures for custom spacegroup set"
            )
        try:
            spacegroups = [int(x) for x in spacegroups.split()]
        except ValueError as e:
            LOG.error("Error interpreting requested spacegroups: %s", e)
            raise ValueError("Invalid spacegroup") from e
        for x in spacegroups:
            spacegroup_sampling[x] = number_structures
    
    return spacegroup_sampling        


def main():
    parser = argparse.ArgumentParser(
        description="App for running landscape generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "xyz_files",
        type=str,
        nargs="+",
        help="Xyz files containing molecules for generation",
    )
    parser.add_argument(
        "-s",
        "--initial-seed",
        default=1,
        type=int,
        help="Initial seed for sobol sampling",
    )
    parser = add_CLG_arguments(parser)
    parser = add_common_arguments(parser)
    args = parser.parse_args()

    app = CspyApp(app_name='cspy-clg', args=args, rank=0, size=1)
    app.configure_app(clg=True, minimisation=False, skip_header=args.skip_header)

    n_mols = len(args.xyz_files)
    LOG.info(
        "Generated crystals will have %d molecule%s in their asymmetric unit",
        n_mols, "" if n_mols < 2 else "s",
    )

    seed = args.initial_seed
    default_name = app.default_name
    spacegroup_sampling = initialize_spacegroups(app)
    for sg, num_structures in spacegroup_sampling.items():
        zip_name = f"{default_name}_{sg}.zip"
        start_time = time.time()
        mols = [Molecule.load(x) for x in args.xyz_files]
        clg = CrystalGenerator(mols, sg, nudge=app.args.nudge, adaptcell=app.args.adaptcell, asi=app.args.asi)

        success = 0
        crystals = []
        while success != num_structures:
            crystal = clg.generate(seed)
            if crystal:
                success += 1
                crystals.append((crystal, seed))
            seed += 1
        run_time = time.time() - start_time
        rate = 100 * success / (seed - args.initial_seed)
        results = {"Results": {
            "Accepted": success,
            "Rejected": seed - success - args.initial_seed,
            "Success Rate": f"{rate:.06f}%",
            "Run time": run_time
        }}
        LOG.info("Summary:\n%s", pd.DataFrame(results).T)

        LOG.info("Writing structures to %s", zip_name)
        with zipfile.ZipFile(zip_name, "a") as zip_file:
            for crystal, seed in crystals:
                filename = f"{default_name}_{sg}_{seed}.res"
                titl = f"{default_name} cspy generated sg={sg} seed={seed}"
                shelx_string = crystal.to_shelx_string(titl=titl)
                zip_file.writestr(filename, shelx_string)


if __name__ == "__main__":
    main()
