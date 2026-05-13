import sqlite3
import os
from cspy import Crystal
from cspy.db import CspDataStore
from cspy.minimize.mace_minimizer import MaceMinimizer
import logging
from cspy.db.clustering import calculate_missing_xrd

LOG = logging.getLogger(__name__)


def get_ids_in_energy_window(db_file, energy_window):
    ds = CspDataStore(db_file)
    min_energy = ds.query('select min(energy) from crystal').fetchone()[0]

    if min_energy == None:
        return None
    result = ds.query('select * from crystal where energy < {} ORDER BY energy'.format(min_energy+energy_window)).fetchall()

    if result:
        LOG.info(f'{len(result)} structures found in {energy_window} from the global minimum of {db_file}')
        ds.close()
        return result
    else:
        return None


def process_batch(batch, **kwargs):
    input_db_file = kwargs.get("input_db_name", None)
    results_db_file = kwargs.get("results_db_file", None)
    if results_db_file is None or input_db_file is None:
        LOG.info("No results_db_file/input_db_file specified")
        return

    input_ds = CspDataStore(input_db_file)
    batch_ds = CspDataStore(results_db_file)

    if type(batch) is tuple:
        batch = [batch]

    for row in batch:
        ID = row[0]
        file_contents = row[5]
        minimization_step, trial_number = input_ds.query("select minimization_step, trial_number from trial_structure "
                                           "where ID='{}'".format(ID)).fetchone()
        updated_ID = f"{ID[:ID.rfind('-')]}-{minimization_step+1}"
        existing_id = batch_ds.query("SELECT ID FROM crystal WHERE ID='{}'".format(updated_ID)).fetchone()

        if existing_id:
            LOG.info(f'ID {updated_ID} already exists in the results database. Skipping...')
            continue
        try:
            c = Crystal.from_shelx_string(file_contents)
        except:
            LOG.info(f'{ID} cannot be loaded by cspy.Crystal.load() method.', row)
            continue

        try:
            result = c.minimize_with_mace(minimization_step=minimization_step+1,
                    updated_titl =  updated_ID + '_opt_by_mace',
                    **kwargs)

        except:
            LOG.info(f"MaceMinimizer failed for ID {ID}")

        spacegroup = result.space_group.international_tables_number
        density = result.density
        energy = result.properties['lattice_energy']
        file_contents = result.to_shelx_string()
        updated_crystal_row = {
            'ID': updated_ID,
            'spacegroup': spacegroup,
            'density': density,
            'energy': energy,
            'molecule_id': row[4],
            'file_content': file_contents
        }
        batch_ds.insert_one('crystal', updated_crystal_row)
        batch_ds.insert_one('equivalent_to', {'unique_id': updated_ID})

        updated_trial_structure_row = {
            'ID': updated_ID,
            'minimization_step': minimization_step + 1,
            'trial_number' : trial_number,
            'valid': int(result.properties['valid_minimization']),
            'minimization_time': result.properties['minimization_time'],
            'metadata': None
        }
        batch_ds.insert_one('trial_structure', updated_trial_structure_row)

        try:
            x = calculate_missing_xrd(updated_ID, file_contents)
            if x is not None:
                missing_descriptors = {updated_ID: x}
    
            batch_ds.add_descriptors("xrd", missing_descriptors, metadata="{'two_theta': (0, 20), 'sep': 0.02}")
        except:
            LOG.debug('PXRD calculation failed.')
        
    batch_ds.close()
    input_ds.close()


def apply_func(args):
    x, kwargs = args
    return process_batch(x, **kwargs)


def update_kwargs(**kwargs):
    return kwargs


def main():
    import argparse
    import multiprocessing
    parser = argparse.ArgumentParser()
    parser.add_argument("db_files", nargs='+', help="Path to database files")
    parser.add_argument("-e", "--energy_window", type=float, required=True, help="Energy window", default=1)
    parser.add_argument("--fmax", type=float, help="Convergence criterion, Max force. Defaults to 0.05", default=0.05)
    parser.add_argument("-b", "--batch_size", help="Batch size for updating database.", default=10, type=int)
    parser.add_argument("-d", "--device", help="Device", choices=('cpu', 'cuda'), default='cpu', type=str)
    parser.add_argument("-n", "--ncore", help="Number of processors", default=1, type=int)
    parser.add_argument("-s", "--scale_by_element_count",
                        help="atomic number of the element you want to scale the total energy with.",
                        default=0,
                        type=int
                        )
    parser.add_argument("--default_dtype", help="Use float64 for MACECalculator, which is slower but more accurate. "
                        "Recommended for geometry optimization. Use float32 for MACECalculator, which is faster but "
                        "less accurate. Recommended for MD. Default = float64", 
                        default='float64', 
                        type=str, 
                        choices=('float32', 'float64'))
    parser.add_argument("-m", "--mace_model",
                        help="Mace model filename OR foundation models from mace.calculators.",
                        default='mace_mp',
                        type=str
                        )
    parser.add_argument("--include_dispersion", action="store_true", default=False,
                        help="Calculate a single point energy (don't optimize)")
    parser.add_argument("--log-level",
                        default="INFO",
                        choices=("INFO", "DEBUG", "WARN", "ERROR"),
                        help="Logging verbosity",
                        )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
    )

    for input_db_name in args.db_files:
        results_db_file = f"{input_db_name.split('.')[0]}_opt.db"
        kwargs = vars(args)
        kwargs = update_kwargs(dispersion=kwargs.get('include_dispersion'), **kwargs)
        kwargs = update_kwargs(results_db_file=results_db_file, **kwargs)
        kwargs = update_kwargs(input_db_name=input_db_name, **kwargs)
        ds = CspDataStore(input_db_name)
        ds.copy_schema_to(results_db_file)
        results = get_ids_in_energy_window(input_db_name, args.energy_window)
        batches = [results[i:i+args.batch_size] for i in range(0, len(results), args.batch_size)]

        pool = multiprocessing.Pool(processes=args.ncore)
        for batch in batches:
            if type(batch) is tuple:
                batch = [batch]
            pool.map(apply_func, [(b, kwargs) for b in batch])

    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
