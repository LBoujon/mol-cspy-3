import argparse
from cspy.db import CspDataStoreMPS
from cspy.sauce.clustering import prepare_data, get_pair_obj, make_fake_bond
from ccdc.descriptors import MolecularDescriptors as MD
import multiprocessing
import logging

LOG = logging.getLogger(__name__)

def compare_pairs(pairB : str, pair_dataB : dict, pairsA : str, pair_dataA : dict, matches : dict) -> None:
    """ Overlay two pairs and calculate their rmsd.
    If rmsd is less than 0.6, we treat them as equivalent.

    Parameters
    ----------
    pairB : str
        ID of reference molecular pair

    pair_dataB : dict
        Dictionary where the keys are the names of pairs and the values 
        are another dictionary containing mol2 strings and the indices of
        fake bond pairs

    pairA : str
        ID of molecular pair to compare against

    pair_dataA : dict
        Dictionary where the keys are the names of pairs and the values 
        are another dictionary containing mol2 strings and the indices of
        fake bond pairs

    matches : dict
        Dictionary where Keys are pairB and the values are the name of
        a matching pair.

    """

    pair_data1 = pair_dataB[pairB]
    pair1, mol2_string, xyz_string, fake_bond_pair_inds1, cog_diff = get_pair_obj(None, None, pair_data1['mol2'], pair_data1['fake_bond_pair_inds'])
    pair1 = make_fake_bond(pair1, fake_bond_pair_inds1)
    for pairA in pairsA:
        pair_data2 = pair_dataA[pairA]
        pair2, mol2_string, xyz_string, fake_bond_pair_inds2, cog_diff = get_pair_obj(None, None, pair_data2['mol2'], pair_data2['fake_bond_pair_inds'])
        pair2 = make_fake_bond(pair2, fake_bond_pair_inds2)
        
        try:
            molecule, rmsd, rmsd_tanimoto, transformation = MD.overlay_rmsds_and_transformation(pair1, pair2, with_symmetry=True, atoms=None)
        except:
            LOG.info("Failed to overlay. Setting rmsd to 1000.")
            rmsd = 1000

        if rmsd < 0.6:
            matches[pairB] = pairA
            break


def main(sys_args=None):
        parser = argparse.ArgumentParser(
                prog='cspy-mps crossref',
                description='Check to see whether pairs in database B appear in database A')
        parser.add_argument('databaseA', nargs=1, type=str,
                help='Specify database file to read')
        parser.add_argument('databaseB', nargs=1, type=str,
                help='Specify database file to read')
        parser.add_argument('-np', '--numproc', nargs=1, type=int,
                help='Specify number of paralllel process')
        parser.add_argument('-d', '--dump', action='store_true',
                help='Dump pairs')

        args = parser.parse_args(sys_args)

        logging.basicConfig(
        level=args.log_level,
        format='%(asctime)s - %(levelname)s - %(module)s %(lineno)d - '
               '%(message)s')

        workers = args.numproc[0]

        db_nameA = args.databaseA[0]
        dbA = CspDataStoreMPS(db_nameA)

        pairsA = []
        for row in dbA.select("pairs",
                                ['id']):
            pairsA.append(row[0])
        dbA.disconnect()

        db_nameB = args.databaseB[0]
        dbB = CspDataStoreMPS(db_nameB)
        pairsB = []
        manager = multiprocessing.Manager()
        matches = manager.dict()
        for row in dbB.select("pairs",
                                ['id']):
            pairsB.append(row[0])
            matches[row[0]] = None
        dbB.disconnect()

        pair_dataA = manager.dict()
        pair_dataB = manager.dict()

        properties = dict()
        unique_molecules = dict()

        with multiprocessing.Pool(workers) as pool:
                pool.starmap(prepare_data, [[db_nameA, pairsA, Q, workers, properties, unique_molecules, pair_dataA] for Q in range(workers)])

        with multiprocessing.Pool(workers) as pool:
                pool.starmap(prepare_data, [[db_nameB, pairsB, Q, workers, properties, unique_molecules, pair_dataB] for Q in range(workers)])

        with multiprocessing.Pool(workers) as pool:
                pool.starmap(compare_pairs, [[pairsB[Q], pair_dataB, pairsA, pair_dataA, matches] for Q in range(len(pairsB))])

        dbA = CspDataStoreMPS(db_nameA)
        for ind, pairB in enumerate(matches.keys()):
                match = matches[pairB]
                if match == None:
                      LOG.info("Failed to find %s pair.", pairB)
                else:
                        frequency, xyz = dbA.query("select frequency,xyz_coordinates from pairs where id = '" + match + "';").fetchall()[0]
                        LOG.info("Found match with frequency: %s", frequency)
                        if args.dump:
                                with open('pair_' + str(ind) + '.xyz', 'w') as f:
                                        f.writelines(xyz)                
