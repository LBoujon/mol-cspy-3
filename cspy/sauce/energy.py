from cspy.db import CspDataStoreMPS, CspDataStoreAUs
import sqlite3
import argparse
import sys
import multiprocessing
from cspy.formats.dma import parse_dma
from cspy.chem.energy import BuckinghamPotential
from cspy.chem.molecule import Molecule
from cspy.util.misc import mp_check_reservation
import logging
from typing import Literal

LOG = logging.getLogger(__name__)

def get_au_atom_ids(mol_ids : list, molecules : dict) -> list:
    """The order of the molecules in the list should match the 
    order of the molecules in an associated CSPy Molecules
    object. This function returns the atom IDs which belong
    to each component of the Molecule.

    Parameters
    ----------
    mol_ids : list
        List of molecule IDs

    molecules : dict
        Dictionary where keys are molecule IDs and the values 
        are a dictionary containing molecular properties.
        Critically, num_atoms.

    Returns
    -------
    atom_ids : list of list of iints
        A list which contains one list per molecule in the 
        asymmetric unit. Each sublist contains a list of atom
        ids so that we can know which atom belongs to which
        molecule.
    """
    
    atom_ids = []
    for mol_id in mol_ids:
        mol_num_atoms = molecules[mol_id]['num_atoms']
        if len(atom_ids) == 0:
            prev_mol_num_atoms = 0
        else:
            prev_mol_num_atoms = atom_ids[-1][-1] + 1
        mol_atom_ids = [i for i in range(prev_mol_num_atoms, prev_mol_num_atoms + mol_num_atoms)]
        atom_ids.append(mol_atom_ids)

    return atom_ids


def get_cluster_energy(clusters : list, 
                       Q : int, 
                       workers : int, 
                       properties : dict, 
                       atom_types : list, 
                       charges : list, 
                       molecules : dict, 
                       cluster_type : Literal["MPS", "AUT"], 
                       pot : str, 
                       cluster_energies : dict, 
                       Zp : int) -> None:
    """Iterate through a list of clusters and get their 
    energies, as defined by a Buckingham potential and 
    point-charge electrostatics.

    Parameters
    ----------
    clusters : list
        List of cluster IDs

    Q : int
        Unique worker id for parallel process. Used for mp_check_reservation function

    workers : int
        Number of workers operating in parallel. Used for mp_check_reservation function

    properties : dict
        Dictionary where the keys are names of clusters and the values
        are dictionary. This dictionary contains various properties such
        as xyz_coordinates.

    atom_types : list
        List of list of NEIGHCRYS atom typings. One value per atom.

    charges : list
        List of list of point charges. One value per atom.

    molecules : dict
        Dictionary where keys are molecule IDs and the values 
        are a dictionary containing molecular properties.
        Critically, num_atoms.

    cluster_type : MPS or AUT
        Type of cluster based on where it was sourced from MPS or AUT/UC2AU

    pot : str
        Name of Buckingham potential in mol-CSPy collection

    cluster_energies : dict
        Dictionary where the values are IDs of clusters and values
        are the energies of those clusters in kJ/mol per formula unit

    Zp : int
        Z'/Zp/Z prime is the number of formula units in the cluster.

    """
    
    from cspy.sauce.clg_mps import get_pair_info

    if cluster_type == 'AUT':
        mol_ids = properties[clusters[0]]['molecule_ids'].replace('"', '').replace(',', '').replace('[', '').replace(']', '').split()
        atom_ids = get_au_atom_ids(molecules, mol_ids)

    for ind, cluster in enumerate(clusters):
        # this if statement checks the cluster was intended for this worker
        if mp_check_reservation(ind, Q, workers):
            xyz_string = properties[cluster]['xyz_coordinates']
            cluster_CSPyMol = Molecule.from_xyz_string(xyz_string)
            if cluster_type == 'MPS':
                molecule1, molecule2, mol1_atom_ids, mol2_atom_ids = get_pair_info(properties, molecules, cluster)
                cluster_CSPyMol.def_components([mol1_atom_ids, mol2_atom_ids])

                relevant_atom_types = [atom_types[molecule1], atom_types[molecule2]]
                relevant_charges = [charges[molecule1], charges[molecule2]]
            else:
                cluster_CSPyMol.def_components(atom_ids)
                relevant_atom_types = atom_types
                relevant_charges = charges

            cluster_calcualtor = BuckinghamPotential(cluster_CSPyMol, pot, relevant_atom_types, charges=relevant_charges, Zp=Zp)
            cluster_calcualtor.calc_energy()
            energy = cluster_calcualtor.energy['kJ/mol']
            
            cluster_energies[cluster] = energy


def main(sys_args=None):
        parser = argparse.ArgumentParser(
                prog='cspy-mps cluster',
                description='Read all clusters from database and calculate their energies')
        parser.add_argument('database', nargs=1, type=str,
                help='Specify database file to read')
        parser.add_argument('-d', '--dma', nargs=1, type=str,
                help='Rank0 dma file containing atom types and charges')
        parser.add_argument('-di', '--dmaIDs', nargs="+", type=str,
                help='Provide molecule ids for entries in DMA file.\
                Required for MPS but not AUT')
        parser.add_argument('-Zp', '--Zp', nargs=1, type=str,
                help='Z\' for asymmetric unit. Default=1\
                Important for AUT but not MPS.')
        parser.add_argument('-p', '--pot', nargs=1, type=str,
                help='Name of potential file without extension')
        parser.add_argument('-np', '--numproc', nargs=1, type=int,
                help='Specify number of paralllel process')
        parser.add_argument(
                "--log-level", type=str,
                choices=("INFO", "DEBUG", "ERROR", "WARN"),
                default="INFO",
                help="Control level of logging output")

        args = parser.parse_args(sys_args)

        logging.basicConfig(
        level=args.log_level,
        format='%(asctime)s - %(levelname)s - %(module)s %(lineno)d - '
               '%(message)s')

        db_name = args.database[0]
        workers = args.numproc[0]
        dma_file = args.dma[0]
        pot = args.pot[0]

        con = sqlite3.connect(db_name)
        cursor = con.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        db_tables = cursor.fetchall()

        cluster_type = None

        if tuple(['asym_units']) in db_tables:
            LOG.info("Database is of schema cspy2_aus.sql")
            cluster_type = "AUT"
            if args.Zp:
                Zp = args.Zp
            else:
                Zp = 1
        elif tuple(['pairs']) in db_tables:
            LOG.info("Database is of schema cspy2_mps.sql")
            cluster_type = "MPS"
            if args.dmaIDs:
                dmaIDs = args.dmaIDs
            Zp = 1
        else:
            LOG.error("Database does not match compatible SAUCE schemas.\n Exiting...")
            sys.exit()

        dma_data = parse_dma(dma_file)

        manager = multiprocessing.Manager()
        cluster_energies = manager.dict()
        properties = dict()
        molecules = dict()
        if cluster_type == "MPS":
            # for MPS, we treat atom_types and charges as dict now and convert to list later
            # we don't need all the data for each pair, and we don't know which data until
            # later
            atom_types = dict()
            charges = dict()
            for ind, molecule_dma in enumerate(dma_data):
                atom_types[dmaIDs[ind]] = molecule_dma['atom_types']
                charges[dmaIDs[ind]] = molecule_dma['charges']

            db = CspDataStoreMPS(db_name)
            for row in db.select("pairs",
                                    ['id', "min_energy", "energy_list",
                            "min_density", "max_density", 
                            "median_density", "density_list",
                            "frequency", "xyz_coordinates", "mol2_coordinates", 
                            "asymmetric", "symmetric", 
                            "molecule1", "molecule2"]):
            
                properties[row[0]]={"min_energy" : row[1], "energy_list" : row[2],
                            "min_density" : row[3], "max_density" : row[4], 
                            "median_density" : row[5], "density_list" : row[6],
                            "frequency" : row[7], "xyz_coordinates" : row[8], "mol2_coordinates" : row[9], 
                            "asymmetric" : row[10], "symmetric" : row[11], 
                            "molecule1" : row[12], "molecule2" : row[13]}

            for row in db.select("molecules",
                                    ['id', 'xyz_coordinates','equivalent_atoms']):
                molecules[row[0]] = {'num_atoms' : int(row[1].split('\n')[0]), "xyz_coordinates": row[1], "equivalent_atoms": row[2]} 

            clusters = list(properties.keys())

        if cluster_type == "AUT":

            atom_types = []
            charges = []
            for molecule_dma in dma_data:
                atom_types.append(molecule_dma['atom_types'])
                charges.append(molecule_dma['charges'])

            db = CspDataStoreAUs(db_name)
            for row in db.select("asym_units",
                                    ['id',"energy", "xyz_coordinates", 
                            "molecule_ids"]):
            
                properties[row[0]]={"energy" : row[1], "xyz_coordinates" : row[8], "molecule_ids" : row[9]}

            for row in db.select("molecules",
                                    ['id', 'xyz_coordinates','equivalent_atoms']):
                molecules[row[0]] = {'num_atoms' : int(row[1].split('\n')[0]), "xyz_coordinates": row[1], "equivalent_atoms": row[2]} 

            clusters = list(properties.keys())

        db.disconnect()

        with multiprocessing.Pool(workers) as pool:
            pool.starmap(get_cluster_energy, [[clusters, Q, workers, properties, atom_types, charges, molecules, cluster_type, pot, cluster_energies, Zp] for Q in range(workers)])


        for cluster in properties.keys():
            energy = cluster_energies[cluster]
            properties[cluster]["min_energy"] = energy

        for molecule in molecules:
            del molecules[molecule]['num_atoms']

        new_db_name = db_name[:-3] + '_energies'

        LOG.info("Writing data to database %s", new_db_name)
        if cluster_type == 'AUT':
            from cspy.sauce.extract_aut import create_aus_database, update_aus_database
            create_aus_database(new_db_name)
            update_aus_database(molecules, new_db_name)
            update_aus_database(properties, new_db_name)

        elif cluster_type == 'MPS':
            from cspy.sauce.extract_mps import create_mps_database, set_molecules, add_to_mps_database
            create_mps_database(new_db_name)
            set_molecules(molecules, new_db_name)
            add_to_mps_database(properties, new_db_name)