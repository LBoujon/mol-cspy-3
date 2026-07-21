from cspy.db import CspDataStore, CspDataStoreMPS
from cspy.crystal import Crystal
from cspy.crystal.util import get_contacts
from cspy.chem.molecule import Molecule
from cspy.util.misc import mp_check_reservation
from openbabel import openbabel
import numpy as np
import itertools
import math
from scipy.spatial import cKDTree as KDTree
from cspy.chem.element import chemical_formula
import multiprocessing
import argparse
import os
import random
from typing import Optional, Tuple
import logging

LOG = logging.getLogger(__name__)

def create_mps_database(filename : str = "mps") -> None:
    """Create a blank database"""
    filename = filename + "_mps.db"

    ds = CspDataStoreMPS.create_and_connect(filename)
    ds.disconnect()


def set_molecules(unique_molecules : dict, filename : str = "mps") -> None:
    """
    Define the molecules which consistute the pairs in the database

    Parameters
    ----------
    unique_molecules : dict
        Dictionary where the keys are names of unique molecules and
        values are dictionaries containing xyz_coordinates and
        equivalent_atoms

    filename: str
        Seed of database name. Final name will include _mps.db

    """
    from cspy.db.datastore_writer_mps import DatastoreWriterMPS

    filename = filename + "_mps.db"

    db_writer = DatastoreWriterMPS(
        unique_molecules,
        filename,
    )

    db_writer.run()


def add_to_mps_database(pair_properties : dict, filename : str = "mps") -> None:
    """
    Define the molecules which consistute the pairs in the database

    Parameters
    ----------
    pair_properties : dict
        Dictionary where the keys are names of pairs and
        values are dictionaries containing their properties.
        See db.schemas.cspy2_mps.sql for more details.

    filename: str
        Seed of database name. Final name will include _mps.db

    """
    from cspy.db.datastore_writer_mps import DatastoreWriterMPS

    filename = filename + "_mps.db"

    db_writer = DatastoreWriterMPS(
        pair_properties,
        filename,
    )

    db_writer.run()


def compare_pairs(pairs : dict) -> None:
    """
    Overlay two molecular pairs and if their RMSD is less than 0.3,
    we treat them as equivalent. Note that the RMSD is stricter here
    than in SAUCE.clustering as here we are only comparing pairs from
    a single crystal structure.

    Parameters
    ----------
    pairs : dict
        Dictionary where the keys are names of pairs and
        values are dictionaries containing their properties.

    """

    from cspy.sauce.clustering import get_nearest_neighbours, make_fake_bond
    import ccdc
    from ccdc.descriptors import MolecularDescriptors as MD
    
    unique_pairs = pairs.copy()
    pair_pairs = itertools.combinations(pairs.keys(), 2)
    for pair_pair in pair_pairs:
        pair1 = ccdc.molecule.Molecule.from_string(pairs[pair_pair[0]]["mol2"])
        pair2 = ccdc.molecule.Molecule.from_string(pairs[pair_pair[1]]["mol2"])

        comp1 = pair1.components
        atoms1_1 = comp1[0].atoms
        atoms1_2 = comp1[1].atoms

        fake_bond_pair_inds1 = get_nearest_neighbours(atoms1_1, atoms1_2)
        pair1 = make_fake_bond(pair1, fake_bond_pair_inds1)

        comp2 = pair2.components
        atoms2_1 = comp2[0].atoms
        atoms2_2 = comp2[1].atoms

        fake_bond_pair_inds2 = get_nearest_neighbours(atoms2_1, atoms2_2)
        pair2 = make_fake_bond(pair2, fake_bond_pair_inds2)
        
        try:
            molecule, rmsd, rmsd_tanimoto, transformation = MD.overlay_rmsds_and_transformation(pair1, pair2, with_symmetry=True, atoms=None)
        except:
            continue

        if rmsd < 0.3:
            if pair_pair[0] in unique_pairs.keys():
                del unique_pairs[pair_pair[0]]
                if pairs[pair_pair[0]]['asymm']:
                    unique_pairs[pair_pair[1]]['asymm'] = True
                if pairs[pair_pair[0]]['symm']:
                    unique_pairs[pair_pair[1]]['symm'] = True

    return unique_pairs


def get_mol2(mol1 : Molecule, mol2 : Molecule, i1 : int, i2 : int) -> Tuple:
    """
    Take the two molecules which constitute a pair and store
    them in a mol2 format.

    Parameters
    ----------
    mol1 : Molecule
        CSPy Molecule object defining molecule 1 in the pair

    mol2 : Molecule
        CSPy Molecule object defining molecule 2 in the pair

    i1 : int
        Unique ID for molecule 1

    i2 : int
        Unique ID for molecule 2

    Returns
    -------
    pairname : str
        Unique name for molecular pair

    xyz_coords : str
        Molecular pair in xyz format

    mol2_coords : str
        Molecular pair in mol2 format

    """
    elements = mol1.elements + mol2.elements
    lines = [
            f"{len(mol1) + len(mol2)}",
            chemical_formula(elements, subscript=False),
                ]
    
    for el, (x, y, z) in zip(mol1.elements, mol1.positions):
            lines.append(f"{el} {x: 20.12f} {y: 20.12f} {z: 20.12f}")
    for el, (x, y, z) in zip(mol2.elements, mol2.positions):
            lines.append(f"{el} {x: 20.12f} {y: 20.12f} {z: 20.12f}")

    pairname = ''.join([str(i1), '_', str(i2)])

    xyz_coords = '\n'.join(lines)

    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("xyz", "mol2")

    mol = openbabel.OBMol()
    obConversion.ReadString(mol, xyz_coords)

    mol.AddHydrogens()

    mol2_coords = obConversion.WriteString(mol)

    return pairname, xyz_coords, mol2_coords


def get_pairs(data : list, molecules_elements : dict, Q : int, workers : int, n_symops : int, args : dict) -> None:
    """
    Iterate over all crystals, assign each crystal to a unique worker.
    Then extract all possible molecular pairs from the asymmetric unit
    and cluster down to only unique pairs.

    Parameters
    ----------
    data : List
        List of CSPy Crystal objects

    molecules_elements : dict
        Dictionary where the keys are the names of molecules
        and the values are a list of that molecules elements.

    Q : int
        Unique worker id for parallel process. Used for mp_check_reservation function

    workers : int
        Number of workers operating in parallel. Used for mp_check_reservation function

    n_symops : int
        Number of symmetry operators for the space group of the crystal

    args : dict
        Further arguments from argparse

    """
    num_pair_pairs = 0
    for crys_ind, crystal in enumerate(data):
        # this if statement checks the pair_pair was intended for this worker
        if mp_check_reservation(crys_ind, Q, workers):
            pairs_data = dict()
            molecules = []
            num_pair_pairs = num_pair_pairs + 1
            id, sg, density, energy, mol_id, file_content, _min_step, trial_number, _min_time = crystal

            if args.spacegroup:
                sg = args.spacegroup[0]

            crystal1 = Crystal.from_shelx_string(file_content)

            crystal1.symmetry_unique_molecules()
            num_unique_mols = len(crystal1._symmetry_unique_molecules)
            expected_uc_mols = num_unique_mols * n_symops[str(sg)]
            num_uc_mols = len(crystal1.unit_cell_molecules())
            for mol in crystal1._symmetry_unique_molecules:
                mol_elements = mol.elements
                for unique_molecule in molecules_elements.keys():
                    if mol_elements == molecules_elements[unique_molecule]:
                        # will need to add extra check with overlap for molecules with the same elements
                        molecules.append(str(unique_molecule))
                        break

            # check for Buckingham catastrophes
            if num_unique_mols == args.G[0] and expected_uc_mols == num_uc_mols:
                num_uc_atoms = 0
                for mol in crystal1._unit_cell_molecules:
                    num_uc_atoms += len(mol.positions)

                u = 3
                v = 3
                w = 3
                num_unit_cells = u * v * w
                center_cell = math.floor(num_unit_cells/ 2)
                crystal1_sc = crystal1.as_P1_supercell((u, v, w))
                sc_mols = crystal1_sc.unit_cell_molecules()
                if not len(sc_mols) == (num_uc_mols * num_unit_cells):
                    LOG.error("Failed Buckingham castrophe check")
                    continue
                sc_unique_mol_ids = [(mol_id * num_unit_cells) + center_cell for mol_id in range(num_unique_mols)]
                sc_asymm_mol_ids = []
                for mol_id in range(num_unique_mols):
                    for cell_ind in range(num_unit_cells): 
                        sc_asymm_mol_ids.append((mol_id * num_unit_cells) + cell_ind)
                sc_pos = np.concatenate([mol.positions for mol in sc_mols], axis=0)
                vdws = np.array([i.vdw for mol in sc_mols for i in mol.elements])

                start = 0

                for ind, i in enumerate(sc_unique_mol_ids):
                    n_atoms = len(sc_mols[i].positions)
                    start += n_atoms * center_cell
                    end = start + n_atoms

                    mol_pos = sc_pos[start: end]
                    mol_vdws = vdws[start: end]

                    mask_pos = np.ones(sc_pos.shape, dtype=bool)
                    mask_pos[start: end] = False
                    all_neighs = np.reshape(sc_pos[mask_pos], (-1, 3))

                    mask_vdw = np.ones(vdws.shape, dtype=bool)
                    mask_vdw[start: end] = False
                    neigh_vdws = vdws[mask_vdw]

                    real_contacts, distances, box_idxs, at_col_vectors = get_contacts(mol_pos, mol_vdws, all_neighs, neigh_vdws, tolerance=0.1)

                    if len(real_contacts) == 0:
                        continue
                    # concatenate to get list of unique atomic neighbours
                    unique_contacts = np.unique(np.concatenate(real_contacts, axis=0))
                    # find ids where mask is True (i.e. which atoms are in box)
                    mask_indices = np.where(box_idxs)[0]
                    # get ids of elements in all_neighs where box_idxs are true and appear in unique_contacts
                    unique_contacts = [mask_indices[n] for n in unique_contacts]
                    # find ids where mask is True (i.e. which atoms are not in the reference molecule)
                    mask_indices = np.where(mask_pos[:,0])[0]
                    # get ids of elements in sc_pos where mask_pos are true and appear in unique_contacts
                    unique_contacts = [mask_indices[n] for n in unique_contacts]

                    start2 = 0
                    for i2, mol in enumerate(sc_mols):
                        end2 = start2 + len(mol.positions)
                        atom_ids = np.arange(start2, end2)
                        if len(np.intersect1d(atom_ids, unique_contacts)) > 0:
                            if i2 in sc_asymm_mol_ids:
                                asymmetric = True
                            else:
                                asymmetric = False
                            if args.asymm and not asymmetric:
                                pass
                            else:
                                pairname, xyz_coords, mol2_coords = get_mol2(sc_mols[i], mol, i, i2)
                                # mol1 is always asymmetric mol
                                mol1_asym_ind = int(math.floor(i / num_unit_cells))
                                mol1_id = molecules[mol1_asym_ind]
                                mol2_uc_ind = int(math.floor(i2 / num_unit_cells))
                                # correct mol ind from uc ind to asym ind
                                mol2_asym_ind = int(mol2_uc_ind - (math.floor(mol2_uc_ind / num_unique_mols) * num_unique_mols))
                                mol2_id = molecules[mol2_asym_ind]
                                pairs_data[pairname] = {"xyz" : xyz_coords,
                                                        "mol2" : mol2_coords,
                                                        "asymm" : asymmetric,
                                                        "symm" : not asymmetric,
                                                        "mol1_id" : mol1_id,
                                                        "mol2_id" : mol2_id}

                        start2 = end2

                    start += n_atoms * ((u*v*w) - center_cell)

            else:
                LOG.error("Failed Buckingham castrophe check")

            unique_pairs = compare_pairs(pairs_data)
            pair_properties = {}
            for pair_ind, pair in enumerate(unique_pairs):
                id = str(crys_ind) + '_' + str(pair_ind)
                xyz_coords = pairs_data[pair]["xyz"]
                mol2_coords = pairs_data[pair]["mol2"]
                asymmetric = pairs_data[pair]["asymm"]
                symmetric = pairs_data[pair]["symm"]
                molecule1 = pairs_data[pair]["mol1_id"]
                molecule2 = pairs_data[pair]["mol2_id"]
                pair_properties[id] = {"min_energy" : energy, "energy_list" : [energy],
                        "min_density" : density, "max_density" : density, 
                        "median_density" : density, "density_list" : [density],
                        "frequency" : 1, "xyz_coordinates" : ''.join(xyz_coords), "mol2_coordinates" : mol2_coords, 
                        "asymmetric" : asymmetric, "symmetric" : symmetric, "molecule1" : molecule1, "molecule2" : molecule2}

            if len(unique_pairs) == 0:
                LOG.error("nNo pairs for %s", str(crys_ind))
            else:
                add_to_mps_database(pair_properties, 'pairs_' + str(Q))

def main(sys_args=None):
        parser = argparse.ArgumentParser(
                prog='cspy-mps extract',
                description='Extract molecular pairs from crystals in db')
        parser.add_argument('database', nargs=1, type=str,
                help='Specify database file to read')
        parser.add_argument('-G', '--G', nargs=1, type=int,
                help='Specify number of free species in asymmetric unit')
        parser.add_argument('-spg', '--spacegroup', nargs=1, type=int,
                help='Override spacegroup label from db')
        parser.add_argument('-np', '--numproc', nargs=1, type=int,
                help='Specify number of paralllel process')
        parser.add_argument('-nc', '--numcrys', nargs=1, type=int,
                help='Specify number of crystals to sample')
        parser.add_argument('-ui', '--unknownid', action='store_true',
                help='Declare module ids as unknown')
        parser.add_argument('-r', '--random', action='store_true',
                help='Sample crystals randomly (must also specify -nc)')
        parser.add_argument('-as', '--asymm', action='store_true',
                help='Store only asymmetric pairs')
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

        from cspy.crystal.space_group import _sgdata_dict
        n_symops = dict()
        for sg in _sgdata_dict.keys():
            n_symops[sg] = len(_sgdata_dict[sg][0][8])

        db = CspDataStore(args.database[0])
        data = db.final_minimizations(with_trial_data=True).fetchall()
        if args.numcrys:
            if args.random:
                random.shuffle(data)
            data = data[:args.numcrys[0]]

        unique_molecules = dict()
        molecules_elements = dict()
        if not args.unknownid:
            for datum in data:
                mol_ids = datum[4].split('.')
                for mol_id in mol_ids:
                    # ignore number of times molecule appears in cell
                    if mol_id[-2] == 'x' and mol_id[-1].isdigit():
                        mol_id = mol_id[0:-2]
                    if not mol_id in unique_molecules.keys():
                        unique_molecules[mol_id] = dict()

            for molecule in unique_molecules:
                cspyMol = Molecule.from_xyz_file(molecule + '.xyz')
                unique_molecules[molecule]["xyz_coordinates"] = cspyMol.to_xyz_string()
                molecules_elements[molecule] = cspyMol.elements
                if len(molecules_elements[molecule]) == 1:
                    cspyMol.equivalent_atoms = [{0:0}]
                else:
                    cspyMol.get_symmetry_equivalent_atoms()
                unique_molecules[molecule]["equivalent_atoms"] = cspyMol.equivalent_atoms
        else:
            unique_molecules["unknownid"] = {"xyz_coordinates": None, "equivalent_atoms" : []}

        workers = args.numproc[0]

        for Q in range(workers):
            try:
                create_mps_database('pairs_' + str(Q))
                set_molecules(unique_molecules,'pairs_' + str(Q))
            except:
                LOG.error('pairs_%s_mps.db already exists.', str(Q))

        LOG.info("Starting extraction")
        with multiprocessing.Pool(workers) as pool:
                pool.starmap(get_pairs, [[data, molecules_elements, Q, workers, n_symops, args] for Q in range(workers)])

        T = {}
        for Q in range(workers):
            db = CspDataStoreMPS('pairs_' + str(Q)+"_mps.db")
            for row in db.select("pairs",
                                    ['id','min_energy','min_density',
                                    'xyz_coordinates','mol2_coordinates',
                                    'asymmetric','symmetric', 'molecule1', 'molecule2']):
                T[row[0]]={"min_energy" : row[1], "energy_list" : [row[1]],
                        "min_density" : row[2], "max_density" : row[2], 
                        "median_density" : row[2], "density_list" : [row[2]],
                        "frequency" : 1, "xyz_coordinates" : row[3], "mol2_coordinates" : row[4], 
                        "asymmetric" : row[5], "symmetric" : row[6], 
                        "molecule1" : row[7], "molecule2" : row[8]}
            db.disconnect()
        create_mps_database("jointDB")
        set_molecules(unique_molecules,"jointDB")
        add_to_mps_database(T,"jointDB")

        for Q in range(workers):
            try:
                os.remove('pairs_' + str(Q) + '_mps.db')
            except:
                pass