from cspy.db import CspDataStore, CspDataStoreAUs
from cspy.formats.dma import parse_dma
from cspy.crystal import Crystal
from cspy.chem.molecule import Molecule
from cspy.util.misc import mp_check_reservation
from cspy.crystal.util import find_formula_unit
from cspy.sauce.extract_aut import energy_filter, create_aus_database, update_aus_database, get_au_xyz, calculate_au_energy
import multiprocessing
import argparse
import os
import random
import numpy as np
from cspy.crystal.space_group import n_symops_from_SG
import logging

LOG = logging.getLogger(__name__)


def extract_uc_as_au(crystal : Crystal, 
                     G : int, 
                     sg : int, 
                     sorted_molecules : list, 
                     molecules_elements : list, 
                     Zp : int, 
                     pot : str, 
                     atom_types : list, 
                     charges : list, 
                     atom_ids : list) -> None: 
    """Extract unit cell molecules from a crystal.
    Calculate the energy of each representation and store that representation only.

    Parameters
    ----------
    crystal : Crystal
        mol-CSPy Crystal object to extract asymmetric unit from

    G : int
        Number of independent molecules/ions in the asymmetric unit

    sg : int
        Crystal space group

    sorted_molecules : list
        List of list of NEIGHCRYS atom typings. One value per atom.

    molecules_elements : dict
        Dictionary where the keys are the names of molecules
        and the values are a list of that molecules elements.

    Zp : int
        Z'/Zp/Z prime is the number of formula units in the cluster.

    pot : str
        Name of Buckingham potential in mol-CSPy collection

    atom_types : list
        List of list of NEIGHCRYS atom typings. One value per atom.

    charges : list
        List of list of point charges. One value per atom.

    atom_ids : list of list of iints
        A list which contains one list per molecule in the 
        asymmetric unit. Each sublist contains a list of atom
        ids so that we can know which atom belongs to which
        molecule.

    Returns
    -------
    aus_data : dict
        Dictionary where keys are unique names of asymmetric units
        and values are dictionaries of properties.
    """

    from cspy.cspympi.cspy_tasks import MinimizedStructure

    n_symops = n_symops_from_SG(sg, context='default')

    aus_data = dict()
    molecules = []
    # check what format the crystal is in
    # cspy Crystal obj
    if isinstance(crystal, Crystal):
        pass
    else:
        # Tuple from cspy_tasks
        if isinstance(crystal, MinimizedStructure):
            id, spacegroup, trial_number, minimization_step, energy, density, file_content, xrd, time, molecule_id = crystal
            
        # from a SQL database
        else:
            id, sg, density, energy, mol_id, file_content, _min_step, trial_number, _min_time = crystal

        crystal = Crystal.from_shelx_string(file_content)

    crystal.symmetry_unique_molecules()
    num_unique_mols = len(crystal._symmetry_unique_molecules)
    expected_uc_mols = num_unique_mols * n_symops
    num_uc_mols = len(crystal.unit_cell_molecules())
    for mol in crystal._symmetry_unique_molecules:
        mol_elements = mol.elements
        for unique_molecule in molecules_elements.keys():
            if mol_elements == molecules_elements[unique_molecule]:
                # will need to add extra check with overlap for molecules with the same elements
                molecules.append(str(unique_molecule))
                break

    # check for Buckingham catastrophes
    if num_unique_mols == G and expected_uc_mols == num_uc_mols:
        candidate_aus = []
        # move molecules into unit cell
        unit_cell_molecules = []
        direct = crystal.unit_cell.direct
        inverse = crystal.unit_cell.inverse
        for mol in crystal._unit_cell_molecules:
            cent = mol.centroid @ inverse
            move_to_uc = (np.mod(cent, 1) - cent) @ direct
            mol.translate(move_to_uc)
            unit_cell_molecules.append(mol)
        xyz_coords = get_au_xyz(unit_cell_molecules)

        candidate_aus.append(xyz_coords)

        if len(candidate_aus) == 0:
            return None
        
        best_energy = 99999999999999999999
        for au_ind, candidate_au in enumerate(candidate_aus):
            candidate_au_xyz_coords, energy = calculate_au_energy(candidate_au, molecules, Zp, pot, atom_types, charges, atom_ids)
            if energy < best_energy:
                best_energy = energy
                au_xyz_coords = candidate_au_xyz_coords

        aus_data = {"energy" : best_energy,
                    "xyz_coordinates" : au_xyz_coords,
                    "molecule_ids" : sorted_molecules}

    else:
        LOG.error("Failed Buckingham castrophe check 1")
        return None

    return aus_data


def parallelise_uc2aut_extraction(data : list, 
                                  sorted_molecules : list, 
                                  molecules_elements : list, 
                                  Zp : int, 
                                  pot : str, 
                                  atom_types : list, 
                                  charges : list, 
                                  atom_ids : list, 
                                  Q : int, 
                                  workers : int, 
                                  args : dict) -> None:
    """Iterate over all crystals, assign each crystal to a unique worker.
    Then extract the unit cell molecules.

    Parameters
    ----------
    data : list
        List of mol-CSPy Crystal objects to extract asymmetric units from

    sorted_molecules : list
        List of list of NEIGHCRYS atom typings. One value per atom.

    molecules_elements : dict
        Dictionary where the keys are the names of molecules
        and the values are a list of that molecules elements.

    Zp : int
        Z'/Zp/Z prime is the number of formula units in the cluster.

    pot : str
        Name of Buckingham potential in mol-CSPy collection

    atom_types : list
        List of list of NEIGHCRYS atom typings. One value per atom.

    charges : list
        List of list of point charges. One value per atom.

    atom_ids : list of list of iints
        A list which contains one list per molecule in the 
        asymmetric unit. Each sublist contains a list of atom
        ids so that we can know which atom belongs to which
        molecule.

    Q : int
        Unique worker id for parallel process. Used for mp_check_reservation function

    workers : int
        Number of workers operating in parallel. Used for mp_check_reservation function

    args : dict
        Further arguments from argparse

    """
    for crys_ind, crystal in enumerate(data):
        # this if statement checks the crystal was intended for this worker
        if mp_check_reservation(crys_ind, Q, workers):
            aus_data = extract_uc_as_au(crystal, args.G[0], args.spacegroup[0], sorted_molecules, molecules_elements, Zp, pot, atom_types, charges, atom_ids)

            au_properties = {}
            id = str(crys_ind)
            au_properties[id] = aus_data
                
            if aus_data:
                if len(aus_data) == 0:
                    LOG.error("No asymmetric unit for %s", str(crys_ind))
                else:
                    update_aus_database(au_properties, 'AU_' + str(Q))
            else:
                LOG.error("No asymmetric unit for %s", str(crys_ind))


def main(sys_args=None):
    parser = argparse.ArgumentParser(
            prog='cspy-sauce extract_uc2au',
            description='Extract unit cells from crystals in db')
    parser.add_argument('database', nargs=1, type=str,
            help='Specify database file to read')
    parser.add_argument('-G', '--G', nargs=1, type=int,
            help='Specify number of free species in asymmetric unit')
    parser.add_argument('-spg', '--spacegroup', nargs=1, type=int,
            help='Override spacegroup label from db')
    parser.add_argument('-p', '--pot', nargs=1, type=str,
        help='Name of potential file without extension')
    parser.add_argument('-d', '--dma', nargs=1, type=str,
        help='Rank0 dma file containing atom types and charges')
    parser.add_argument('-np', '--numproc', nargs=1, type=int,
            help='Specify number of paralllel process')
    parser.add_argument('-nc', '--numcrys', nargs=1, type=int,
            help='Specify number of crystals to sample')
    parser.add_argument('-et', '--energytol', nargs=1, type=float,
            help='Only use crystals within energy tolerance of global minimum')
    parser.add_argument('-bt', '--buckinghamtol', nargs=1, type=float,
            help='Treat crystals with less energy has a buckingham catastrophe')
    parser.add_argument('-r', '--random', action='store_true',
            help='Sample crystals randomly (must also specify -nc)')

    args = parser.parse_args(sys_args)

    logging.basicConfig(
        level=args.log_level,
        format='%(asctime)s - %(levelname)s - %(module)s %(lineno)d - '
               '%(message)s')

    db = CspDataStore(args.database[0])
    LOG.info("Reading database")
    data = db.final_minimizations(with_trial_data=True).fetchall()
    LOG.info("Done reading database")
    if args.numcrys:
        if args.random:
            random.shuffle(data)
        if args.energytol:
            if not args.buckinghamtol:
                buckingham_tol = [-9999999]
            else:
                buckingham_tol = args.buckinghamtol
            data = energy_filter(data, args.numcrys[0], args.energytol[0], buckingham_tol[0])
        else:
            data = data[:args.numcrys[0]]

    unique_molecules = dict()
    molecules_elements = dict()
    sorted_molecules = []
    n_symops = n_symops_from_SG(int(args.spacegroup[0]), context='default')
    datum = data[0]
    #for datum in data:
    mol_ids = datum[4].split('.')
    for mol_id in mol_ids:
        # ignore number of times molecule appears in cell
        if mol_id[-2] == 'x' and mol_id[-1].isdigit():
            numeracy = int(mol_id[-1]) * n_symops
            mol_id = mol_id[0:-2]
        else:
            numeracy = n_symops
        for _ in range(numeracy):
            sorted_molecules.append(mol_id)
        if not mol_id in unique_molecules.keys():
            unique_molecules[mol_id] = dict()

    # get Z' by finding greatest common divisor
    formula_unit, unique_molecules, Zp = find_formula_unit(sorted_molecules)

    # get a list of list where each list contains the unique numerical id of each atom
    atom_ids = []
    for molecule in sorted_molecules:
        mol_num_atoms = len(Molecule.from_xyz_file(molecule + '.xyz').elements)
        if len(atom_ids) == 0:
            prev_mol_num_atoms = 0
        else:
            prev_mol_num_atoms = atom_ids[-1][-1] + 1
        mol_atom_ids = [i for i in range(prev_mol_num_atoms, prev_mol_num_atoms + mol_num_atoms)]
        atom_ids.append(mol_atom_ids)

    for molecule in unique_molecules:
        cspyMol = Molecule.from_xyz_file(molecule + '.xyz')
        unique_molecules[molecule]["xyz_coordinates"] = cspyMol.to_xyz_string()
        molecules_elements[molecule] = cspyMol.elements
        if len(molecules_elements[molecule]) == 1:
            cspyMol.equivalent_atoms = [{0:0}]
        else:
            cspyMol.get_symmetry_equivalent_atoms()
        unique_molecules[molecule]["equivalent_atoms"] = cspyMol.equivalent_atoms

    dma_file = args.dma[0]
    pot = args.pot[0]
    dma_data = parse_dma(dma_file)
    atom_types = []
    charges = []
    for molecule_dma in dma_data:
        atom_types.append(molecule_dma['atom_types'])
        charges.append(molecule_dma['charges'])

    workers = args.numproc[0]

    for Q in range(workers):
        try:
            create_aus_database('AU_' + str(Q))
            update_aus_database(unique_molecules,'AU_' + str(Q))
        except:
            LOG.error('AU_%s_aus.db already exists.', str(Q))

    LOG.info("Starting extraction")
    with multiprocessing.Pool(workers) as pool:
            pool.starmap(parallelise_uc2aut_extraction, [[data, sorted_molecules, molecules_elements, Zp, pot, atom_types, charges, atom_ids, Q, workers, args] for Q in range(workers)])

    T = {}
    for Q in range(workers):
        db = CspDataStoreAUs('AU_' + str(Q)+"_aus.db")
        for row in db.select("asym_units",
                                ['id','energy', 'xyz_coordinates','molecule_ids']):
            T[row[0]]={"energy" : row[1], "xyz_coordinates" : row[2], "molecule_ids" : row[3]}
            

        db.disconnect()
    create_aus_database("jointDB")
    update_aus_database(unique_molecules,"jointDB")
    update_aus_database(T,"jointDB")

    for Q in range(workers):
        try:
            os.remove('AsymU_' + str(Q) + '_mps.db')
        except:
            pass