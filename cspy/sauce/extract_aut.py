from cspy.db import CspDataStore, CspDataStoreAUs
from cspy.formats.dma import parse_dma
from cspy.chem.energy import BuckinghamPotential
from cspy.crystal import Crystal
from cspy.chem.molecule import Molecule
from cspy.util.misc import mp_check_reservation
from cspy.chem.element import chemical_formula
from cspy.crystal.util import get_contacts, find_formula_unit
from typing import Optional, Tuple
import multiprocessing
import argparse
import os
import random
import math
import numpy as np
import copy
import logging

LOG = logging.getLogger(__name__)

def energy_filter(data : list, num_crys : int, energy_tol : int, buckingham_tol : int = -99999999) -> list:
    """
    Filter out crystals that are above energy tolerance and also
    apply energy-based Buckingham catastrophe check.

    Parameters
    ----------
    data : list
        List of crystals

    num_crys : int
        Number of crystals to extract asymmetric units from

    energy_tol : int
        Maximum allowed lattice energy

    buckingham_tol : int
        Any crystal with an energy below this value is assumed to be a buckingham catastrophe

    Returns
    -------
    pruned_data : list
        List of crystals that satisfy energy criteria

    """
    min_energy = 0

    for crystal in data:
        id, sg, density, energy, mol_id, file_content, _min_step, trial_number, _min_time = crystal
        if energy > buckingham_tol:
            if energy < min_energy:
                min_energy = energy
                LOG.info("Min energy: %d", min_energy)

    pruned_data = []
    for crystal in data:
        id, sg, density, energy, mol_id, file_content, _min_step, trial_number, _min_time = crystal
        if energy > buckingham_tol:
            if (energy - min_energy) < energy_tol:
                pruned_data.append(crystal)

    return pruned_data[0:num_crys]


def create_aus_database(filename : str = "aut"):
    """Create a blank database"""
    filename = filename + "_aus.db"

    ds = CspDataStoreAUs.create_and_connect(filename)
    ds.disconnect()


def update_aus_database(update_data : dict, filename="aut"):
    """
    Update the database of asymmetric units with either molecules
    or asymmetric units

    Parameters
    ----------
    update_data : dict
        Dictionary where the keys are either names of unique
        molecules or unique names of asymmetric units.
        The values are their properties.
        See db.schemas.cspy2_aus.sql for more details.

    filename: str
        Seed of database name. Final name will include _aus.db

    """
    from cspy.db.datastore_writer_aus import DatastoreWriterAUs

    update_data_list = list()
    for id in update_data.keys():
        update_datum = copy.deepcopy(update_data[id])
        update_datum["id"] = id
    update_data_list.append(update_datum)

    filename = filename + "_aus.db"

    db_writer = DatastoreWriterAUs(
        update_data,
        filename,
    )

    db_writer.run_once()


def get_au_xyz(mols : list[Molecule]) -> str:
    """
    Iterate over list of molecule objects and
    summarise them into a single xyz string.

    Parameters
    ----------
    mols : list of Molecules
        List containing mol-CSPy Molecule objects
        which consitute an asymmetric unit.


    """
    elements = sum([mol.elements for mol in mols], [])
    lines = [
            f"{len(elements)}",
            chemical_formula(elements, subscript=False),
                ]
    
    for mol in mols:
        for el, (x, y, z) in zip(mol.elements, mol.positions):
                lines.append(f"{el} {x: 20.12f} {y: 20.12f} {z: 20.12f}")

    xyz_coords = '\n'.join(lines)

    return xyz_coords


def calculate_au_energy(candidate_au : str, Zp : int, pot : str, atom_types : list, charges : list, atom_ids : list) -> Tuple:
    """Calculate the energy of an asymmetric unit

    Parameters
    ----------
    candidate_au : str
        Asymmetric unit in xyz format

    Zp : int
        Z'/Zp/Z prime is the number of formula units in the cluster.

    pot : str
        Name of Buckingham potential in mol-CSPy collection

    atom_types : list
        List of list of NEIGHCRYS atom typings. One value per atom.

    charges : list
        List of list of point charges. One value per atom.

    atom_ids : list of list of ints
        A list which contains one list per molecule in the 
        asymmetric unit. Each sublist contains a list of atom
        ids so that we can know which atom belongs to which
        molecule.

    Returns
    -------
    candidate_au : str
        Asymmetric unit in xyz format

    energy : str
        Energy of asymmeric unit, as defined by a Buckingham 
        potential and point-charge electrostatics.

    """
    cluster_CSPyMol = Molecule.from_xyz_string(candidate_au)
    cluster_CSPyMol.def_components(atom_ids)
    relevant_atom_types = atom_types
    relevant_charges = charges

    au_calcualtor = BuckinghamPotential(cluster_CSPyMol, pot, relevant_atom_types, charges=relevant_charges, Zp=Zp)
    au_calcualtor.calc_energy()
    energy = au_calcualtor.energy['kJ/mol']

    return candidate_au, energy


def extract_min_energy_au(crystal : Crystal, 
                          G : int, 
                          sg : int, 
                          sorted_molecules : list, 
                          molecules_elements : dict, 
                          Zp : int, 
                          pot : str, 
                          atom_types : list, 
                          charges : list, 
                          atom_ids : list) -> dict:

    """Extract various representations of the asymmetric unit from a crystal.
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

    from cspy.crystal.space_group import n_symops_from_SG
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
    n_asym_atoms = np.sum(np.array([len(mol.positions) for mol in crystal._symmetry_unique_molecules]))
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

        u = 3
        v = 3
        w = 3
        num_unit_cells = u * v * w
        center_cell = math.floor(num_unit_cells/ 2)
        crystal_sc = crystal.as_P1_supercell((u, v, w))
        # sc_mols ordering is grouped by U*V*W copies of each mol in each asym unit
        # all mols from an asym unit are included before moving onto the next asym unit

        sc_mols = crystal_sc.unit_cell_molecules()
        if not len(sc_mols) == (num_uc_mols * num_unit_cells):
            LOG.error("Failed Buckingham castrophe check 2")
            return None

        sc_unique_mol_ids = [(mol_id * num_unit_cells) + center_cell for mol_id in range(num_unique_mols)]
        sc_pos = np.concatenate([mol.positions for mol in sc_mols], axis=0)
        vdws = np.array([i.vdw for mol in sc_mols for i in mol.elements])

        candidate_aus = []
        for nucleating_mol_id in range(G):
            contact_asym_mol_ids = [sc_unique_mol_ids[nucleating_mol_id]]

            # We want the mols in the asym unit to be in contact, so will have to check pbc
            start1 = 0
            end1 = 0
            for ind1, asym_mol_id1 in enumerate(sc_unique_mol_ids):
                n_atoms1 = len(sc_mols[asym_mol_id1].positions)
                start1 = end1
                end1 = start1 + (n_atoms1 * num_unit_cells)

                if not ind1 == nucleating_mol_id:
                    mask_pos = np.zeros(sc_pos.shape, dtype=bool)
                    mask_vdw = np.zeros(vdws.shape, dtype=bool)
                    for asym_ind in range(n_symops):
                        shift = n_asym_atoms * asym_ind * num_unit_cells
                        mask_pos[start1 + shift: end1 + shift] = True
                        mask_vdw[start1 + shift: end1 + shift] = True

                    all_neighs = np.reshape(sc_pos[mask_pos], (-1, 3))
                    neigh_vdws = vdws[mask_vdw]

                    smallest_distance = 1000000
                    best_mol = None

                    for ind0, asym_mol_id0 in enumerate(contact_asym_mol_ids):
                        reference_mol = sc_mols[contact_asym_mol_ids[ind0]]

                        mol_pos = reference_mol.positions
                        mol_vdws = np.array([i.vdw for i in reference_mol.elements])

                        real_contacts, distances, box_idxs, at_col_vectors = get_contacts(mol_pos, mol_vdws, all_neighs, neigh_vdws, tolerance=1.0)
                        if len(real_contacts) == 0:
                            continue

                        # concatenate to get list of atomic neighbours
                        real_contacts = np.concatenate(real_contacts, axis=0)
                        distances = np.concatenate(distances, axis=0)
                        if len(real_contacts) == 0:
                            continue
                        # find ids where mask is True (i.e. which atoms are in box)
                        mask_indices = np.where(box_idxs)[0]
                        # get ids of elements in all_neighs where box_idxs are true and appear in real_contacts
                        real_contacts = [mask_indices[n] for n in real_contacts]
                        # find ids where mask is True (i.e. which atoms are not in the reference molecule)
                        mask_indices = np.where(mask_pos[:,0])[0]
                        # get ids of elements in sc_pos where mask_pos are true and appear in unique_contacts
                        real_contacts = [mask_indices[n] for n in real_contacts]

                        min_dist = np.min(distances)
                        if min_dist < smallest_distance:
                            min_dist_id = np.argmin(distances)
                            min_dist_atom = real_contacts[min_dist_id]
                            # which asymmetric unit the atom belongs to
                            asym_num = math.floor(min_dist_atom / (n_asym_atoms * num_unit_cells))

                            # id of the first atom in this asymmetric unit
                            asym_start = asym_num * (num_unit_cells * n_asym_atoms)

                            # how far into that asymmetric unit that atom is
                            pos_in_asym = min_dist_atom - asym_start

                            # subtract atoms that exist earlier in the asymmetric unit and their SC copies
                            # offset = pos_in_asym
                            for ind2, mol in enumerate(crystal._symmetry_unique_molecules):
                                if ind2 == ind1:
                                    break
                                else:
                                    pos_in_asym -= (len(mol.elements) * num_unit_cells)

                            # get the id of the uc which contains that mol        
                            mol_asym_uc_id = math.floor((pos_in_asym)/n_atoms1)

                            # add back on the number of mols earlier in that asymmetric unit and all preceeding asymmetric units
                            min_dist_mol_id = mol_asym_uc_id + (asym_num * G * num_unit_cells) + (num_unit_cells * ind1)

                            best_mol = min_dist_mol_id

                    if best_mol is None:
                        break
                    contact_asym_mol_ids.append(best_mol)

            if None in contact_asym_mol_ids or not len(contact_asym_mol_ids) == G:
                continue

            # Nucleating mol has been moved to start of list. Lets put it back in the right place
            contact_asym_mol_ids.insert(nucleating_mol_id,contact_asym_mol_ids.pop(0))

            contact_asym_mols = []
            for id in contact_asym_mol_ids:
                contact_asym_mols.append(copy.deepcopy(sc_mols[id]))
            centroid = np.mean(np.concatenate([mol.positions for mol in contact_asym_mols], axis=0), axis=0)
            for mol in contact_asym_mols:
                mol.translate(-1 * centroid)

            # sometimes cspy finds molecules from the asymmetric unit in the wrong order
            sorted_contact_asym_mols = []
            reordered_mols = []
            for molecule0 in sorted_molecules:
                for ind, molecule1 in enumerate(molecules):
                    if molecule0 == molecule1 and not ind in reordered_mols:
                        sorted_contact_asym_mols.append(contact_asym_mols[ind])
                        reordered_mols.append(ind)
                        continue

            xyz_coords = get_au_xyz(sorted_contact_asym_mols)
            candidate_aus.append(xyz_coords)

        if len(candidate_aus) == 0:
            return None
        
        best_energy = 99999999999999999999
        for au_ind, candidate_au in enumerate(candidate_aus):
            candidate_au_xyz_coords, energy = calculate_au_energy(candidate_au, Zp, pot, atom_types, charges, atom_ids)
            if energy < best_energy:
                best_energy = energy
                au_xyz_coords = candidate_au_xyz_coords

        # these asymmtric units probably aren't very good
        if best_energy > 0:
            return None

        aus_data = {"energy" : best_energy,
                    "xyz_coordinates" : au_xyz_coords,
                    "molecule_ids" : sorted_molecules}


    else:
        LOG.error("Failed Buckingham castrophe check 1")
        return None

    return aus_data


def parallelise_au_extraction(data : list, 
                              sorted_molecules : list, 
                              molecules_elements : dict, 
                              Zp : int, 
                              pot : str, 
                              atom_types : list, 
                              charges : list, 
                              atom_ids : list, 
                              Q : int, 
                              workers : int, 
                              args) -> None:
    
    """Iterate over all crystals, assign each crystal to a unique worker.
    Then extract the lowest energy representation of the asymmetric unit.

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
            aus_data = extract_min_energy_au(crystal, args.G[0], args.spacegroup[0], sorted_molecules, molecules_elements, Zp, pot, atom_types, charges, atom_ids)

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
            prog='cspy-mps extract',
            description='Extract asymmetric units from crystals in db')
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
    datum = data[0]
    #for datum in data:
    mol_ids = datum[4].split('.')
    for mol_id in mol_ids:
        # ignore number of times molecule appears in cell
        if mol_id[-2] == 'x' and mol_id[-1].isdigit():
            numeracy = int(mol_id[-1])
            mol_id = mol_id[0:-2]
        else:
            numeracy = 1
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
            pool.starmap(parallelise_au_extraction, [[data, sorted_molecules, molecules_elements, Zp, pot, atom_types, charges, atom_ids, Q, workers, args] for Q in range(workers)])

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