import logging
from cspy.util.logging_config import FORMATS, DATEFMT
import argparse
import numpy as np
import os.path
import sys
import re
from pathlib import Path
from cspy.db import CspDataStoreMolecules
from cspy.formats.dma import parse_dma
from cspy.apps.dma import generate_combined_name

LOG = logging.getLogger(__name__)

def reorder_xyz(old_file_content, element_ordering):
    """ Rearange the lines of the xyz so that elements are
        grouped according to the order of elements in element_ordering

        Parameters
        ----------
        old_file_content : string
            file_content of xyz_file
        element_ordering : list of strings
            atoms in xyz files / res files should be ordered 
            according to the ordering of the elements in this list 

        Returns
        -------
        file_content : string
            file_content of xyz_file
    """

    file_content_lines = old_file_content.split('\n')
    file_content = [file_content_lines[0], file_content_lines[1]]

    for element in element_ordering:
        for line in file_content_lines[2:]:
            line_split = line.split()
            # there might be a blank line at the end. We'll skip it
            if len(line_split) > 0:
                element_label = line_split[0]
                if element_label == element:
                    file_content.append(line)

    return '\n'.join(file_content)


def merge_charges_files(all_charges_content, relabel_dicts, element_ordering, combined_name, skip_writing=False):
    """ Take 2 or more DMACRYS charges file contents (string) and combine them
        to return a new DMACRYS charges file content (string).
        Make sure that the element ordering is consistent 
        and label indexing is updated

        Parameters
        ----------
        all_charges_content : list of strings
            list of the file content of each _rank0.dma file as strings
        relabel_dicts : dict
            dictionary where both keys and values are NEIGHCRYS atom labels
            It is intended that atoms labelled with the key are relabeled to the value
        element_ordering : list of strings
            atoms in xyz files / res files should be ordered 
            according to the ordering of the elements in this list 
        combined_name : string
            string that represents the combination of two or more molecules
            see cspy.apps.dma.generate_combined_name
        skip_writing : bool
            skip writing the file, and just return the string instead
    """
    relabelled_charges = []

    for ind, charges_content in enumerate(all_charges_content):
        relabel_dict = relabel_dicts[ind]
        # list must be reversed so that each label is only replaced once
        relabel_dict_keys = reversed(list(relabel_dict.keys()))
        for label in relabel_dict_keys:
            label = str(label)
            charges_content = charges_content.replace(label, relabel_dict[label])

        relabelled_charges.append(charges_content)

        atom_dict = dict()
        for element in element_ordering:
            atom_dict[element] = []
        lines = charges_content.split('\n')
        for line in lines:
            line_split = line.split()
            if len(line_split) == 6:
                # element symbol has 1 character
                if line_split[0][1] == '_':
                    element_label = line_split[0].split('_')[0]
                # element symbol has 2 characters
                else:
                    element_label = line_split[0][:2]
                atom_dict[element_label].append(line)
            elif len(line_split) == 1 and not 'ENDMOL' in line:
                # charges are stored undernearth coordinates. 
                # element_label is remembered from the previous loop iteration
                atom_dict[element_label][-1] = atom_dict[element_label][-1] + '\n' + line

        new_relabelled_charges = []
        for element in element_ordering:
            for atom in atom_dict[element]:
                new_relabelled_charges.append(atom)

        new_relabelled_charges.append('\n#ENDMOL\n')
        relabelled_charges[-1] = '\n'.join(new_relabelled_charges)

    combined_relabelled_charges = ''.join(relabelled_charges)

    if skip_writing:
        return combined_relabelled_charges
    
    else:
        with open(combined_name + '_rank0.dma', 'w') as f:
            f.write(combined_relabelled_charges)


def merge_mults_files(all_mults_content, relabel_dicts, element_ordering, combined_name, skip_writing=False):
    """ Take 2 or more DMACRYS multipole file contents (string) and combine them 
        to return a new DMACRYS multipole file content (string).
        Make sure that the element ordering is consistent 
        and label indexing is updated

        Parameters
        ----------
        all_mults_content : list of strings
            list of the file content of each .dma file as strings
        relabel_dicts : dict
            dictionary where both keys and values are NEIGHCRYS atom labels
            It is intended that atoms labelled with the key are relabeled to the value
        element_ordering : list of strings
            atoms in xyz files / res files should be ordered 
            according to the ordering of the elements in this list 
        combined_name : string
            string that represents the combination of two or more molecules
            see cspy.apps.dma.generate_combined_name
        skip_writing : bool
            skip writing the file, and just return the string instead
    """

    relabelled_mults = []

    for ind, mults_content in enumerate(all_mults_content):
        relabel_dict = relabel_dicts[ind]
        # list must be reversed so that each label is only replaced once
        relabel_dict_keys = reversed(list(relabel_dict.keys()))
        for label in relabel_dict_keys:
            label = str(label)
            mults_content = mults_content.replace(label, relabel_dict[label])

        relabelled_mults.append(mults_content)

        atom_dict = dict()
        for element in element_ordering:
            atom_dict[element] = []
        lines = mults_content.split('\n')
        for line in lines:
            line_split = line.split()
            if len(line_split) == 6:
                # element symbol has 1 character
                if line_split[0][1] == '_':
                    element_label = line_split[0].split('_')[0]
                # element symbol has 2 characters
                else:
                    element_label = line_split[0][:2]
                atom_dict[element_label].append(line)
            elif len(line_split) > 0 and not 'ENDMOL' in line:
                # mults are stored undernearth coordinates. 
                # element_label is remembered from the previous loop iteration
                atom_dict[element_label][-1] = atom_dict[element_label][-1] + '\n' + line

        new_relabelled_mults = []
        for element in element_ordering:
            for atom in atom_dict[element]:
                new_relabelled_mults.append(atom)

        new_relabelled_mults.append('\n#ENDMOL\n')
        relabelled_mults[-1] = '\n'.join(new_relabelled_mults)

    combined_relabelled_mults = ''.join(relabelled_mults)

    if skip_writing:
        return combined_relabelled_mults
    
    else:
        with open(combined_name + '.dma', 'w') as f:
            f.write(combined_relabelled_mults)


def merge_axes_files(all_axes_content, relabel_dicts, combined_name, skip_writing=False):
    """ Take 2 or more DMACRYS axes file contents and combine them
        to return a new DMACRYS axes file content.
        Make sure that label indexing is updated

        Parameters
        ----------
        all_axes_content : list of strings
            list of the file content of each .mols file as strings
        relabel_dicts : dict
            dictionary where both keys and values are NEIGHCRYS atom labels
            It is intended that atoms labelled with the key are relabeled to the value
        combined_name : string
            string that represents the combination of two or more molecules
            see cspy.apps.dma.generate_combined_name
        skip_writing : bool
            skip writing the file, and just return the string instead
    """

    relabelled_axes = []
    num_lone_nuclei = 0
    
    for ind, axes_content in enumerate(all_axes_content):
        if not "MOLX 0" in axes_content:
            relabel_dict = relabel_dicts[ind]
            # list must be reversed so that each label is only replaced once
            relabel_dict_keys = reversed(list(relabel_dict.keys()))
            for label in relabel_dict_keys:
                label = str(label)
                axes_content = axes_content.replace(label, relabel_dict[label])

            relabelled_axes.append(axes_content)
        else:
            num_lone_nuclei += 1

    num_mols = len(all_axes_content) - num_lone_nuclei

    combined_relabelled_axes = ''.join(relabelled_axes)
    combined_relabelled_axes = combined_relabelled_axes.replace('\nENDSMOLX 1', '')
    combined_relabelled_axes = combined_relabelled_axes.replace('MOLX 1', 'MOLX ' + str(num_mols))

    if skip_writing:
        return combined_relabelled_axes
    
    else:
        with open(combined_name + '.mols', 'w') as f:
            f.write(combined_relabelled_axes)


def merge_neighcrys_labels(all_charges_content):
    """ Need to update the label indexing so that 
        2 or more molecules can't contain atoms with
        the same neighcrys label

        Parameters
        ----------
        all_charges_content : list of strings
            list of the file content of each _rank0.dma file as strings

        Returns
        -------
        relabel_dicts : dict
            dictionary where both keys and values are NEIGHCRYS atom labels
            It is intended that atoms labelled with the key are relabeled to the value
        element_ordering : list of strings
            atoms in xyz files / res files should be ordered 
            according to the ordering of the elements in this list 
    """

    label_multiplicity = dict()
    element_ordering = []
    # list of dictionaries of label changes
    relabel_dicts = []
    for charges in all_charges_content:

        relabel_dict = dict()
        # searches for NEIGHCRYS labels
        matches = re.findall("^[a-zA-z0-9]+_\w+", charges, re.MULTILINE)
        for label in matches:
            suffix = re.findall("_+\d+_+$", label)[0]
            chem_label = label.replace(suffix, '')
            # element symbol has 1 character
            if label[1] == '_':
                element_label = chem_label.split('_')[0]
                count = int(suffix.replace('_', ''))
            # element symbol has 2 characters
            else:
                element_label = chem_label[:2]
                count = int(suffix.replace('_', ''))

            if not element_label in element_ordering:
                label_multiplicity[element_label] = 0
                element_ordering.append(element_label)

            old_count = count

            label_multiplicity[element_label] += 1
            count = label_multiplicity[element_label]

            new_label = chem_label + suffix.replace(str(old_count), str(count))
            # NEIGHCRYS wants the labels to be 10 characters long so we trim the tail
            new_label = new_label[:10]
            relabel_dict[label] = new_label

        relabel_dicts.append(relabel_dict)

    return relabel_dicts, element_ordering


def write_xyz_files(molecule_seeds, molecules_data, element_ordering, combined_name, overwrite=False, skip_writing=False):
    """ Write xyz files after reordering them such that element ordering is consistent between the xyz files

        Parameters
        ----------
        molecule_seeds : list of strings
            names of molecules without file extension
        molecules_data : dict
            dictionary of molecules and the file content for their input files
            key is the name of the molecules.
            The value is a list of strings, where each string is the
            the .xyz file content , the total charge, the _rank0.dma file content, 
            the .dma file content, and the .mols file content.
        element_ordering : list of strings
            atoms in xyz files / res files should be ordered 
            according to the ordering of the elements in this list 
        combined_name : string
            string that represents the combination of two or more molecules
            see cspy.apps.dma.generate_combined_name
        overwrite : bool
            allow xyz files to be overwritten
        skip_writing : bool
            skip writing the file, and just return the string instead
    """

    all_content = []
    for seed in molecule_seeds:
        molecule_data = molecules_data[seed]
        filename = seed + '.xyz'
        file_content = reorder_xyz(molecule_data[0], element_ordering)

        if skip_writing:
            all_content.append(file_content)
        
        else:
            if not overwrite and os.path.isfile(filename):
                dir_name = combined_name + '_xyz'
                LOG.info("Refusing to overwrite existing " + filename + " file. Saving new xyz files into " + dir_name + '/')
                if not os.path.isdir(dir_name):
                    Path(dir_name).mkdir(parents=True, exist_ok=True)
                with open(dir_name + '/' + filename, 'w') as f:
                    f.write(file_content)

            else:
                with open(filename, 'w') as f:
                    f.write(file_content)

    if skip_writing:
        return all_content

def scrape_molecular_data(molecule_seeds, file_extensions):
    """ Look for files .xyz, _rank0.dma, .dma, and .mols files, and extract 
    xyz, total charges, _rank0.dma, .dma, and .mols data

        Parameters
        ----------
        molecule_seeds : string
            name of molecule without file extension
        file_extensions : list of strings
            file_extensions to look for to scrape data from
    """

    molecules_data = dict()
    for seed in molecule_seeds:
        molecule_data = []
        for ext in file_extensions:
            filename = seed + ext

            if ext == '_rank0.dma':
                dma_data = parse_dma(filename)
                charge_data = dma_data[0]["charges"]
                total_charge = int(round(np.sum(charge_data), 0))

                molecule_data.append(total_charge)

            file_content = Path(filename).read_text()
            molecule_data.append(file_content)

        mol_name = seed
        if '/' in mol_name:
            mol_name = mol_name.split('/')[-1]
        molecules_data[mol_name] = molecule_data

    return molecules_data


def create_molecule_database(filename="molecules.db"):
    """ Create an empty database for molecules

        Parameters
        ----------
        filename : string
            name of sql database
    """

    if not '.db' in filename:
        filename = filename + ".db"

    ds = CspDataStoreMolecules.create_and_connect(filename)
    ds.disconnect()

    return filename


def add_to_molecules_database(molecules_data, filename="molecules.db"):
    """ Add .xyz, total charges, _rank0.dma, .dma, and .mols data 
    to molecule database

        Parameters
        ----------
        molecules_data : dict
            dictionary of molecules and the file content for their input files
            key is the name of the molecules.
            The value is a list of strings, where each string is the
            the .xyz file content , the total charge, the _rank0.dma file content, 
            the .dma file content, and the .mols file content.
        filename : string
            name of sql database
    """

    from cspy.db.datastore_writer_molecules import DatastoreWriterMolecules

    db_writer = DatastoreWriterMolecules(
        molecules_data,
        filename,
    )

    db_writer.run()


def main(sys_args=None):
        parser = argparse.ArgumentParser(
                prog='cspy-gplus',
                description='Script for handling the setup up of co-crystal / \
                    salt CSPs from a database of molecules/ions')
        parser.add_argument("xyz_files", nargs="*", type=str,
                help="Xyz files containing molecules")
        parser.add_argument('-d', '--database', nargs=1, type=str,
                help='Specify molecules database file to read')
        parser.add_argument('-l', '--list', nargs='?', const='any',
                help='List molecules in database. \
                    If an int is provided, only list molecules with that charge.')
        parser.add_argument('-a', '--add', action="store_true",
                help='Add molecules to database instead of setting up CSP')
        args = parser.parse_args(sys_args)

        logging.basicConfig(
        level='INFO', format=FORMATS['INFO'], datefmt=DATEFMT
            )
        logging.getLogger(__name__).setLevel('INFO')

        file_extensions = ['.xyz', '_rank0.dma', '.dma', '.mols']

        if args.database:
            database = args.database[0]
            # check if the database exists
            if not os.path.isfile(database):
                if args.add:
                    LOG.info("Database %s does not exist. Creating...", database)
                    database = create_molecule_database(database)
                else:
                    LOG.error("Database %s does not exist. Exiting...", database)
                    sys.exit()
        else:
            database = None
            # check if arguments are compatible
            if args.add:
                LOG.error("No database specified so cannot add to it. Exiting...")
                sys.exit()
            if args.list:
                LOG.error("No database specified so cannot list its contents. Exiting...")
                sys.exit()

        molecule_seeds = []
        combined_name_input = []
        if args.xyz_files:
            for xyz in args.xyz_files:
                if '.xyz' in xyz:
                    molecule_seeds.append(xyz.split('.xyz')[0])
                    combined_name_input.append(xyz)
                else:
                    molecule_seeds.append(xyz)
                    combined_name_input.append(xyz + '.xyz')

            combined_name = generate_combined_name(combined_name_input)

        if args.add:
            # check if all the relevant files exist
            for seed in molecule_seeds:
                for ext in file_extensions:
                    if not os.path.isfile(seed + ext):
                        LOG.error("File %s missing. Exiting...", seed + ext)
                        sys.exit()

            molecules_data = scrape_molecular_data(molecule_seeds, file_extensions)
            add_to_molecules_database(molecules_data, filename=database)

        # write out .xyz, _rank0.dma, .dma, and .axes files for molecule combination
        elif args.xyz_files:
            molecules_data = dict()
            if database:
                db = CspDataStoreMolecules(database)
            cell_charge = 0
            all_charges_content = []
            all_mults_content = []
            all_axes_content = []

            if not database:
                # check if all the relevant files exist
                for seed in molecule_seeds:
                    for ext in file_extensions:
                        if not os.path.isfile(seed + ext):
                            LOG.error("File %s missing. Exiting...", seed + ext)
                            sys.exit()
                molecules_data = scrape_molecular_data(molecule_seeds, file_extensions)

            for seed in molecule_seeds:
                # xyz_coordinates, total_charge, charges, mults, axes
                if database:
                    molecules_data[seed] = list(db.molecule_data(seed).fetchall()[0])
                cell_charge += molecules_data[seed][1]
                all_charges_content.append(molecules_data[seed][2])
                all_mults_content.append(molecules_data[seed][3])
                all_axes_content.append(molecules_data[seed][4])

            if not cell_charge == 0:
                LOG.error("Resultant charge on cell equals %s. \nDMACRYS won't like that. Refusing to continue...", database)
                sys.exit()

            # get the new neighcrys labels
            relabel_dicts, element_ordering = merge_neighcrys_labels(all_charges_content)

            # finally, write input files for CSPy
            write_xyz_files(molecule_seeds, molecules_data, element_ordering, combined_name)
            merge_charges_files(all_charges_content, relabel_dicts, element_ordering, combined_name)
            merge_mults_files(all_mults_content, relabel_dicts, element_ordering, combined_name)
            merge_axes_files(all_axes_content, relabel_dicts, combined_name)

        # return list of molecules in database
        if args.list:
            db = CspDataStoreMolecules(database)
            if args.list == "any":
                data = db.list_molecules()
            else:
                data = db.list_molecules_with_charge(int(args.list))

            for molecule in data:
                print(molecule[0])
