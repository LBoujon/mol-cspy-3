from unittest import TestCase
from os.path import abspath, join, dirname
from tempfile import TemporaryDirectory
import os
import numpy as np
from pathlib import Path
from cspy.apps.gplus import create_molecule_database, add_to_molecules_database, \
    merge_neighcrys_labels, merge_charges_files, merge_mults_files, merge_axes_files
from cspy.db import CspDataStoreMolecules
from cspy.formats.dma import parse_dma
from cspy.apps.dma import generate_combined_name

EXAMPLES_DIR = abspath(join(dirname(__file__), "../../../examples"))
EXPERIMENTAL = join(EXAMPLES_DIR, "experimental_structures")
GENERATED = join(EXAMPLES_DIR, "generated_structures")
ACETAC01 = join(EXAMPLES_DIR, "ACETAC01")

REFERENCE_OUTPUT = "MOLX 2\nX LINE  C_F1_3____ C_F1_4____ 1\nY PLANE C_F1_3____ C_F1_4____ 1 H_F2_5____ 2\nX LINE  C_F1_7____ C_F1_8____ 1\nY PLANE C_F1_7____ C_F1_8____ 1 H_F2_13___ 2\nENDS"

class GplusTests(TestCase):
    file_extensions = ['.xyz', '_rank0.dma', '.dma', '.mols']
    mol_names = [ACETAC01 + '/ACETAC01', ACETAC01 + '/ACETAC01']
    db_name = 'test.db'
    combined_name = generate_combined_name(['ACETAC01.xyz', 'ACETAC01.xyz'])

    def setUp(self):
        self._original_directory = os.getcwd()
        self._temporary_directory = TemporaryDirectory()
        os.chdir(self._temporary_directory.name)

    def tearDown(self):
        os.chdir(self._original_directory)
        self._temporary_directory.cleanup()


    def test_setup_database(self):
        new_db_name = create_molecule_database(self.db_name)
        assert new_db_name == 'test.db'
        self.db_name = new_db_name

        molecules_data = dict()
        for seed in self.mol_names:
            molecule_data = []
            for ext in self.file_extensions:
                filename = seed + ext

                if ext == '_rank0.dma':
                    dma_data = parse_dma(filename)
                    charge_data = dma_data[0]["charges"]
                    total_charge = int(np.sum(charge_data))

                    molecule_data.append(total_charge)

                file_content = Path(filename).read_text()
                molecule_data.append(file_content)

            mol_name = seed
            mol_name = mol_name.split('/')[-1]
            molecules_data[mol_name] = molecule_data

        add_to_molecules_database(molecules_data, filename=self.db_name)


    def test_setup_cocrystal(self):

        self.test_setup_database()
        
        molecules_data = dict()

        assert os.path.isfile('test.db') == True

        db = CspDataStoreMolecules('test.db')

        data = db.list_molecules()
        print(list(data))
        for molecule in data:
            print(molecule[0])

        cell_charge = 0
        all_charges_content = []
        all_mults_content = []
        all_axes_content = []

        for seed in self.mol_names:
            # xyz_coordinates, total_charge, charges, mults, axes
            mol_name = seed
            mol_name = mol_name.split('/')[-1]
            molecules_data[mol_name] = list(db.molecule_data(mol_name).fetchall()[0])
            cell_charge += molecules_data[mol_name][1]
            all_charges_content.append(molecules_data[mol_name][2])
            all_mults_content.append(molecules_data[mol_name][3])
            all_axes_content.append(molecules_data[mol_name][4])

        # xyz files are 1 molecule per file
        for seed in self.mol_names:
            mol_name = seed
            mol_name = mol_name.split('/')[-1]
            molecule_data = molecules_data[mol_name]
            filename = mol_name + '.xyz'
            if not os.path.isfile(filename):
                with open(filename, 'w') as f:
                    f.write(molecule_data[0])

        relabel_dicts, element_ordering = merge_neighcrys_labels(all_charges_content)
        merge_charges_files(all_charges_content, relabel_dicts,  element_ordering, self.combined_name)
        merge_mults_files(all_mults_content, relabel_dicts,  element_ordering, self.combined_name)
        merge_axes_files(all_axes_content, relabel_dicts, self.combined_name)

        with open(self.combined_name + '.mols', 'r') as f:
            axes_output = f.read()

        assert axes_output == REFERENCE_OUTPUT

        # clean up after test
        os.remove(self.combined_name + '_rank0.dma')
        os.remove(self.combined_name + '.dma')
        os.remove(self.combined_name + '.mols')
        os.remove('test.db')
