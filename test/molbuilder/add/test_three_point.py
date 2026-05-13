from unittest import TestCase
from rdkit import Chem
from cspy.molbuilder.addition.addition import add_fragment
from cspy.molbuilder.addition.addition import three_point_attachment_positions


class TestAddFragment(TestCase):

    def setUp(self):
        self.napthalene = Chem.MolFromSmiles('c1ccc2ccccc2c1')
        self.acenaphthylene = Chem.MolFromSmiles('c1cc2cccc3c2c(c1)C=C3')

        self.five_membered_ring_fragment = Chem.MolFromSmiles('c1c***1')

    def test_add_fragment_adds_a_five_membered_ring_fragment_to_napthalene_to_create_acenaphthylene(self):
        acenaphthylene = add_fragment(self.napthalene, self.five_membered_ring_fragment, (2, 3, 4), True)
        self.assertEqual(Chem.MolToInchi(acenaphthylene, options='-SNon'), 'InChI=1S/C12H8/c1-3-9-4-2-6-11-8-7-10(5-1)12(9)11/h1-8H')
        acenaphthylene = add_fragment(self.napthalene, self.five_membered_ring_fragment, (2, 3, 4), False)
        self.assertEqual(Chem.MolToInchi(acenaphthylene, options='-SNon'), 'InChI=1S/C12H8/c1-3-9-4-2-6-11-8-7-10(5-1)12(9)11/h1-8H')

    def test_add_fragment_adds_a_five_membered_ring_fragment_to_acenaphthylene_to_create_acenaphthylene(self):
        pyracyclene = add_fragment(self.acenaphthylene, self.five_membered_ring_fragment, (1, 2, 3), True)
        self.assertEqual(Chem.MolToInchi(pyracyclene, options='-SNon'), 'InChI=1S/C14H8/c1-2-10-7-8-12-4-3-11-6-5-9(1)13(10)14(11)12/h1-8H')
        pyracyclene = add_fragment(self.acenaphthylene, self.five_membered_ring_fragment, (1, 2, 3), False)
        self.assertEqual(Chem.MolToInchi(pyracyclene, options='-SNon'), 'InChI=1S/C14H8/c1-2-10-7-8-12-4-3-11-6-5-9(1)13(10)14(11)12/h1-8H')


class TestThreePointAttachmentPositions(TestCase):

    def setUp(self):
        self.benzene = Chem.MolFromSmiles('c1ccccc1')
        self.napthalene = Chem.MolFromSmiles('c1ccc2ccccc2c1')
        self.anthracene = Chem.MolFromSmiles('c1ccc2cc3ccccc3cc2c1')
        self.phenanthrene = Chem.MolFromSmiles('c1ccc2c(c1)ccc3c2cccc3')
        self.naphtho_2_3_b_thiophene = Chem.MolFromSmiles('c1ccc2cc3c(ccs3)cc2c1')
        self.acenaphthylene = Chem.MolFromSmiles('c1cc2cccc3c2c(c1)C=C3')
        self.isochrysene = Chem.MolFromSmiles('c1ccc2c(c1)c3ccccc3c4c2cccc4')

    # Benzene
    def test_three_point_attachment_positions_for_benzene(self):
        positions = three_point_attachment_positions(self.benzene)
        self.assertEqual([], positions)

    # Napthalene
    def test_three_point_attachment_positions_for_napthalene(self):
        positions = three_point_attachment_positions(self.napthalene)
        self.assertEqual([(2, 3, 4), (7, 8, 9)], positions)

    # Anthracene
    def test_three_point_attachment_positions_for_anthracene(self):
        positions = three_point_attachment_positions(self.anthracene)
        self.assertEqual([(2, 3, 4), (4, 5, 6), (9, 10, 11), (11, 12, 13)], positions)

    # Phenanthrene
    def test_three_point_attachment_positions_for_phenanthrene(self):
        positions = three_point_attachment_positions(self.phenanthrene)
        self.assertEqual([(5, 4, 6), (7, 8, 13)], positions)

    # Naphtho[2,3-b]thiophene
    def test_three_point_attachment_positions_for_naphtho_2_3_b_thiophene(self):
        positions = three_point_attachment_positions(self.naphtho_2_3_b_thiophene)
        self.assertEqual([(2, 3, 4), (10, 11, 12)], positions)

    # Acenaphthylene
    def test_three_point_attachment_positions_for_acenaphthylene(self):
        positions = three_point_attachment_positions(self.acenaphthylene)
        self.assertEqual([(1, 2, 3)], positions)

    # Isochrysene
    def test_three_point_attachment_positions_for_isochrysene(self):
        positions = three_point_attachment_positions(self.isochrysene)
        self.assertEqual([], positions)
