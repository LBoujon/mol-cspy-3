from unittest import TestCase
from rdkit import Chem
from cspy.molbuilder.addition.addition import add_fragment
from cspy.molbuilder.addition.addition import two_point_attachment_positions


class TestAddFragment(TestCase):

    def setUp(self):
        self.benzene = Chem.MolFromSmiles('c1ccccc1')
        self.thiophene = Chem.MolFromSmiles('c1ccsc1')
        self.pyridine = Chem.MolFromSmiles('c1ccncc1')
        self.napthalene = Chem.MolFromSmiles('c1ccc2ccccc2c1')
        self.acenaphthylene = Chem.MolFromSmiles('c1cc2cccc3c2c(c1)C=C3')
        self.barrelene = Chem.MolFromSmiles('C1=CC([H])2C=CC([H])1C=C2')
        self.spiro_4_4_nona_1_3_6_8_tetraene = Chem.MolFromSmiles('C1=CC2(C=C1)C=CC=C2')
        self.cyclopentadiene = Chem.MolFromSmiles('C1C=CC=C1')
        self.benzoquinone = Chem.MolFromSmiles('C1=CC(=O)C=CC1=O')

        self.benzene_fragment = Chem.MolFromSmiles('c1c**cc1')
        self.thiophene_fragment_1 = Chem.MolFromSmiles('c1**sc1')
        self.thiophene_fragment_2 = Chem.MolFromSmiles('*1*csc1')
        self.barrelene_fragment = Chem.MolFromSmiles('*1=*C([H])2C=CC([H])1C=C2')
        self.pyrrole_fragment = Chem.MolFromSmiles('*1*c[nH]c1')
        self.cyclopentadiene_fragment = Chem.MolFromSmiles('C1*=*C=C1')
        self.maleimide_fragment = Chem.MolFromSmiles('*1=*C(=O)NC(=O)1')
        self.one_h_imidazol_2_ol_fragment = Chem.MolFromSmiles('*1*NC(=O)N1')
        self.benzoquinone_fragment = Chem.MolFromSmiles('*1=*C(=O)C=CC1=O')

    # Benzene
    def test_add_fragment_adds_a_benzene_fragment_to_benzene_to_create_naphthalene(self):
        naphthalene = add_fragment(self.benzene, self.benzene_fragment, (0, 1), True)
        self.assertEqual(Chem.MolToInchi(naphthalene, options='-SNon'), 'InChI=1S/C10H8/c1-2-6-10-8-4-3-7-9(10)5-1/h1-8H')
        naphthalene = add_fragment(self.benzene, self.benzene_fragment, (0, 1), False)
        self.assertEqual(Chem.MolToInchi(naphthalene, options='-SNon'), 'InChI=1S/C10H8/c1-2-6-10-8-4-3-7-9(10)5-1/h1-8H')

    def test_add_fragment_adds_thiophene_fragment_1_to_benzene_to_create_benzothiophene(self):
        benzothiophene = add_fragment(self.benzene, self.thiophene_fragment_1, (0, 1), True)
        self.assertEqual(Chem.MolToInchi(benzothiophene, options='-SNon'), 'InChI=1S/C8H6S/c1-2-4-8-7(3-1)5-6-9-8/h1-6H')
        benzothiophene = add_fragment(self.benzene, self.thiophene_fragment_1, (0, 1), False)
        self.assertEqual(Chem.MolToInchi(benzothiophene, options='-SNon'), 'InChI=1S/C8H6S/c1-2-4-8-7(3-1)5-6-9-8/h1-6H')

    def test_add_fragment_adds_thiophene_fragment_2_to_benzene_to_create_two_benzothiophene(self):
        two_benzothiophene = add_fragment(self.benzene, self.thiophene_fragment_2, (0, 1), True)
        self.assertEqual(Chem.MolToInchi(two_benzothiophene, options='-SNon'), 'InChI=1S/C8H6S/c1-2-4-8-6-9-5-7(8)3-1/h1-6H')
        two_benzothiophene = add_fragment(self.benzene, self.thiophene_fragment_2, (0, 1), False)
        self.assertEqual(Chem.MolToInchi(two_benzothiophene, options='-SNon'), 'InChI=1S/C8H6S/c1-2-4-8-6-9-5-7(8)3-1/h1-6H')

    def test_add_fragment_adds_a_barrelene_fragment_to_benzene_to_create_tricyclo_6_2_2_0_2_7_dodeca_2_4_6_9_11_pentaene(self):
        tricyclo_6_2_2_0_2_7_dodeca_2_4_6_9_11_pentaene = add_fragment(self.benzene, self.barrelene_fragment, (0, 1), True)
        self.assertEqual(Chem.MolToInchi(tricyclo_6_2_2_0_2_7_dodeca_2_4_6_9_11_pentaene, options='-SNon'), 'InChI=1S/C12H10/c1-2-4-12-10-7-5-9(6-8-10)11(12)3-1/h1-10H')
        tricyclo_6_2_2_0_2_7_dodeca_2_4_6_9_11_pentaene = add_fragment(self.benzene, self.barrelene_fragment, (0, 1), False)
        self.assertEqual(Chem.MolToInchi(tricyclo_6_2_2_0_2_7_dodeca_2_4_6_9_11_pentaene, options='-SNon'), 'InChI=1S/C12H10/c1-2-4-12-10-7-5-9(6-8-10)11(12)3-1/h1-10H')

    def test_add_fragment_adds_a_maleimide_fragment_to_benzene_to_create_phthalimide(self):
        phthalimide = add_fragment(self.benzene, self.maleimide_fragment, (0, 1), True)
        self.assertEqual(Chem.MolToInchi(phthalimide, options='-SNon'), 'InChI=1S/C8H5NO2/c10-7-5-3-1-2-4-6(5)8(11)9-7/h1-4H,(H,9,10,11)')
        phthalimide = add_fragment(self.benzene, self.maleimide_fragment, (0, 1), False)
        self.assertEqual(Chem.MolToInchi(phthalimide, options='-SNon'), 'InChI=1S/C8H5NO2/c10-7-5-3-1-2-4-6(5)8(11)9-7/h1-4H,(H,9,10,11)')

    def test_add_fragment_adds_a_one_h_imidazol_2_ol_fragment_to_benzene_to_create_1_3_dihydro_2h_benzimidazol_2_one(self):
        one_3_dihydro_2h_benzimidazol_2_one = add_fragment(self.benzene, self.one_h_imidazol_2_ol_fragment, (0, 1), True)
        self.assertEqual(Chem.MolToInchi(one_3_dihydro_2h_benzimidazol_2_one, options='-SNon'), 'InChI=1S/C7H6N2O/c10-7-8-5-3-1-2-4-6(5)9-7/h1-4H,(H2,8,9,10)')
        one_3_dihydro_2h_benzimidazol_2_one = add_fragment(self.benzene, self.one_h_imidazol_2_ol_fragment, (0, 1), False)
        self.assertEqual(Chem.MolToInchi(one_3_dihydro_2h_benzimidazol_2_one, options='-SNon'), 'InChI=1S/C7H6N2O/c10-7-8-5-3-1-2-4-6(5)9-7/h1-4H,(H2,8,9,10)')

    def test_add_fragment_adds_a_benzoquinone_fragment_to_benzene_to_create_1_4_naphthoquinone(self):
        one_4_naphthoquinone = add_fragment(self.benzene, self.benzoquinone_fragment, (0, 1), True)
        self.assertEqual(Chem.MolToInchi(one_4_naphthoquinone, options='-SNon'), 'InChI=1S/C10H6O2/c11-9-5-6-10(12)8-4-2-1-3-7(8)9/h1-6H')
        one_4_naphthoquinone = add_fragment(self.benzene, self.benzoquinone_fragment, (0, 1), False)
        self.assertEqual(Chem.MolToInchi(one_4_naphthoquinone, options='-SNon'), 'InChI=1S/C10H6O2/c11-9-5-6-10(12)8-4-2-1-3-7(8)9/h1-6H')

    # Thiophene
    def test_add_fragment_adds_a_benzene_fragment_to_thiophene_to_create_benzothiophene(self):
        benzothiophene = add_fragment(self.thiophene, self.benzene_fragment, (1, 2), True)
        self.assertEqual(Chem.MolToInchi(benzothiophene, options='-SNon'), 'InChI=1S/C8H6S/c1-2-4-8-7(3-1)5-6-9-8/h1-6H')
        benzothiophene = add_fragment(self.thiophene, self.benzene_fragment, (1, 2), False)
        self.assertEqual(Chem.MolToInchi(benzothiophene, options='-SNon'), 'InChI=1S/C8H6S/c1-2-4-8-7(3-1)5-6-9-8/h1-6H')

    def test_add_fragment_adds_a_benzene_fragment_to_thiophene_to_create_two_benzothiophene(self):
        two_benzothiophene = add_fragment(self.thiophene, self.benzene_fragment, (0, 1), True)
        self.assertEqual(Chem.MolToInchi(two_benzothiophene, options='-SNon'), 'InChI=1S/C8H6S/c1-2-4-8-6-9-5-7(8)3-1/h1-6H')
        two_benzothiophene = add_fragment(self.thiophene, self.benzene_fragment, (0, 1), False)
        self.assertEqual(Chem.MolToInchi(two_benzothiophene, options='-SNon'), 'InChI=1S/C8H6S/c1-2-4-8-6-9-5-7(8)3-1/h1-6H')

    def test_add_fragment_adds_thiophene_fragment_1_to_thiophene_to_create_thienothiophene(self):
        thienothiophene = add_fragment(self.thiophene, self.thiophene_fragment_1, (1, 2), True)
        self.assertEqual(Chem.MolToInchi(thienothiophene, options='-SNon'), 'InChI=1S/C6H4S2/c1-3-7-6-2-4-8-5(1)6/h1-4H')

    def test_add_fragment_adds_thiophene_fragment_1_to_thiophene_to_create_thieno_2_3_b_thiophene(self):
        thieno_2_3_b_thiophene = add_fragment(self.thiophene, self.thiophene_fragment_1, (1, 2), False)
        self.assertEqual(Chem.MolToInchi(thieno_2_3_b_thiophene, options='-SNon'), 'InChI=1S/C6H4S2/c1-3-7-6-5(1)2-4-8-6/h1-4H')

    # Pyridine
    def test_add_fragment_adds_a_benzene_fragment_to_pyridine_to_create_isoquinoline(self):
        isoquinoline = add_fragment(self.pyridine, self.benzene_fragment, (0, 1), True)
        self.assertEqual(Chem.MolToInchi(isoquinoline, options='-SNon'), 'InChI=1S/C9H7N/c1-2-4-9-7-10-6-5-8(9)3-1/h1-7H')
        isoquinoline = add_fragment(self.pyridine, self.benzene_fragment, (0, 1), False)
        self.assertEqual(Chem.MolToInchi(isoquinoline, options='-SNon'), 'InChI=1S/C9H7N/c1-2-4-9-7-10-6-5-8(9)3-1/h1-7H')

    def test_add_fragment_adds_a_benzene_fragment_to_pyridine_to_create_leucoline(self):
        leucoline = add_fragment(self.pyridine, self.benzene_fragment, (1, 2), True)
        self.assertEqual(Chem.MolToInchi(leucoline, options='-SNon'), 'InChI=1S/C9H7N/c1-2-6-9-8(4-1)5-3-7-10-9/h1-7H')
        leucoline = add_fragment(self.pyridine, self.benzene_fragment, (1, 2), False)
        self.assertEqual(Chem.MolToInchi(leucoline, options='-SNon'), 'InChI=1S/C9H7N/c1-2-6-9-8(4-1)5-3-7-10-9/h1-7H')

    # Napthalene
    def test_add_fragment_adds_a_benzene_fragment_to_napthalene_to_create_anthracene(self):
        anthracene = add_fragment(self.napthalene, self.benzene_fragment, (0, 1), True)
        self.assertEqual(Chem.MolToInchi(anthracene, options='-SNon'), 'InChI=1S/C14H10/c1-2-6-12-10-14-8-4-3-7-13(14)9-11(12)5-1/h1-10H')
        anthracene = add_fragment(self.napthalene, self.benzene_fragment, (0, 1), False)
        self.assertEqual(Chem.MolToInchi(anthracene, options='-SNon'), 'InChI=1S/C14H10/c1-2-6-12-10-14-8-4-3-7-13(14)9-11(12)5-1/h1-10H')

    def test_add_fragment_adds_a_benzene_fragment_to_napthalene_to_create_phenanthrene(self):
        phenanthrene = add_fragment(self.napthalene, self.benzene_fragment, (1, 2), True)
        self.assertEqual(Chem.MolToInchi(phenanthrene, options='-SNon'), 'InChI=1S/C14H10/c1-3-7-13-11(5-1)9-10-12-6-2-4-8-14(12)13/h1-10H')
        phenanthrene = add_fragment(self.napthalene, self.benzene_fragment, (1, 2), False)
        self.assertEqual(Chem.MolToInchi(phenanthrene, options='-SNon'), 'InChI=1S/C14H10/c1-3-7-13-11(5-1)9-10-12-6-2-4-8-14(12)13/h1-10H')

    # Acenaphthylene
    def test_add_fragment_adds_thiophene_fragment_1_to_acenaphthylene_to_create_acenaphtho_1_2_b_thiophene(self):
        acenaphtho_1_2_b_thiophene = add_fragment(self.acenaphthylene, self.thiophene_fragment_1, (10, 11), True)
        self.assertEqual(Chem.MolToInchi(acenaphtho_1_2_b_thiophene, options='-SNon'), 'InChI=1S/C14H8S/c1-3-9-4-2-6-12-13(9)10(5-1)11-7-8-15-14(11)12/h1-8H')
        acenaphtho_1_2_b_thiophene = add_fragment(self.acenaphthylene, self.thiophene_fragment_1, (10, 11), False)
        self.assertEqual(Chem.MolToInchi(acenaphtho_1_2_b_thiophene, options='-SNon'), 'InChI=1S/C14H8S/c1-3-9-4-2-6-12-13(9)10(5-1)11-7-8-15-14(11)12/h1-8H')

    def test_add_fragment_adds_thiophene_fragment_2_to_acenaphthylene_to_create_acenaphtho_1_2_c_thiophene(self):
        acenaphtho_1_2_c_thiophene = add_fragment(self.acenaphthylene, self.thiophene_fragment_2, (10, 11), True)
        self.assertEqual(Chem.MolToInchi(acenaphtho_1_2_c_thiophene, options='-SNon'), 'InChI=1S/C14H8S/c1-3-9-4-2-6-11-13-8-15-7-12(13)10(5-1)14(9)11/h1-8H')
        acenaphtho_1_2_c_thiophene = add_fragment(self.acenaphthylene, self.thiophene_fragment_2, (10, 11), False)
        self.assertEqual(Chem.MolToInchi(acenaphtho_1_2_c_thiophene, options='-SNon'), 'InChI=1S/C14H8S/c1-3-9-4-2-6-11-13-8-15-7-12(13)10(5-1)14(9)11/h1-8H')

    # Barrelene
    def test_add_fragment_adds_a_pyrrole_fragment_to_barralene_to_create_4_azatricyclo_5_2_2_0_2_6_undeca_2_5_8_10_tetraene(self):
        azatricyclo_5_2_2_0_2_6_undeca_2_5_8_10_tetraene = add_fragment(self.barrelene, self.pyrrole_fragment, (0, 1), True)
        self.assertEqual(Chem.MolToInchi(azatricyclo_5_2_2_0_2_6_undeca_2_5_8_10_tetraene, options='-SNon'), 'InChI=1S/C10H9N/c1-2-8-4-3-7(1)9-5-11-6-10(8)9/h1-8,11H')
        azatricyclo_5_2_2_0_2_6_undeca_2_5_8_10_tetraene = add_fragment(self.barrelene, self.pyrrole_fragment, (0, 1), False)
        self.assertEqual(Chem.MolToInchi(azatricyclo_5_2_2_0_2_6_undeca_2_5_8_10_tetraene, options='-SNon'), 'InChI=1S/C10H9N/c1-2-8-4-3-7(1)9-5-11-6-10(8)9/h1-8,11H')

    # Spiro[4.4]nona-1,3,6,8-tetraene
    def test_add_fragment_adds_a_benzene_fragment_to_spiro_4_4_nona_1_3_6_8_tetraene_to_create_molecule_0(self):
        molecule = add_fragment(self.spiro_4_4_nona_1_3_6_8_tetraene, self.benzene_fragment, (0, 1), False)
        self.assertEqual(Chem.MolToInchi(molecule, options='-SNon'), 'InChI=1S/C13H10/c1-2-6-12-11(5-1)7-10-13(12)8-3-4-9-13/h1-10H')

    # 1,3-Cyclopentadiene
    def test_add_fragment_adds_cyclopentadiene_fragment_to_cyclopentadiene_to_create_1_5_dihydropentalene(self):
        one_5_dihydropentalene = add_fragment(self.cyclopentadiene, self.cyclopentadiene_fragment, (2, 3), False)
        self.assertEqual(Chem.MolToInchi(one_5_dihydropentalene, options='-SNon'), 'InChI=1S/C8H8/c1-3-7-5-2-6-8(7)4-1/h1,3,5-6H,2,4H2')

    # 1,4-Benzoquinone
    def test_add_fragment_adds_a_benzene_fragment_to_benzoquinone_to_create_1_4_naphthoquinone(self):
        one_4_naphthoquinone = add_fragment(self.benzoquinone, self.benzene_fragment, (0, 1), False)
        self.assertEqual(Chem.MolToInchi(one_4_naphthoquinone, options='-SNon'), 'InChI=1S/C10H6O2/c11-9-5-6-10(12)8-4-2-1-3-7(8)9/h1-6H')
        one_4_naphthoquinone = add_fragment(self.benzoquinone, self.benzene_fragment, (4, 5), False)
        self.assertEqual(Chem.MolToInchi(one_4_naphthoquinone, options='-SNon'), 'InChI=1S/C10H6O2/c11-9-5-6-10(12)8-4-2-1-3-7(8)9/h1-6H')
        one_4_naphthoquinone = add_fragment(self.benzoquinone, self.benzene_fragment, (0, 1), True)
        self.assertEqual(Chem.MolToInchi(one_4_naphthoquinone, options='-SNon'), 'InChI=1S/C10H6O2/c11-9-5-6-10(12)8-4-2-1-3-7(8)9/h1-6H')
        one_4_naphthoquinone = add_fragment(self.benzoquinone, self.benzene_fragment, (4, 5), True)
        self.assertEqual(Chem.MolToInchi(one_4_naphthoquinone, options='-SNon'), 'InChI=1S/C10H6O2/c11-9-5-6-10(12)8-4-2-1-3-7(8)9/h1-6H')


class TestTwoPointAttachementPositions(TestCase):

    def setUp(self):
        self.benzene = Chem.MolFromSmiles('c1ccccc1')
        self.thiophene = Chem.MolFromSmiles('c1ccsc1')
        self.pyridine = Chem.MolFromSmiles('c1ccncc1')
        self.napthalene = Chem.MolFromSmiles('c1ccc2ccccc2c1')
        self.benzothiophene = Chem.MolFromSmiles('c1ccc2c(c1)ccs2')
        self.two_benzothiophene = Chem.MolFromSmiles('c1ccc2cscc2c1')
        self.isoquinoline = Chem.MolFromSmiles('c1ccc2cnccc2c1')
        self.naphtho_2_3_c_thiophene = Chem.MolFromSmiles('s2cc1cc3c(cc1c2)cccc3')
        self.naphtho_1_2_c_thiophene = Chem.MolFromSmiles('c1ccc2c(c1)ccc3c2csc3')
        self.thieno_3_2_f_1_benzothiophene = Chem.MolFromSmiles('c1csc2c1cc3ccsc3c2')
        self.acenaphthylene = Chem.MolFromSmiles('c1cc2cccc3c2c(c1)C=C3')
        self.barrelene = Chem.MolFromSmiles('C1=CC([H])2C=CC([H])1C=C2')
        self.tricyclo_6_2_2_0_2_7_dodeca_2_4_6_9_11_pentaene = Chem.MolFromSmiles('c1ccc2c(c1)C([H])3C=CC([H])2C=C3')
        self.phthalimide = Chem.MolFromSmiles('c1ccc2c(c1)C(=O)NC2=O')
        self.imidazolone = Chem.MolFromSmiles('C1=NC(=O)N=C1')
        self.thieno_2_3_b_thiophene = Chem.MolFromSmiles('c1csc2c1ccs2')
        self.bisthieno_2_3_b_3_2_d_thiophene = Chem.MolFromSmiles('c1csc2c1c3ccsc3s2')
        self.four_azatricyclo_5_2_2_0_2_6_undeca_2_5_8_10_tetraene = Chem.MolFromSmiles('c1c2c(c[nH]1)C3C=CC2C=C3')
        self.spiro_4_4_nona_1_3_6_8_tetraene = Chem.MolFromSmiles('C1=CC2(C=C1)C=CC=C2')
        self.benzoquinone = Chem.MolFromSmiles('C1=CC(=O)C=CC1=O')

        self.benzene_fragment = Chem.MolFromSmiles('c1c**cc1')
        self.thiophene_fragment_1 = Chem.MolFromSmiles('c1**sc1')
        self.thiophene_fragment_2 = Chem.MolFromSmiles('*1*csc1')
        self.thieno_3_4_b_pyridine_fragment = Chem.MolFromSmiles('c1scc2n**cc12')
        self.barrelene_fragment = Chem.MolFromSmiles('*1=*C([H])2C=CC([H])1C=C2')
        self.maleimide_fragment = Chem.MolFromSmiles('*1=*C(=O)NC(=O)1')
        self.benzoquinone_fragment = Chem.MolFromSmiles('*1=*C(=O)C=CC1=O')

    # Benzene
    def test_two_point_attachment_positions_for_benzene_with_a_benzene_fragment(self):
        positions = two_point_attachment_positions(self.benzene, self.benzene_fragment)
        self.assertEqual([(0, 1), (0, 5), (1, 2), (2, 3), (3, 4), (4, 5)], positions)

    def test_two_point_attachment_positions_for_benzene_with_a_barrelene_fragment(self):
        positions = two_point_attachment_positions(self.benzene, self.benzene_fragment)
        self.assertEqual([(0, 1), (0, 5), (1, 2), (2, 3), (3, 4), (4, 5)], positions)

    def test_two_point_attachment_positions_for_benzene_with_a_maleimide_fragment(self):
        positions = two_point_attachment_positions(self.benzene, self.benzene_fragment)
        self.assertEqual([(0, 1), (0, 5), (1, 2), (2, 3), (3, 4), (4, 5)], positions)

    def test_two_point_attachment_positions_for_benzene_with_a_benzoquinone_fragment(self):
        positions = two_point_attachment_positions(self.benzene, self.benzene_fragment)
        self.assertEqual([(0, 1), (0, 5), (1, 2), (2, 3), (3, 4), (4, 5)], positions)

    # Thiophene
    def test_two_point_attachment_positions_for_thiophene_with_a_benzene_fragment(self):
        positions = two_point_attachment_positions(self.thiophene, self.benzene_fragment)
        self.assertEqual([(0, 1), (0, 4), (1, 2)], positions)

    def test_two_point_attachment_positions_for_thiophene_with_thiophene_fragment_2(self):
        positions = two_point_attachment_positions(self.thiophene, self.thiophene_fragment_2)
        self.assertEqual([(0, 4), (1, 2)], positions)

    def test_two_point_attachment_positions_for_thiophene_with_a_thieno_3_4_b_pyridine_fragment(self):
        positions = two_point_attachment_positions(self.thiophene, self.thieno_3_4_b_pyridine_fragment)
        self.assertEqual([(0, 4), (1, 2)], positions)

    def test_two_point_attachment_positions_for_thiophene_with_a_maleimide_fragment(self):
        positions = two_point_attachment_positions(self.thiophene, self.maleimide_fragment)
        self.assertEqual([(0, 1), (0, 4), (1, 2)], positions)

    # Pyridine
    def test_two_point_attachment_positions_for_pyridine_with_a_benzene_fragment(self):
        positions = two_point_attachment_positions(self.pyridine, self.benzene_fragment)
        self.assertEqual([(0, 1), (0, 5), (1, 2), (4, 5)], positions)

    # Napthalene
    def test_two_point_attachment_positions_for_napthalene_with_thiophene_fragment_2(self):
        positions = two_point_attachment_positions(self.napthalene, self.thiophene_fragment_2)
        self.assertEqual([(0, 1), (0, 9), (1, 2), (4, 5), (5, 6), (6, 7)], positions)

    # Benzothiophene
    def test_two_point_attachment_positions_for_benzothiophene_with_thiophene_fragment_2(self):
        positions = two_point_attachment_positions(self.benzothiophene, self.thiophene_fragment_2)
        self.assertEqual([(0, 1), (0, 5), (1, 2), (6, 7)], positions)

    # 2-Benzothiophene
    def test_two_point_attachment_positions_for_2_benzothiophene_with_thiophene_fragment_2(self):
        positions = two_point_attachment_positions(self.two_benzothiophene, self.thiophene_fragment_2)
        self.assertEqual([(0, 8), (1, 2)], positions)

    # Isoquinoline
    def test_two_point_attachment_positions_for_isoquinoline_with_a_benzene_fragment(self):
        positions = two_point_attachment_positions(self.isoquinoline, self.benzene_fragment)
        self.assertEqual([(0, 1), (0, 9), (1, 2), (6, 7)], positions)

    def test_two_point_attachment_positions_for_isoquinoline_with_thiophene_fragment_1(self):
        positions = two_point_attachment_positions(self.isoquinoline, self.thiophene_fragment_1)
        self.assertEqual([(0, 1), (0, 9), (1, 2), (6, 7)], positions)

    def test_two_point_attachment_positions_for_isoquinoline_with_thiophene_fragment_2(self):
        positions = two_point_attachment_positions(self.isoquinoline, self.thiophene_fragment_2)
        self.assertEqual([(0, 1), (0, 9), (1, 2), (6, 7)], positions)

    # Naphtho[2,3-c]thiophene
    def test_two_point_attachment_positions_for_naphtho_2_3_c_thiophene_with_thiophene_fragment_2(self):
        positions = two_point_attachment_positions(self.naphtho_2_3_c_thiophene, self.thiophene_fragment_2)
        self.assertEqual([(9, 10), (11, 12)], positions)

    # Naphtho[1,2-c]thiophene
    def test_two_point_attachment_positions_for_naphtho_1_2_c_thiophene_with_thiophene_fragment_2(self):
        positions = two_point_attachment_positions(self.naphtho_1_2_c_thiophene, self.thiophene_fragment_2)
        self.assertEqual([(0, 1), (0, 5), (1, 2), (6, 7)], positions)

    # Thieno[3,2-f][1]benzothiophene
    def test_two_point_attachment_positions_for_thieno_3_2_f_1_benzothiophene_with_a_benzene_fragment(self):
        positions = two_point_attachment_positions(self.thieno_3_2_f_1_benzothiophene, self.benzene_fragment)
        self.assertEqual([(0, 1), (7, 8)], positions)

    # Acenaphthylene
    def test_two_point_attachment_positions_for_acenaphthylene_with_thiophene_fragment_1(self):
        positions = two_point_attachment_positions(self.acenaphthylene, self.thiophene_fragment_1)
        self.assertEqual([(0, 1), (0, 9), (3, 4), (4, 5), (10, 11)], positions)

    def test_two_point_attachment_positions_for_acenaphthylene_with_thiophene_fragment_2(self):
        positions = two_point_attachment_positions(self.acenaphthylene, self.thiophene_fragment_2)
        self.assertEqual([(0, 1), (0, 9), (3, 4), (4, 5), (10, 11)], positions)

    # Barrelene
    def test_two_point_attachment_positions_for_barrelene_with_a_benzene_fragment(self):
        positions = two_point_attachment_positions(self.barrelene, self.benzene_fragment)
        self.assertEqual([(0, 1), (3, 4), (6, 7)], positions)

    def test_two_point_attachment_positions_for_barrelene_with_thiophene_fragment_2(self):
        positions = two_point_attachment_positions(self.barrelene, self.thiophene_fragment_2)
        self.assertEqual([(0, 1), (3, 4), (6, 7)], positions)

    # Tricyclo[6.2.2.0~2,7~]dodeca-2,4,6,9,11-pentaene
    def test_two_point_attachment_positions_for_tricyclo_6_2_2_0_2_7_dodeca_2_4_6_9_11_pentaene_with_a_benzene_fragment(self):
        positions = two_point_attachment_positions(self.tricyclo_6_2_2_0_2_7_dodeca_2_4_6_9_11_pentaene, self.benzene_fragment)
        self.assertEqual([(0, 1), (0, 5), (1, 2), (7, 8), (10, 11)], positions)

    # Phthalimide
    def test_two_point_attachment_positions_for_phthalimide_with_a_benzene_fragment(self):
        positions = two_point_attachment_positions(self.phthalimide, self.benzene_fragment)
        self.assertEqual([(0, 1), (0, 5), (1, 2)], positions)

    def test_two_point_attachment_positions_for_phthalimide_with_thiophene_fragment_1(self):
        positions = two_point_attachment_positions(self.phthalimide, self.thiophene_fragment_1)
        self.assertEqual([(0, 1), (0, 5), (1, 2)], positions)

    def test_two_point_attachment_positions_for_phthalimide_with_thiophene_fragment_2(self):
        positions = two_point_attachment_positions(self.phthalimide, self.thiophene_fragment_2)
        self.assertEqual([(0, 1), (0, 5), (1, 2)], positions)

    # Imidazolone
    def test_two_point_attachment_positions_for_imidazolone_with_a_benzene_fragment(self):
        positions = two_point_attachment_positions(self.imidazolone, self.benzene_fragment)
        self.assertEqual([(0, 5)], positions)

    def test_two_point_attachment_positions_for_imidazolone_with_thiophene_fragment_1(self):
        positions = two_point_attachment_positions(self.imidazolone, self.thiophene_fragment_1)
        self.assertEqual([(0, 5)], positions)

    def test_two_point_attachment_positions_for_imidazolone_with_thiophene_fragment_2(self):
        positions = two_point_attachment_positions(self.imidazolone, self.thiophene_fragment_2)
        self.assertEqual([], positions)

    # Thieno[2,3-b]thiophene
    def test_two_point_attachment_positions_for_thieno_2_3_b_thiophene_with_a_benzene_fragment(self):
        positions = two_point_attachment_positions(self.thieno_2_3_b_thiophene, self.benzene_fragment)
        self.assertEqual([(0, 1), (5, 6)], positions)

    def test_two_point_attachment_positions_for_thieno_2_3_b_thiophene_with_thiophene_fragment_1(self):
        positions = two_point_attachment_positions(self.thieno_2_3_b_thiophene, self.thiophene_fragment_1)
        self.assertEqual([(0, 1), (5, 6)], positions)

    def test_two_point_attachment_positions_for_thieno_2_3_b_thiophene_with_thiophene_fragment_2(self):
        positions = two_point_attachment_positions(self.thieno_2_3_b_thiophene, self.thiophene_fragment_2)
        self.assertEqual([(0, 1), (5, 6)], positions)

    # Bisthieno[2,3-b:3',2'-d]thiophene
    def test_two_point_attachment_positions_for_bisthieno_2_3_b_3_2_d_thiophene_with_a_benzene_fragment(self):
        positions = two_point_attachment_positions(self.bisthieno_2_3_b_3_2_d_thiophene, self.benzene_fragment)
        self.assertEqual([(0, 1), (6, 7)], positions)

    def test_two_point_attachment_positions_for_bisthieno_2_3_b_3_2_d_thiophene_with_thiophene_fragment_1(self):
        positions = two_point_attachment_positions(self.bisthieno_2_3_b_3_2_d_thiophene, self.thiophene_fragment_1)
        self.assertEqual([(0, 1), (6, 7)], positions)

    def test_two_point_attachment_positions_for_bisthieno_2_3_b_3_2_d_thiophene_with_thiophene_fragment_2(self):
        positions = two_point_attachment_positions(self.bisthieno_2_3_b_3_2_d_thiophene, self.thiophene_fragment_2)
        self.assertEqual([(0, 1), (6, 7)], positions)

    # 4-Azatricyclo[5.2.2.02,6]undeca-2,5,8,10-tetraene
    def test_two_point_attachment_positions_for_four_azatricyclo_5_2_2_0_2_6_undeca_2_5_8_10_tetraene_with_a_benzene_fragment(self):
        positions = two_point_attachment_positions(self.four_azatricyclo_5_2_2_0_2_6_undeca_2_5_8_10_tetraene, self.benzene_fragment)
        self.assertEqual([(6, 7), (9, 10)], positions)

    def test_two_point_attachment_positions_for_four_azatricyclo_5_2_2_0_2_6_undeca_2_5_8_10_tetraene_with_thiophene_fragment_1(self):
        positions = two_point_attachment_positions(self.four_azatricyclo_5_2_2_0_2_6_undeca_2_5_8_10_tetraene, self.thiophene_fragment_1)
        self.assertEqual([(6, 7), (9, 10)], positions)

    def test_two_point_attachment_positions_for_four_azatricyclo_5_2_2_0_2_6_undeca_2_5_8_10_tetraene_with_thiophene_fragment_2(self):
        positions = two_point_attachment_positions(self.four_azatricyclo_5_2_2_0_2_6_undeca_2_5_8_10_tetraene, self.thiophene_fragment_2)
        self.assertEqual([(6, 7), (9, 10)], positions)

    # Spiro[4.4]nona-1,3,6,8-tetraene
    def test_two_point_attachment_positions_for_spiro_4_4_nona_1_3_6_8_tetraene_with_a_benzene_fragment(self):
        positions = two_point_attachment_positions(self.spiro_4_4_nona_1_3_6_8_tetraene, self.benzene_fragment)
        self.assertEqual([(0, 1), (0, 4), (3, 4), (5, 6), (6, 7), (7, 8)], positions)

    def test_two_point_attachment_positions_for_spiro_4_4_nona_1_3_6_8_tetraene_with_thiophene_fragment_1(self):
        positions = two_point_attachment_positions(self.spiro_4_4_nona_1_3_6_8_tetraene, self.thiophene_fragment_1)
        self.assertEqual([(0, 1), (0, 4), (3, 4), (5, 6), (6, 7), (7, 8)], positions)

    def test_two_point_attachment_positions_for_spiro_4_4_nona_1_3_6_8_tetraene_with_thiophene_fragment_2(self):
        positions = two_point_attachment_positions(self.spiro_4_4_nona_1_3_6_8_tetraene, self.thiophene_fragment_2)
        self.assertEqual([(0, 1), (3, 4), (5, 6), (7, 8)], positions)

    # 1,4-Benzoquinone
    def test_two_point_attachment_positions_for_benzoquinone_with_a_benzene_fragment(self):
        positions = two_point_attachment_positions(self.benzoquinone, self.benzene_fragment)
        self.assertEqual([(0, 1), (4, 5)], positions)
