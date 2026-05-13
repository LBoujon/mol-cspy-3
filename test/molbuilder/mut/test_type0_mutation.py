from unittest import TestCase
from rdkit import Chem
from cspy.molbuilder.mutation.substitution import type0_mutation
from cspy.molbuilder.mutation.substitution import mutable_positions


class TestType0Mutation(TestCase):

    def setUp(self):
        self.benzene = Chem.MolFromSmiles('c1ccccc1')
        self.benzonitrile = Chem.MolFromSmiles('c1ccc(cc1)C#N')
        self.pyridine = Chem.MolFromSmiles('c1ccncc1')
        self.dichloro_5_methylene_1_3_cyclopentadiene = Chem.MolFromSmiles('C=C1C(=CC=C1Cl)Cl')
        self.piazthiole = Chem.MolFromSmiles('c1ccc2c(c1)nsn2')
        self.cyclopentadiene = Chem.MolFromSmiles('C1C=CC=C1')
        self.chlorocyclopentadiene = Chem.MolFromSmiles('C1C=CC=C1Cl')
        self.one_2_dichloro_1_3_cyclopentadiene = Chem.MolFromSmiles('C1C=CC(=C1Cl)Cl')
        self.trichlorocyclopentadiene = Chem.MolFromSmiles('C1C=C(C(=C1Cl)Cl)Cl')

        self.hydrogen_fragment = Chem.MolFromSmarts('[#6R1&H]')
        self.fluorine_fragment = Chem.MolFromSmarts('[#6R1]-F')
        self.chlorine_fragment = Chem.MolFromSmarts('[#6R1]-Cl')
        self.cyanide_fragment = Chem.MolFromSmarts('[#6R1]-C#N')
        self.nitrogen_fragment = Chem.MolFromSmarts('[#7R1&H0]')

    # Benzene
    def test_type0_mutator_with_fluorine_fragment_and_benzene_creates_fluorobenzene(self):
        fluorobenzene = type0_mutation(self.benzene, self.fluorine_fragment, 0)
        self.assertEqual(Chem.MolToInchi(fluorobenzene, options='-SNon'), 'InChI=1S/C6H5F/c7-6-4-2-1-3-5-6/h1-5H')

    def test_type0_mutator_with_cyanide_fragment_and_benzene_creates_benzonitrile(self):
        benzonitrile = type0_mutation(self.benzene, self.cyanide_fragment, 0)
        self.assertEqual(Chem.MolToInchi(benzonitrile, options='-SNon'), 'InChI=1S/C7H5N/c8-6-7-4-2-1-3-5-7/h1-5H')

    def test_type0_mutator_with_benzene_and_nitrogen_fragment_to_create_pyridine(self):
        pyridine = type0_mutation(self.benzene, self.nitrogen_fragment, 0)
        self.assertEqual(Chem.MolToInchi(pyridine, options='-SNon'), 'InChI=1S/C5H5N/c1-2-4-6-5-3-1/h1-5H')

    # Benzonitrile
    def test_type0_mutator_with_hydrogen_fragment_and_benzonitrile_creates_benzene(self):
        benzene = type0_mutation(self.benzonitrile, self.hydrogen_fragment, 3)
        self.assertEqual(Chem.MolToInchi(benzene, options='-SNon'), 'InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H')

    def test_type0_mutator_with_benzonitrile_and_fluorine_fragment_create_fluorobenzene(self):
        fluorobenzene = type0_mutation(self.benzonitrile, self.fluorine_fragment, 3)
        self.assertEqual(Chem.MolToInchi(fluorobenzene, options='-SNon'), 'InChI=1S/C6H5F/c7-6-4-2-1-3-5-6/h1-5H')

    def test_type0_mutator_with_cyanide_fragment_and_benzonitrile_creates_phthalonitrile(self):
        phthalonitrile = type0_mutation(self.benzonitrile, self.cyanide_fragment, 2)
        self.assertEqual(Chem.MolToInchi(phthalonitrile, options='-SNon'), 'InChI=1S/C8H4N2/c9-5-7-3-1-2-4-8(7)6-10/h1-4H')

    # Pyridine
    def test_type0_mutator_with_pyridine_and_hydrogen_fragment_to_create_benzene(self):
        benzene = type0_mutation(self.pyridine, self.hydrogen_fragment, 3)
        self.assertEqual(Chem.MolToInchi(benzene, options='-SNon'), 'InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H')

    def test_type0_mutator_with_pyridine_and_nitrogen_fragment_to_create_pyrazine(self):
        pyrazine = type0_mutation(self.pyridine, self.nitrogen_fragment, 0)
        self.assertEqual(Chem.MolToInchi(pyrazine, options='-SNon'), 'InChI=1S/C4H4N2/c1-2-6-4-3-5-1/h1-4H')

    def test_type0_mutator_with_pyridine_and_nitrogen_fragment_to_create_pyridazine(self):
        pyridazine = type0_mutation(self.pyridine, self.nitrogen_fragment, 2)
        self.assertEqual(Chem.MolToInchi(pyridazine, options='-SNon'), 'InChI=1S/C4H4N2/c1-2-4-6-5-3-1/h1-4H')

    # 1,4-Dichloro-5-methylene-1,3-cyclopentadiene
    def test_type0_mutator_with_dichloro_5_methylene_1_3_cyclopentadiene_and_fluorine_fragment_create_chloro_5_methylene_1_3_cyclopentadiene(self):
        chloro_5_methylene_1_3_cyclopentadiene = type0_mutation(self.dichloro_5_methylene_1_3_cyclopentadiene, self.hydrogen_fragment, 2)
        self.assertEqual(Chem.MolToInchi(chloro_5_methylene_1_3_cyclopentadiene, options='-SNon'), 'InChI=1S/C6H5Cl/c1-5-3-2-4-6(5)7/h2-4H,1H2')

    # Piazthiole
    def test_type0_mutator_with_piazthiole_and_chlorine_fragment_to_create_chloro_2_1_benzothiazole(self):
        chloro_2_1_benzothiazole = type0_mutation(self.piazthiole, self.chlorine_fragment, 6)
        self.assertEqual(Chem.MolToInchi(chloro_2_1_benzothiazole, options='-SNon'), 'InChI=1S/C7H4ClNS/c8-7-5-3-1-2-4-6(5)9-10-7/h1-4H')

    # 1,3-Cyclopentadiene
    def test_type0_mutator_with_cyclopentadiene_and_chlorine_fragment_to_create_chlorocyclopentadiene(self):
        chlorocyclopentadiene = type0_mutation(self.cyclopentadiene, self.chlorine_fragment, 1)
        self.assertEqual(Chem.MolToInchi(chlorocyclopentadiene, options='-SNon'), 'InChI=1S/C5H5Cl/c6-5-3-1-2-4-5/h1-3H,4H2')

    # 1-Chloro-1,3-cyclopentadiene
    def test_type0_mutator_with_chlorocyclopentadiene_and_chlorine_fragment_to_create_1_2_dichloro_1_3_cyclopentadiene(self):
        one_2_dichloro_1_3_cyclopentadiene = type0_mutation(self.chlorocyclopentadiene, self.chlorine_fragment, 3)
        self.assertEqual(Chem.MolToInchi(one_2_dichloro_1_3_cyclopentadiene, options='-SNon'), 'InChI=1S/C5H4Cl2/c6-4-2-1-3-5(4)7/h1-2H,3H2')

    # 1,2-Dichloro-1,3-cyclopentadiene
    def test_type0_mutator_with_1_2_dichloro_1_3_cyclopentadiene_and_chlorine_fragment_to_create_trichlorocyclopentadiene(self):
        trichlorocyclopentadiene = type0_mutation(self.one_2_dichloro_1_3_cyclopentadiene, self.chlorine_fragment, 2)
        self.assertEqual(Chem.MolToInchi(trichlorocyclopentadiene, options='-SNon'), 'InChI=1S/C5H3Cl3/c6-3-1-2-4(7)5(3)8/h1H,2H2')

    # 1,2,3-Trichloro-1,3-cyclopentadiene
    def test_type0_mutator_with_trichlorocyclopentadiene_and_chlorine_fragment_to_create_1_2_3_4_tetrachloro_1_3_cyclopentadiene(self):
        one_2_3_4_tetrachloro_1_3_cyclopentadiene = type0_mutation(self.trichlorocyclopentadiene, self.chlorine_fragment, 1)
        self.assertEqual(Chem.MolToInchi(one_2_3_4_tetrachloro_1_3_cyclopentadiene, options='-SNon'), 'InChI=1S/C5H2Cl4/c6-2-1-3(7)5(9)4(2)8/h1H2')


class TestType0MutablePositions(TestCase):

    def setUp(self):
        self.benzene = Chem.MolFromSmiles('c1ccccc1')
        self.benzonitrile = Chem.MolFromSmiles('c1ccc(cc1)C#N')
        self.napthalene = Chem.MolFromSmiles('c1ccc2ccccc2c1')
        self.acridine = Chem.MolFromSmiles('c1ccc2c(c1)cc3ccccc3n2')
        self.pyrrole = Chem.MolFromSmiles('c1cc[nH]c1')
        self.fulvene = Chem.MolFromSmiles('C=C1C=CC=C1')
        self.pyridine = Chem.MolFromSmiles('c1ccncc1')
        self.benzothiophene = Chem.MolFromSmiles('c1ccc2c(c1)ccs2')
        self.pyrazine = Chem.MolFromSmiles('c1cnccn1')
        self.piazthiole = Chem.MolFromSmiles('c1ccc2c(c1)nsn2')
        self.spiro_4_4_nona_1_3_6_8_tetraene = Chem.MolFromSmiles('C1=CC2(C=C1)C=CC=C2')
        self.cyclopentadiene = Chem.MolFromSmiles('C1C=CC=C1')

        self.hydrogen_fragment = Chem.MolFromSmarts('[#6R1&H]')
        self.fluorine_fragment = Chem.MolFromSmarts('[#6R1]-F')
        self.chlorine_fragment = Chem.MolFromSmarts('[#6R1]-Cl')
        self.nitrogen_fragment = Chem.MolFromSmarts('[#7R1&H0]')
        self.cyanide_fragment = Chem.MolFromSmarts('[#6R1]-C#N')

        type0_fragments_smarts = ['[#6R1&H]', '[#6R1]-F', '[#6R1]-Cl', '[#7R1&H0]', '[#6R1]-C#N']
        self.type0_fragment = [Chem.MolFromSmarts(i) for i in type0_fragments_smarts]

    # Benzene
    def test_type0_bondable_positions_for_benzene_with_hydrogen_fragment(self):
        bondable = mutable_positions(self.benzene, self.hydrogen_fragment, self.type0_fragment)
        self.assertEqual([], bondable)

    def test_type0_bondable_positions_for_benzene_with_fluorine_fragment(self):
        bondable = mutable_positions(self.benzene, self.fluorine_fragment, self.type0_fragment)
        self.assertEqual([0, 1, 2, 3, 4, 5], bondable)

    def test_mutable_positions_for_benzene_with_nitrogen_fragment(self):
        bondable = mutable_positions(self.benzene, self.nitrogen_fragment, self.type0_fragment)
        self.assertEqual([0, 1, 2, 3, 4, 5], bondable)

    def test_type0_bondable_positions_for_benzene_with_cyanide_fragment(self):
        bondable = mutable_positions(self.benzene, self.cyanide_fragment, self.type0_fragment)
        self.assertEqual([0, 1, 2, 3, 4, 5], bondable)

    # Benzonitrile
    def test_type0_bondable_positions_for_benzonitrile_with_hydrogen_fragment(self):
        bondable = mutable_positions(self.benzonitrile, self.hydrogen_fragment, self.type0_fragment)
        self.assertEqual([3], bondable)

    def test_type0_bondable_positions_for_benzonitrile_with_fluorine_fragment(self):
        bondable = mutable_positions(self.benzonitrile, self.fluorine_fragment, self.type0_fragment)
        self.assertEqual([0, 1, 2, 3, 4, 5], bondable)

    def test_mutable_positions_for_benzonitrile_with_nitrogen_fragment(self):
        bondable = mutable_positions(self.benzonitrile, self.nitrogen_fragment, self.type0_fragment)
        self.assertEqual([0, 1, 2, 3, 4, 5], bondable)

    def test_type0_bondable_positions_for_benzonitrile_with_cyanide_fragment(self):
        bondable = mutable_positions(self.benzonitrile, self.cyanide_fragment, self.type0_fragment)
        self.assertEqual([0, 1, 2, 4, 5], bondable)

    # Napthalene
    def test_type0_bondable_positions_for_napthalene_with_hydrogen_fragment(self):
        bondable = mutable_positions(self.napthalene, self.hydrogen_fragment, self.type0_fragment)
        self.assertEqual([], bondable)

    def test_type0_bondable_positions_for_napthalene_with_fluorine_fragment(self):
        bondable = mutable_positions(self.napthalene, self.fluorine_fragment, self.type0_fragment)
        self.assertEqual([0, 1, 2, 4, 5, 6, 7, 9], bondable)

    def test_mutable_positions_for_napthalene_with_nitrogen_fragment(self):
        bondable = mutable_positions(self.napthalene, self.nitrogen_fragment, self.type0_fragment)
        self.assertEqual([0, 1, 2, 4, 5, 6, 7, 9], bondable)

    def test_type0_bondable_positions_for_napthalene_with_cyanide_fragment(self):
        bondable = mutable_positions(self.napthalene, self.cyanide_fragment, self.type0_fragment)
        self.assertEqual([0, 1, 2, 4, 5, 6, 7, 9], bondable)

    # Acridine
    def test_type0_bondable_positions_for_acridine_with_hydrogen_fragment(self):
        bondable = mutable_positions(self.acridine, self.hydrogen_fragment, self.type0_fragment)
        self.assertEqual([13], bondable)

    def test_type0_bondable_positions_for_acridine_with_fluorine_fragment(self):
        bondable = mutable_positions(self.acridine, self.fluorine_fragment, self.type0_fragment)
        self.assertEqual([0, 1, 2, 5, 6, 8, 9, 10, 11, 13], bondable)

    def test_type0_bondable_positions_for_acridine_with_cyanide_fragment(self):
        bondable = mutable_positions(self.acridine, self.cyanide_fragment, self.type0_fragment)
        self.assertEqual([0, 1, 2, 5, 6, 8, 9, 10, 11, 13], bondable)

    # Pyrrole
    def test_type0_bondable_positions_for_pyrrole_with_fluorine_fragment(self):
        bondable = mutable_positions(self.pyrrole, self.fluorine_fragment, self.type0_fragment)
        self.assertEqual([0, 1, 2, 4], bondable)

    # Fulvene
    def test_type0_bondable_positions_for_fulvene_with_fluorine_fragment(self):
        bondable = mutable_positions(self.fulvene, self.fluorine_fragment, self.type0_fragment)
        self.assertEqual([2, 3, 4, 5], bondable)

    # Pyridine
    def test_mutable_positions_for_pyridine_with_nitrogen_fragment(self):
        bondable = mutable_positions(self.pyridine, self.nitrogen_fragment, self.type0_fragment)
        self.assertEqual([0, 1, 2, 4, 5], bondable)

    # Benzothiophene
    def test_mutable_positions_for_benzothiophene_with_nitrogen_fragment(self):
        bondable = mutable_positions(self.benzothiophene, self.nitrogen_fragment, self.type0_fragment)
        self.assertEqual([0, 1, 2, 5, 6, 7], bondable)

    # Pyrazine
    def test_mutable_positions_for_pyrazine_with_nitrogen_fragment(self):
        bondable = mutable_positions(self.pyrazine, self.nitrogen_fragment, self.type0_fragment)
        self.assertEqual([0, 1, 3, 4], bondable)

    def test_mutable_positions_for_piazthiole_with_cyanide_fragment(self):
        bondable = mutable_positions(self.piazthiole, self.cyanide_fragment, self.type0_fragment)
        self.assertEqual([0, 1, 2, 5, 6, 8], bondable)

    # Spiro[4.4]nona-1,3,6,8-tetraene
    def test_mutable_positions_for_spiro_4_4_nona_1_3_6_8_tetraene_with_fluorine_fragment(self):
        bondable = mutable_positions(self.spiro_4_4_nona_1_3_6_8_tetraene, self.fluorine_fragment, self.type0_fragment)
        self.assertEqual([0, 1, 3, 4, 5, 6, 7, 8], bondable)

    # 1,3-Cyclopentadiene
    def test_mutable_positions_for_cyclopentadiene_with_chlorine_fragment(self):
        bondable = mutable_positions(self.cyclopentadiene, self.chlorine_fragment, self.type0_fragment)
        self.assertEqual([1, 2, 3, 4], bondable)
