from unittest import TestCase
from rdkit import Chem
from cspy.molbuilder.mutation.substitution import type1_mutation
from cspy.molbuilder.mutation.substitution import mutable_positions


class TestType1Mutation(TestCase):

    def setUp(self):
        self.thiophene = Chem.MolFromSmiles('c1ccsc1')
        self.fulvene = Chem.MolFromSmiles('C=C1C=CC=C1')
        self.indole = Chem.MolFromSmiles('c1ccc2c(c1)cc[nH]2')
        self.methylene_2h_indene = Chem.MolFromSmiles('C=C1C=c2ccccc2=C1')
        self.dichlorothiophene = Chem.MolFromSmiles('c1cc(sc1Cl)Cl')
        self.dichloro_5_methylene_1_3_cyclopentadiene = Chem.MolFromSmiles('C=C1C(=CC=C1Cl)Cl')

        self.nitrogen_fragment = Chem.MolFromSmarts('[#7R1r5&H]')
        self.oxygen_fragment = Chem.MolFromSmarts('[#8R1r5]')
        self.sulfur_fragment = Chem.MolFromSmarts('[#16R1r5]')
        self.ethene_fragment = Chem.MolFromSmarts('[#6R1r5]=[#6&H2]')

    # Thiophene
    def test_type1_mutator_with_thiophene_with_nitrogen_fragment_to_create_pyrrole(self):
        pyrrole = type1_mutation(self.thiophene, self.nitrogen_fragment, 3)
        self.assertEqual(Chem.MolToInchi(pyrrole, options='-SNon'), 'InChI=1S/C4H5N/c1-2-4-5-3-1/h1-5H')

    def test_type1_mutator_with_thiophene_with_oxygen_fragment_to_create_furan(self):
        furan = type1_mutation(self.thiophene, self.oxygen_fragment, 3)
        self.assertEqual(Chem.MolToInchi(furan, options='-SNon'), 'InChI=1S/C4H4O/c1-2-4-5-3-1/h1-4H')

    def test_type1_mutator_with_thiophene_with_ethene_fragment_to_create_fulvene(self):
        fulvene = type1_mutation(self.thiophene, self.ethene_fragment, 3)
        self.assertEqual(Chem.MolToInchi(fulvene, options='-SNon'), 'InChI=1S/C6H6/c1-6-4-2-3-5-6/h2-5H,1H2')

    # Fulvene
    def test_type1_mutator_with_fulvene_with_nitrogen_fragment_to_create_pyrrole(self):
        pyrrole = type1_mutation(self.fulvene, self.nitrogen_fragment, 1)
        self.assertEqual(Chem.MolToInchi(pyrrole, options='-SNon'), 'InChI=1S/C4H5N/c1-2-4-5-3-1/h1-5H')

    def test_type1_mutator_with_fulvene_with_oxygen_fragment_to_create_furan(self):
        furan = type1_mutation(self.fulvene, self.oxygen_fragment, 1)
        self.assertEqual(Chem.MolToInchi(furan, options='-SNon'), 'InChI=1S/C4H4O/c1-2-4-5-3-1/h1-4H')

    def test_type1_mutator_with_fulvene_with_sulfur_fragment_to_create_thiophene(self):
        thiophene = type1_mutation(self.fulvene, self.sulfur_fragment, 1)
        self.assertEqual(Chem.MolToInchi(thiophene, options='-SNon'), 'InChI=1S/C4H4S/c1-2-4-5-3-1/h1-4H')

    # Indole
    def test_type1_mutator_with_indole_with_nitrogen_fragment_to_create_benzothiophene(self):
        benzothiophene = type1_mutation(self.indole, self.sulfur_fragment, 8)
        self.assertEqual(Chem.MolToInchi(benzothiophene, options='-SNon'), 'InChI=1S/C8H6S/c1-2-4-8-7(3-1)5-6-9-8/h1-6H')

    # 2-Methylene-2H-indene
    def test_type1_mutator_with_methylene_2h_indene_with_nitrogen_fragment_to_create_2h_isoindole(self):
        isoindole = type1_mutation(self.methylene_2h_indene, self.nitrogen_fragment, 1)
        self.assertEqual(Chem.MolToInchi(isoindole, options='-SNon'), 'InChI=1S/C8H7N/c1-2-4-8-6-9-5-7(8)3-1/h1-6,9H')

    def test_type1_mutator_with_methylene_2h_indene_with_sulfur_fragment_to_create_2_benzothiophene(self):
        benzothiophene = type1_mutation(self.methylene_2h_indene, self.sulfur_fragment, 1)
        self.assertEqual(Chem.MolToInchi(benzothiophene, options='-SNon'), 'InChI=1S/C8H6S/c1-2-4-8-6-9-5-7(8)3-1/h1-6H')

    # 2,5-Dichlorothiophene
    def test_type1_mutator_with_dichlorothiophene_with_oxygen_fragment_to_create_dichlorofuran(self):
        dichlorofuran = type1_mutation(self.dichlorothiophene, self.oxygen_fragment, 3)
        self.assertEqual(Chem.MolToInchi(dichlorofuran, options='-SNon'), 'InChI=1S/C4H2Cl2O/c5-3-1-2-4(6)7-3/h1-2H')

    # 1,4-Dichloro-5-methylene-1,3-cyclopentadiene
    def test_type1_mutator_with_dichloro_5_methylene_1_3_cyclopentadiene_with_nitrogen_fragment_to_create_dichlorofuran(self):
        dichlorofuran = type1_mutation(self.dichloro_5_methylene_1_3_cyclopentadiene, self.oxygen_fragment, 1)
        self.assertEqual(Chem.MolToInchi(dichlorofuran, options='-SNon'), 'InChI=1S/C4H2Cl2O/c5-3-1-2-4(6)7-3/h1-2H')


class TestType1MutablePositions(TestCase):

    def setUp(self):
        self.thiophene = Chem.MolFromSmiles('c1ccsc1')
        self.indole = Chem.MolFromSmiles('c1ccc2c(c1)cc[nH]2')
        self.methyleneindene = Chem.MolFromSmiles('C=C1C=Cc2c1cccc2')
        self.benzothiophene_2_carbonitrile = Chem.MolFromSmiles('c1ccc2c(c1)cc(s2)C#N')
        self.methylene_2h_indene = Chem.MolFromSmiles('C=C1C=c2ccccc2=C1')
        self.two_benzimidazolinone = Chem.MolFromSmiles('c1ccc2c(c1)[nH]c(=O)[nH]2')
        self.spiro_4_4_nona_1_3_6_8_tetraene = Chem.MolFromSmiles('C1=CC2(C=C1)C=CC=C2')

        self.nitrogen_fragment = Chem.MolFromSmarts('[#7R1r5&H]')
        self.oxygen_fragment = Chem.MolFromSmarts('[#8R1r5]')
        self.sulfur_fragment = Chem.MolFromSmarts('[#16R1r5]')
        self.ethene_fragment = Chem.MolFromSmarts('[#6R1r5]=[#6&H2]')
        self.carbon_monoxide_fragment = Chem.MolFromSmarts('[#6R1r5]=[#8&H0]')

        type1_fragment_smarts = ['[#7R1r5&H]', '[#8R1r5]', '[#16R1r5]', '[#6R1r5]=[#6&H2]', '[#6R1r5]=[#8&H0]']
        self.type1_fragment = [Chem.MolFromSmarts(i) for i in type1_fragment_smarts]

    # Thiophene
    def test_mutable_positions_for_thiophene_with_oxygen_fragment(self):
        bondable = mutable_positions(self.thiophene, self.oxygen_fragment, self.type1_fragment)
        self.assertEqual([3], bondable)

    # Indole
    def test_mutable_positions_for_indole_with_oxygen_fragment(self):
        bondable = mutable_positions(self.indole, self.oxygen_fragment, self.type1_fragment)
        self.assertEqual([8], bondable)

    # 1-methyleneindene
    def test_mutable_positions_for_methyleneindene_with_oxygen_fragment(self):
        bondable = mutable_positions(self.methyleneindene, self.oxygen_fragment, self.type1_fragment)
        self.assertEqual([1], bondable)

    # 1-Benzothiophene-2-carbonitrile
    def test_mutable_positions_for_benzothiophene_2_carbonitrile_with_oxygen_fragment(self):
        bondable = mutable_positions(self.benzothiophene_2_carbonitrile, self.oxygen_fragment, self.type1_fragment)
        self.assertEqual([8], bondable)

    def test_mutable_positions_for_benzothiophene_2_carbonitrile_with_sulfur_fragment(self):
        bondable = mutable_positions(self.benzothiophene_2_carbonitrile, self.sulfur_fragment, self.type1_fragment)
        self.assertEqual([], bondable)

    # 1-methyleneindene
    def test_mutable_positions_for_methyleneindene_with_ethene_fragment(self):
        bondable = mutable_positions(self.methyleneindene, self.ethene_fragment, self.type1_fragment)
        self.assertEqual([], bondable)

    # 2-Methylene-2H-indene
    def test_mutable_positions_for_methylene_2h_indene_with_ethene_fragment(self):
        bondable = mutable_positions(self.methylene_2h_indene, self.ethene_fragment, self.type1_fragment)
        self.assertEqual([], bondable)

    def test_mutable_positions_for_methylene_2h_indene_with_sulfur_fragment(self):
        bondable = mutable_positions(self.methylene_2h_indene, self.sulfur_fragment, self.type1_fragment)
        self.assertEqual([1], bondable)

    # 1,3-Dihydro-2H-benzimidazol-2-one
    def test_mutable_positions_for_two_benzimidazolinone_with_a_nitrogen_fragment(self):
        bondable = mutable_positions(self.two_benzimidazolinone, self.nitrogen_fragment, self.type1_fragment)
        self.assertEqual([7], bondable)

    def test_mutable_positions_for_two_benzimidazolinone_with_a_carbon_monoxide_fragment(self):
        bondable = mutable_positions(self.two_benzimidazolinone, self.carbon_monoxide_fragment, self.type1_fragment)
        self.assertEqual([6, 9], bondable)

    # Spiro[4.4]nona-1,3,6,8-tetraene
    def test_mutable_positions_for_spiro_4_4_nona_1_3_6_8_tetraene_with_nitrogen_fragment(self):
        bondable = mutable_positions(self.spiro_4_4_nona_1_3_6_8_tetraene, self.nitrogen_fragment, self.type1_fragment)
        self.assertEqual([], bondable)
