from unittest import TestCase
from rdkit import Chem
from cspy.molbuilder.utility import joined_symm_sssr
from cspy.molbuilder.utility import heteroatom_at_equal_dist
from cspy.molbuilder.utility import joined_symm_sssr_connectivities
from cspy.molbuilder.utility import joined_symm_sssr_idx_connections


class TestCagedSSSR(TestCase):

    def setUp(self):
        self.benzene = Chem.MolFromSmiles('c1ccccc1')
        self.napthalene = Chem.MolFromSmiles('c1ccc2ccccc2c1')
        self.pyrene = Chem.MolFromSmiles('c1cc2ccc3cccc4c3c2c(c1)cc4')
        self.tricyclo_6_2_2_0_2_7_dodeca_2_4_6_9_11_pentaene = Chem.MolFromSmiles('c1ccc2c(c1)C([H])3C=CC([H])2C=C3')

    def test_caged_sssr_for_benzene(self):
        cage_ring_atom_idxs_list = joined_symm_sssr(self.benzene)
        self.assertEqual([[0, 1, 2, 3, 4, 5]], cage_ring_atom_idxs_list)

    def test_caged_sssr_for_napthalene(self):
        cage_ring_atom_idxs_list = joined_symm_sssr(self.napthalene)
        self.assertEqual([[0, 1, 2, 3, 8, 9], [3, 4, 5, 6, 7, 8]], cage_ring_atom_idxs_list)

    def test_caged_sssr_for_pyrene(self):
        cage_ring_atom_idxs_list = joined_symm_sssr(self.pyrene)
        self.assertEqual([[0, 1, 2, 11, 12, 13], [2, 3, 4, 5, 10, 11], [5, 6, 7, 8, 9, 10], [9, 10, 11, 12, 14, 15]],
                         cage_ring_atom_idxs_list)

    def test_caged_sssr_for_tricyclo_6_2_2_0_2_7_dodeca_2_4_6_9_11_pentaene(self):
        cage_ring_atom_idxs_list = joined_symm_sssr(self.tricyclo_6_2_2_0_2_7_dodeca_2_4_6_9_11_pentaene)
        self.assertEqual([[0, 1, 2, 3, 4, 5], [3, 4, 6, 7, 8, 9, 10, 11]], cage_ring_atom_idxs_list)


class HeteroatomAtEqualDistances(TestCase):

    def setUp(self):
        self.benzene = Chem.MolFromSmiles('c1ccccc1')
        self.thiophene = Chem.MolFromSmiles('c1ccsc1')
        self.isoindole = Chem.MolFromSmiles('c1ccc2c[nH]cc2c1')
        self.two_benzimidazolinone = Chem.MolFromSmiles('c1ccc2c(c1)[nH]c(=O)[nH]2')
        self.four_azatricyclo_5_2_2_0_2_6_undeca_2_5_8_10_tetraene = Chem.MolFromSmiles('c1c2c(c[nH]1)C3C=CC2C=C3')

        Chem.Kekulize(self.benzene)
        Chem.Kekulize(self.thiophene)
        Chem.Kekulize(self.isoindole)
        Chem.Kekulize(self.two_benzimidazolinone)
        Chem.Kekulize(self.four_azatricyclo_5_2_2_0_2_6_undeca_2_5_8_10_tetraene)

    def test_heteroatom_at_equal_distances_for_benzene(self):
        test = heteroatom_at_equal_dist(self.benzene, 0, 1)
        self.assertEqual(test, False)
        test = heteroatom_at_equal_dist(self.benzene, 1, 2)
        self.assertEqual(test, False)
        test = heteroatom_at_equal_dist(self.benzene, 3, 4)
        self.assertEqual(test, False)
        test = heteroatom_at_equal_dist(self.benzene, 4, 5)
        self.assertEqual(test, False)

    def test_heteroatom_at_equal_distances_for_thiophene(self):
        test = heteroatom_at_equal_dist(self.thiophene, 0, 1)
        self.assertEqual(test, True)
        test = heteroatom_at_equal_dist(self.thiophene, 0, 4)
        self.assertEqual(test, False)
        test = heteroatom_at_equal_dist(self.thiophene, 1, 2)
        self.assertEqual(test, False)

    def test_heteroatom_at_equal_distances_for_isoindole(self):
        test = heteroatom_at_equal_dist(self.isoindole, 0, 1)
        self.assertEqual(test, True)
        test = heteroatom_at_equal_dist(self.isoindole, 0, 8)
        self.assertEqual(test, False)
        test = heteroatom_at_equal_dist(self.isoindole, 1, 2)
        self.assertEqual(test, False)

    def test_heteroatom_at_equal_distances_for_2_benzimidazolinone(self):
        test = heteroatom_at_equal_dist(self.two_benzimidazolinone, 0, 1)
        self.assertEqual(test, False)
        test = heteroatom_at_equal_dist(self.two_benzimidazolinone, 0, 5)
        self.assertEqual(test, False)
        test = heteroatom_at_equal_dist(self.two_benzimidazolinone, 1, 2)
        self.assertEqual(test, False)

    def test_heteroatom_at_equal_distances_for_four_azatricyclo_5_2_2_0_2_6_undeca_2_5_8_10_tetraene(self):
        test = heteroatom_at_equal_dist(self.four_azatricyclo_5_2_2_0_2_6_undeca_2_5_8_10_tetraene, 6, 7)
        self.assertEqual(test, False)
        test = heteroatom_at_equal_dist(self.four_azatricyclo_5_2_2_0_2_6_undeca_2_5_8_10_tetraene, 9, 10)
        self.assertEqual(test, False)


class TestRingConnectivityMatrix(TestCase):

    def setUp(self):
        self.benzene = Chem.MolFromSmiles('c1ccccc1')
        self.napthalene = Chem.MolFromSmiles('c1ccc2ccccc2c1')
        self.anthracene = Chem.MolFromSmiles('c1ccc2cc3ccccc3cc2c1')
        self.pyrene = Chem.MolFromSmiles('c1cc2ccc3cccc4c3c2c(c1)cc4')

    def test_ring_connectivity_matrix_for_benzene(self):
        connectivities_list = joined_symm_sssr_connectivities(self.benzene)
        self.assertEqual([[]], connectivities_list)

    def test_ring_connectivity_matrix_for_napthalene(self):
        connectivities_list = joined_symm_sssr_connectivities(self.napthalene)
        self.assertEqual([[1], [0]], connectivities_list)

    def test_ring_connectivity_matrix_for_anthracene(self):
        connectivities_list = joined_symm_sssr_connectivities(self.anthracene)
        self.assertEqual([[1], [0, 2], [1]], connectivities_list)

    def test_ring_connectivity_matrix_for_pyrene(self):
        connectivities_list = joined_symm_sssr_connectivities(self.pyrene)
        self.assertEqual([[1, 3], [0, 2, 3], [1, 3], [0, 1, 2]], connectivities_list)


class TestRingIdxConnections(TestCase):

    def test_ring_idx_connections_for_benzene(self):
        connections = joined_symm_sssr_idx_connections([[]], 0)
        self.assertEqual([], connections)

    def test_ring_idx_connections_for_napthalene(self):
        connections = joined_symm_sssr_idx_connections([[1], [0]], 0)
        self.assertEqual([1], connections)

    def test_ring_idx_connections_for_anthracene_at_ring_position_0(self):
        connections = joined_symm_sssr_idx_connections([[1], [0, 2], [1]], 0)
        self.assertEqual([1, 2], connections)

    def test_ring_idx_connections_for_anthracene_at_ring_position_1(self):
        connections = joined_symm_sssr_idx_connections([[1], [0, 2], [1]], 1)
        self.assertEqual([0, 2], connections)

    def test_ring_idx_connections_for_anthracene_at_ring_position_2(self):
        connections = joined_symm_sssr_idx_connections([[1], [0, 2], [1]], 2)
        self.assertEqual([0, 1], connections)

    def test_ring_idx_connections_for_pyrene_at_ring_position_0(self):
        connections = joined_symm_sssr_idx_connections([[1, 3], [0, 2, 3], [1, 3], [0, 1, 2]], 0)
        self.assertEqual([1, 2, 3], connections)

    def test_ring_idx_connections_for_pyrene_at_ring_position_1(self):
        connections = joined_symm_sssr_idx_connections([[1, 3], [0, 2, 3], [1, 3], [0, 1, 2]], 1)
        self.assertEqual([0, 2, 3], connections)

    def test_ring_idx_connections_for_pyrene_at_ring_position_2(self):
        connections = joined_symm_sssr_idx_connections([[1, 3], [0, 2, 3], [1, 3], [0, 1, 2]], 2)
        self.assertEqual([0, 1, 3], connections)

    def test_ring_idx_connections_for_pyrene_at_ring_position_3(self):
        connections = joined_symm_sssr_idx_connections([[1, 3], [0, 2, 3], [1, 3], [0, 1, 2]], 3)
        self.assertEqual([0, 1, 2], connections)
