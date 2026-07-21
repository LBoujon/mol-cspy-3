from unittest import TestCase
import numpy as np
from cspy.chem.molecule import Molecule
from cspy.chem.element import Element
from copy import deepcopy


class TestMolecule(TestCase):

    def setUp(self):
        # h = MagicMock(symbol="H", atomic_number=1, mass=1.00794, cov=0.23)
        # c = MagicMock(symbol="C", atomic_number=6, mass=12.0107, cov=0.68)
        # n = MagicMock(symbol="N", atomic_number=7, mass=14.0067, cov=0.68)
        # o = MagicMock(symbol="O", atomic_number=8, mass=15.9994, cov=0.68)
        o = Element.from_atomic_number(8)
        n = Element.from_atomic_number(7)
        c = Element.from_atomic_number(6)
        h = Element.from_atomic_number(1)
        f = Element.from_atomic_number(9)
        cl = Element.from_atomic_number(17)
        br = Element.from_atomic_number(35)

        self.oxygen_1 = Molecule(
            positions=np.array([[0.0000000000, 0.0000000000, 0.0000000000]]),
            elements=[o]
        )

        self.nitrogen_1 = Molecule(
            positions=np.array([
                [0.0000000000, 0.0000000000, 1.0399092291],
                [0.0000000000, 0.0000000000, -1.0399092291]
            ]),
            elements=[n, n]
        )
        # nitrogen_1 moved +1 in the z direction
        self.nitrogen_2 = Molecule(
            positions=np.array([
                [0.0000000000, 0.0000000000, 2.0399092291],
                [0.0000000000, 0.0000000000, -0.0399092291]
            ]),
            elements=[n, n]
        )
        # nitrogen_1 rotated
        self.nitrogen_3 = Molecule(
            positions=np.array([
                [0.0000000000, 1.0399092291, 0.0000000000],
                [0.0000000000, -1.0399092291, 0.0000000000]
            ]),
            elements=[n, n]
        )
        # nitrogen_1 rotated and then translated
        self.nitrogen_4 = Molecule(
            positions=np.array([
                [1.0000000000, 2.0399092291, 0.5000000000],
                [1.0000000000, -0.0399092291, 0.5000000000]
            ]),
            elements=[n, n]
        )

        self.h2o_1 = Molecule(
            positions=np.array([
                [0.0000000000, 0.0000000000, -0.1363928482],
                [0.0000000000, 1.4236595095, 0.9813433754],
                [0.0000000000, -1.4236595095, 0.9813433754]
            ]),
            elements=[o, h, h]
        )
        # h2o_1 moved +1 in the x direction
        self.h2o_2 = Molecule(
            positions=np.array([
                [1.0000000000, 0.0000000000, -0.1363928482],
                [1.0000000000, 1.4236595095, 0.9813433754],
                [1.0000000000, -1.4236595095, 0.9813433754]
            ]),
            elements=[o, h, h]
        )
        # h2o_1 with reordering
        self.h2o_3 = Molecule(
            positions=np.array([
                [0.0000000000, 1.4236595095, 0.9813433754],
                [0.0000000000, 0.0000000000, -0.1363928482],
                [0.0000000000, -1.4236595095, 0.9813433754]
            ]),
            elements=[h, o, h]
        )
        # h2o_1 with reordering and translated +1 in the y direction
        # and +1 in the x direction
        self.h2o_4 = Molecule(
            positions=np.array([
                [1.0000000000, 2.4236595095, 0.9813433754],
                [1.0000000000, -0.4236595095, 0.9813433754],
                [1.0000000000, 1.0000000000, -0.1363928482]
            ]),
            elements=[h, h, o]
        )

        self.c2h4_1 = Molecule(
            positions=np.array([
                [0.0000000000, 1.2594652672, 0.0000000000],
                [0.0000000000, 1.2594652672, 0.0000000000],
                [1.7400646600, 2.3216269636, 0.0000000000],
                [-1.7400646600, 2.3216269636, 0.0000000000],
                [1.7400646600, -2.3216269636, 0.0000000000],
                [-1.7400646600, -2.3216269636, 0.000000000]
            ]),
            elements=[c, c, h, h, h, h]
        )
        # c2h4_1 with hydrogens swtiched
        self.c2h4_2 = Molecule(
            positions=np.array([
                [0.0000000000, 1.2594652672, 0.0000000000],
                [0.0000000000, 1.2594652672, 0.0000000000],
                [1.7400646600, 2.3216269636, 0.0000000000],
                [-1.7400646600, 2.3216269636, 0.0000000000],
                [-1.7400646600, -2.3216269636, 0.000000000],
                [1.7400646600, -2.3216269636, 0.0000000000]
            ]),
            elements=[c, c, h, h, h, h]
        )
        # c2h4_1 with hydrogens reordered
        self.c2h4_3 = Molecule(
            positions=np.array([
                [0.0000000000, 1.2594652672, 0.0000000000],
                [0.0000000000, 1.2594652672, 0.0000000000],
                [-1.7400646600, 2.3216269636, 0.0000000000],
                [-1.7400646600, -2.3216269636, 0.000000000],
                [1.7400646600, 2.3216269636, 0.0000000000],
                [1.7400646600, -2.3216269636, 0.0000000000]
            ]),
            elements=[c, c, h, h, h, h]
        )

        self.aceta_1 = Molecule(
            positions=np.array([
                [2.19748, 1.16892, 0.98592],
                [1.18991, 1.53866, 2.00819],
                [2.42242, 0.20859, -0.66920],
                [1.70368, 2.08590, 2.83258],
                [0.43923, 2.20860, 1.60955],
                [0.70543, 0.68712, 2.42875],
                [1.71300, 0.43968, 0.00000],
                [3.36610, 1.51453, 1.02054]
            ]),
            elements=[c, c, h, h, h, h, o, o]
        )
        # aceta_1 with carbons switched
        self.aceta_2 = Molecule(
            positions=np.array([
                [1.18991, 1.53866, 2.00819],
                [2.19748, 1.16892, 0.98592],
                [2.42242, 0.20859, -0.66920],
                [1.70368, 2.08590, 2.83258],
                [0.43923, 2.20860, 1.60955],
                [0.70543, 0.68712, 2.42875],
                [1.71300, 0.43968, 0.00000],
                [3.36610, 1.51453, 1.02054]
            ]),
            elements=[c, c, h, h, h, h, o, o]
        )
        # aceta_1 with a carbon switched with hydrogen and oxygen
        # switched with hydrogen and translated +1 in x and +1 in y
        self.aceta_3 = Molecule(
            positions=np.array([
                [3.19748, 2.16892, 0.98592],
                [3.42242, 1.20859, -0.66920],
                [2.18991, 2.53866, 2.00819],
                [2.70368, 3.08590, 2.83258],
                [1.43923, 3.20860, 1.60955],
                [2.71300, 1.43968, 0.00000],
                [1.70543, 1.68712, 2.42875],
                [4.36610, 2.51453, 1.02054]
            ]),
            elements=[c, h, c, h, h, o, h, o]
        )
        # aceta_1 with reordering
        self.aceta_4 = Molecule(
            positions=np.array([
                [2.19748, 1.16892, 0.98592],
                [1.18991, 1.53866, 2.00819],
                [1.71300, 0.43968, 0.00000],
                [3.36610, 1.51453, 1.02054],
                [2.42242, 0.20859, -0.66920],
                [1.70368, 2.08590, 2.83258],
                [0.43923, 2.20860, 1.60955],
                [0.70543, 0.68712, 2.42875]
            ]),
            elements=[c, c, o, o, h, h, h, h]
        )
        # aceta_1 with reordering
        self.aceta_5 = Molecule(
            positions=np.array([
                [2.42242, 0.20859, -0.66920],
                [1.70368, 2.08590, 2.83258],
                [2.19748, 1.16892, 0.98592],
                [1.18991, 1.53866, 2.00819],
                [1.71300, 0.43968, 0.00000],
                [3.36610, 1.51453, 1.02054],
                [0.43923, 2.20860, 1.60955],
                [0.70543, 0.68712, 2.42875]
            ]),
            elements=[h, h, c, c, o, o, h, h]
        )
        # aceta_1 rotated
        self.aceta_6 = Molecule(
            positions=np.array([
                [1.16892, -2.19748, 0.98592],
                [1.53866, -1.18991, 2.00819],
                [0.20859, -2.42242, -0.66920],
                [2.08590, -1.70368, 2.83258],
                [2.20860, -0.43923, 1.60955],
                [0.68712, -0.70543, 2.42875],
                [0.43968, -1.71300, 0.00000],
                [1.51453, -3.36610, 1.02054]
            ]),
            elements=[c, c, h, h, h, h, o, o]
        )
        # aceta_5 rotated
        self.aceta_7 = Molecule(
            positions=np.array([
                [0.20859, -2.42242, -0.66920],
                [2.08590, -1.70368, 2.83258],
                [1.16892, -2.19748, 0.98592],
                [1.53866, -1.18991, 2.00819],
                [0.43968, -1.71300, 0.00000],
                [1.51453, -3.36610, 1.02054],
                [2.20860, -0.43923, 1.60955],
                [0.68712, -0.70543, 2.42875]
            ]),
            elements=[h, h, c, c, o, o, h, h]
        )

        self.galgax_1 = Molecule(
            positions=np.array([
                [-2.5685, 0.9087, 1.7825],
                [-3.595, 1.536, 2.4894],
                [-3.5316, 2.9067, 2.749],
                [-2.4354, 3.6513, 2.3015],
                [-1.4117, 3.026, 1.5979],
                [-1.465, 1.6479, 1.3283],
                [-0.3485, 1.0548, 0.5178],
                [0.5542, 1.7753, 0.1078],
                [-0.3263, -0.4475, 0.2216],
                [0.4799, -0.8167, -1.0363],
                [-0.2662, -0.6415, -2.3745],
                [-0.7523, 0.7874, -2.6723],
                [-2.0737, 1.1408, -1.9842],
                [-2.8553, 0.2544, -1.6598],
                [-2.4233, 2.5768, -1.7193],
                [-1.5595, 3.6472, -2.0006],
                [-1.9292, 4.9538, -1.6801],
                [-3.1628, 5.2055, -1.0759],
                [-4.031, 4.1458, -0.7944],
                [-3.6633, 2.8434, -1.1144],
                [-2.6431, -0.1537, 1.5794],
                [-4.4463, 0.9544, 2.8322],
                [-4.3355, 3.3942, 3.2949],
                [-2.387, 4.7197, 2.4934],
                [-0.5632, 3.5891, 1.2255],
                [-1.3399, -0.8501, 0.1523],
                [0.1325, -0.9153, 1.107],
                [0.7819, -1.8687, -0.9558],
                [1.4005, -0.2219, -1.0449],
                [-1.1297, -1.3159, -2.4098],
                [0.4119, -0.9515, -3.1796],
                [-0.9407, 0.9005, -3.7514],
                [0.0261, 1.5111, -2.4182],
                [-0.5921, 3.4711, -2.4578],
                [-1.2512, 5.7745, -1.8987],
                [-3.4454, 6.2242, -0.8222],
                [-4.988, 4.3367, -0.3167],
                [-4.3122, 2.0055, -0.885]
            ]),
            elements=[
                c, c, c, c, c, c, c, o, c, c, c, c, c, o, c, c, c, c, c,
                c, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h
            ]
        )
        self.galgax_2 = Molecule(
            positions=np.array([
                [1.34971573, -1.15591516, -1.07277486],
                [2.60379291, -1.57796137, -0.68013403],
                [2.72865269, -2.52571323, 0.32242538],
                [1.59456025, -3.0524937, 0.91966828],
                [0.33689932, -2.63094269, 0.5295577],
                [0.20044465, -1.67010719, -0.47034031],
                [-1.10747466, -1.13956402, -0.93355468],
                [-1.17207268, -0.42299386, -1.91130904],
                [-2.3399671, -1.54209389, -0.15090264],
                [-2.57268719, -2.56806688, -0.45336585],
                [-2.11765816, -1.54414781, 0.91561269],
                [-3.53955254, -0.63450035, -0.42878884],
                [-4.46017189, -1.18603465, -0.22700421],
                [-3.53459374, -0.36036833, -1.48626514],
                [-3.53995055, 0.63249971, 0.42823006],
                [-4.46078021, 1.18360344, 0.22621749],
                [-3.53512384, 0.35836204, 1.48570809],
                [-2.34076013, 1.54071927, 0.15071462],
                [-2.57414598, 2.56648825, 0.45336747],
                [-2.11820513, 1.54313778, -0.91576368],
                [-1.10818392, 1.13873345, 0.93354413],
                [-1.17245042, 0.42195298, 1.91117417],
                [0.19943708, 1.67009154, 0.47043636],
                [0.33512256, 2.63319052, -0.52739497],
                [1.59247831, 3.05573702, -0.91741273],
                [2.72702003, 2.52761531, -0.32221474],
                [2.60292327, 1.57753741, 0.67824462],
                [1.34913803, 1.15459213, 1.07090191],
                [1.2327052, 0.4112547, 1.84573114],
                [3.48313667, 1.16493603, 1.14861889],
                [3.70564772, 2.85923867, -0.63719606],
                [1.68943518, 3.80127741, -1.6930498],
                [-0.53364476, 3.06308834, -1.00009103],
                [-0.5314667, -3.05977387, 1.00394419],
                [1.6921028, -3.79617914, 1.69700921],
                [3.70752, -2.85656321, 0.63746161],
                [3.48364694, -1.16643448, -1.15211652],
                [1.23267917, -0.41427396, -1.84912368]
            ]),
            elements=[
                c, c, c, c, c, c, c, o, c, h, h, c, h, h, c, h, h, c, h,
                h, c, o, c, c, c, c, c, c, h, h, h, h, h, h, h, h, h, h
            ]
        )
        self.galgax_3 = Molecule(
            positions=np.array([
                [-2.371,  0.5366,  2.3009],
                [-3.412,  0.9914,  3.1346],
                [-3.6158,  2.3712,  3.3223],
                [-2.7748,  3.298,  2.6793],
                [-1.7327,  2.8443,  1.8477],
                [-1.5249,  1.4608,  1.6417],
                [-0.4156,  1.022,  0.7476],
                [0.4541,  1.792,  0.3368],
                [-0.4068, -0.4459,  0.2881],
                [0.4205, -0.7357, -0.988],
                [-0.3224, -0.557, -2.3331],
                [-0.7481,  0.889, -2.6863],
                [-2.1148,  1.3264, -2.1324],
                [-2.9237,  0.472, -1.767],
                [-2.3796,  2.7891, -2.0178],
                [-1.5623,  3.7545, -2.6541],
                [-1.8433,  5.1285, -2.5178],
                [-2.9442,  5.5523, -1.7501],
                [-3.7681,  4.6008, -1.1211],
                [-3.4891,  3.2272, -1.2585],
                [-2.2383, -0.5282,  2.181],
                [-4.0563,  0.2798,  3.6307],
                [-4.4158,  2.7187,  3.9601],
                [-2.9281,  4.3582,  2.8203],
                [-1.0985,  3.5677,  1.3531],
                [-1.4299, -0.7906,  0.1387],
                [0.0017, -1.0257,  1.1153],
                [0.7514, -1.7732, -0.9319],
                [1.3425, -0.1521, -0.9946],
                [-1.176, -1.2354, -2.377],
                [0.3445, -0.9015, -3.1239],
                [-0.8087,  0.9866, -3.7699],
                [0.0233,  1.5822, -2.3516],
                [-0.7131,  3.4618, -3.2533],
                [-1.2123,  5.8593, -3.0029],
                [-3.1575,  6.6062, -1.6449],
                [-4.6139,  4.9218, -0.5303],
                [-4.1261,  2.5068, -0.7631]
            ]),
            elements=[
                c, c, c, c, c, c, c, o, c, c, c, c, c, o, c, c, c, c, c,
                c, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h
            ]
        )
        self.galgax_4 = Molecule(
            positions=np.array([
                [1.3497, -1.1559, -1.0728],
                [2.6038, -1.578, -0.6801],
                [2.7287, -2.5257, 0.3224],
                [1.5946, -3.0525, 0.9197],
                [0.3369, -2.6309, 0.5296],
                [0.2004, -1.6701, -0.4703],
                [-1.1075, -1.1396, -0.9336],
                [-1.1721, -0.423, -1.9113],
                [-2.34, -1.5421, -0.1509],
                [-2.5727, -2.5681, -0.4534],
                [-2.1177, -1.5441, 0.9156],
                [-3.5396, -0.6345, -0.4288],
                [-4.4602, -1.186, -0.227],
                [-3.5346, -0.3604, -1.4863],
                [-3.54, 0.6325, 0.4282],
                [-4.4608, 1.1836, 0.2262],
                [-3.5351, 0.3584, 1.4857],
                [-2.3408, 1.5407, 0.1507],
                [-2.5741, 2.5665, 0.4534],
                [-2.1182, 1.5431, -0.9158],
                [-1.1082, 1.1387, 0.9335],
                [-1.1725, 0.422, 1.9112],
                [0.1994, 1.6701, 0.4704],
                [0.3351, 2.6332, -0.5274],
                [1.5925, 3.0557, -0.9174],
                [2.727, 2.5276, -0.3222],
                [2.6029, 1.5775, 0.6782],
                [1.3491, 1.1546, 1.0709],
                [1.2327, 0.4113, 1.8457],
                [3.4831, 1.1649, 1.1486],
                [3.7056, 2.8592, -0.6372],
                [1.6894, 3.8013, -1.693],
                [-0.5336, 3.0631, -1.0001],
                [-0.5315, -3.0598, 1.0039],
                [1.6921, -3.7962, 1.697],
                [3.7075, -2.8566, 0.6375],
                [3.4836, -1.1664, -1.1521],
                [1.2327, -0.4143, -1.8491]
            ]),
            elements=[
                c, c, c, c, c, c, c, o, c, h, h, c, h, h, c, h, h, c, h,
                h, c, o, c, c, c, c, c, c, h, h, h, h, h, h, h, h, h, h
            ]
        )
        self.galgax_5 = Molecule(
            positions=np.array([
                [-1.5081,  3.7787, -4.309],
                [-1.9588,  3.9037, -5.6382],
                [-3.3282,  4.0852, -5.9061],
                [-4.2481,  4.1427, -4.8432],
                [-3.7981,  4.0177, -3.5143],
                [-2.4251,  3.8313, -3.2304],
                [-1.995,  3.7075, -1.8053],
                [-2.7692,  3.8193, -0.8553],
                [-0.5212,  3.416, -1.5001],
                [-0.1087,  1.9419, -1.7133],
                [-0.8949,  0.9071, -0.8831],
                [-0.8191,  1.149,  0.6385],
                [-1.5214,  0.0712,  1.478],
                [-2.0228, -0.9057,  0.9195],
                [-1.548,  0.2612,  2.9583],
                [-0.9662,  1.3913,  3.5841],
                [-1.0118,  1.535,  4.9852],
                [-1.6374,  0.5538,  5.7764],
                [-2.2189, -0.5727,  5.166],
                [-2.1741, -0.717,  3.7655],
                [-0.4507,  3.6417, -4.1424],
                [-1.2525,  3.8605, -6.455],
                [-3.6732,  4.1807, -6.9257],
                [-5.3004,  4.2826, -5.0453],
                [-4.5173,  4.0639, -2.7077],
                [-0.3344,  3.7014, -0.4652],
                [0.1036,  4.0769, -2.0994],
                [-0.2037,  1.6844, -2.7688],
                [0.9528,  1.844, -1.4842],
                [-0.5055, -0.087, -1.1104],
                [-1.9404,  0.8896, -1.1964],
                [-1.2774,  2.1084,  0.8778],
                [0.2231,  1.1974,  0.9528],
                [-0.4798,  2.1643,  3.0082],
                [-0.5669,  2.4009,  5.4544],
                [-1.6727,  0.6652,  6.8506],
                [-2.7011, -1.3273,  5.7707],
                [-2.6262, -1.5875,  3.3096]
            ]),
            elements=[
                c, c, c, c, c, c, c, o, c, c, c, c, c, o, c, c, c, c, c,
                c, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h
            ]
        )
        self.galgax_6 = Molecule(
            positions=np.array([
                [0.286, 2.9001, 3.1666],
                [-0.6384, 3.6777, 3.8392],
                [-1.8826, 3.1619, 4.165],
                [-2.199, 1.8554, 3.829],
                [-1.2743, 1.0723, 3.1668],
                [-0.0253, 1.5874, 2.8181],
                [0.9075, 0.6924, 2.0907],
                [0.8209, -0.5141, 2.1737],
                [1.9403, 1.3431, 1.195],
                [2.5744, 2.0058, 1.7864],
                [2.5712, 0.5527, 0.7849],
                [1.2818, 2.138, 0.0591],
                [2.0572, 2.4243, -0.6556],
                [0.8428, 3.0535, 0.4603],
                [0.1843, 1.356, -0.6618],
                [-0.6576, 1.1942, 0.0148],
                [-0.1866, 1.9525, -1.4982],
                [0.6547, 0.0036, -1.1907],
                [0.8448, -0.6766, -0.3578],
                [1.5803, 0.1163, -1.7613],
                [-0.3864, -0.6365, -2.0837],
                [-1.4636, -0.1206, -2.2869],
                [-0.0188, -1.9294, -2.7212],
                [-0.8665, -2.4229, -3.7139],
                [-0.586, -3.6124, -4.3565],
                [0.5466, -4.3319, -4.0113],
                [1.3917, -3.8561, -3.0216],
                [1.1163, -2.6628, -2.3794],
                [1.7829, -2.3177, -1.6048],
                [2.2711, -4.4205, -2.7479],
                [0.77, -5.2636, -4.5099],
                [-1.2491, -3.9813, -5.125],
                [-1.745, -1.8477, -3.9661],
                [-1.502, 0.0512, 2.8983],
                [-3.1659, 1.4489, 4.0854],
                [-2.6022, 3.777, 4.6847],
                [-0.3877, 4.6924, 4.1119],
                [1.2568, 3.3137, 2.9391]
            ]),
            elements=[
                c, c, c, c, c, c, c, o, c, h, h, c, h, h, c, h, h, c, h,
                h, c, o, c, c, c, c, c, c, h, h, h, h, h, h, h, h, h, h
            ]
        )
        self.galgax_7 = Molecule(
            positions=np.array([
                [-2.5591, 0.8077, 2.4491],
                [-3.5334, 1.5166, 3.1796],
                [-3.2977, 2.8497, 3.564],
                [-2.0826, 3.4728, 3.2248],
                [-1.1089, 2.7654, 2.4933],
                [-1.341, 1.4316, 2.0852],
                [-0.2836, 0.7216, 1.3097],
                [0.8767, 1.1256, 1.2379],
                [-0.6895, -0.5394, 0.5298],
                [0.0671, -0.7297, -0.8044],
                [-0.2477, 0.3348, -1.8746],
                [-1.6845, 0.2443, -2.4255],
                [-2.0528, 1.4504, -3.2951],
                [-1.965, 1.3358, -4.5167],
                [-2.4987, 2.7068, -2.6194],
                [-2.834, 3.8289, -3.4125],
                [-3.2595, 5.0317, -2.8148],
                [-3.3554, 5.1271, -1.4142],
                [-3.0254, 4.0189, -0.6127],
                [-2.5999, 2.8168, -1.2105],
                [-2.7639, -0.2197, 2.1874],
                [-4.4619, 1.0351, 3.4518],
                [-4.0446, 3.3914, 4.1266],
                [-1.8948, 4.4935, 3.5262],
                [-0.179, 3.256, 2.2384],
                [-1.759, -0.5261, 0.3314],
                [-0.5018, -1.3949, 1.1782],
                [1.1413, -0.7307, -0.6101],
                [-0.1551, -1.7195, -1.2042],
                [0.4465, 0.211, -2.7074],
                [-0.0463, 1.3323, -1.4802],
                [-2.4175, 0.1519, -1.6268],
                [-1.7877, -0.6568, -3.0302],
                [-2.7647, 3.7709, -4.4904],
                [-3.5115, 5.8813, -3.4331],
                [-3.6808, 6.0496, -0.9552],
                [-3.097, 4.0888, 0.4637],
                [-2.3536, 1.9922, -0.5613]
            ]),
            elements=[
                c, c, c, c, c, c, c, o, c, c, c, c, c, o, c, c, c, c, c,
                c, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h
            ]
        )
        self.galgax_8 = Molecule(
            positions=np.array([
                [1.6899, 2.0617, 1.4178],
                [1.1563, 3.3117, 1.6731],
                [0.5809, 4.0497, 0.6508],
                [0.5508, 3.5402, -0.6375],
                [1.0904, 2.2958, -0.8975],
                [1.6581, 1.5382, 0.1269],
                [2.2117, 0.2058, -0.2299],
                [2.5051, -0.0622, -1.3732],
                [2.3821, -0.7957, 0.8987],
                [3.3625, -0.6097, 1.3474],
                [1.6267, -0.6263, 1.6651],
                [2.321, -2.2326, 0.3804],
                [2.6117, -2.9252, 1.1724],
                [3.0344, -2.3286, -0.4394],
                [0.9287, -2.591, -0.1412],
                [0.5815, -1.7841, -0.7902],
                [0.992, -3.4942, -0.7501],
                [-0.0925, -2.8304, 0.98],
                [0.0345, -2.092, 1.7714],
                [0.0435, -3.8244, 1.4088],
                [-1.4924, -2.722, 0.4193],
                [-2.2293, -3.6758, 0.3206],
                [-1.8895, -1.3747, -0.0766],
                [-2.6929, -1.312, -1.2141],
                [-3.0753, -0.0921, -1.7382],
                [-2.6809, 1.0819, -1.1173],
                [-1.9083, 1.03, 0.031],
                [-1.5061, -0.1888, 0.5446],
                [-0.9194, -0.2065, 1.4504],
                [-1.6156, 1.9427, 0.5287],
                [-2.982, 2.037, -1.5215],
                [-3.6855, -0.0557, -2.6284],
                [-2.9988, -2.2364, -1.6809],
                [1.0784, 1.8819, -1.8947],
                [0.1078, 4.1147, -1.4373],
                [0.1609, 5.0228, 0.8591],
                [1.1899, 3.7145, 2.6749],
                [2.1521, 1.5104, 2.2227]
            ]),
            elements=[
                c, c, c, c, c, c, c, o, c, h, h, c, h, h, c, h, h, c, h,
                h, c, o, c, c, c, c, c, c, h, h, h, h, h, h, h, h, h, h
            ]
        )
        self.galgax_9 = Molecule(
            positions=np.array([
                [-3.4064, 2.9867, -3.3784],
                [-3.6729, 3.6723, -4.5803],
                [-3.0714, 4.9195, -4.8309],
                [-2.2037, 5.4828, -3.8773],
                [-1.937, 4.7977, -2.6759],
                [-2.5296, 3.5406, -2.4131],
                [-2.2219, 2.8563, -1.1211],
                [-1.5657, 3.3821, -0.2239],
                [-2.7372, 1.4287, -0.9009],
                [-1.9637, 0.3463, -1.6876],
                [-0.4794, 0.15, -1.3063],
                [-0.2433, -0.4582, 0.0933],
                [-0.0447, 0.5935, 1.1932],
                [0.9509, 1.3141, 1.1419],
                [-1.09, 0.7224, 2.2469],
                [-1.8575, -0.3814, 2.6906],
                [-2.8262, -0.2155, 3.7006],
                [-3.0299, 1.0493, 4.2836],
                [-2.2577, 2.148, 3.8629],
                [-1.2877, 1.9821, 2.8556],
                [-3.8888, 2.0347, -3.2188],
                [-4.3418, 3.2416, -5.3115],
                [-3.2766, 5.4451, -5.7524],
                [-1.7418, 6.4414, -4.0652],
                [-1.2685, 5.2435, -1.9515],
                [-2.7031, 1.2145, 0.1658],
                [-3.794, 1.3996, -1.1621],
                [-2.4864, -0.6044, -1.578],
                [-2.0124, 0.5778, -2.7522],
                [0.0727, 1.0854, -1.4193],
                [-0.0344, -0.5265, -2.0367],
                [0.6751, -1.0448, 0.071],
                [-1.0417, -1.1563, 0.3317],
                [-1.7075, -1.3689, 2.282],
                [-3.4092, -1.0617, 4.035],
                [-3.7717, 1.176, 5.059],
                [-2.4061, 3.1197, 4.3118],
                [-0.7012, 2.8331, 2.5341]
            ]),
            elements=[
                c, c, c, c, c, c, c, o, c, c, c, c, c, o, c, c, c, c, c,
                c, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h, h
            ]
        )
        self.galgax_10 = Molecule(
            positions=np.array([
                [-0.6833, -3.7258, 0.5053],
                [-0.0806, -4.7642, 1.1907],
                [1.2663, -4.7, 1.5119],
                [2.0174, -3.5997, 1.1319],
                [1.4256, -2.568, 0.4283],
                [0.0672, -2.6171, 0.1188],
                [-0.518, -1.479, -0.6294],
                [0.1295, -0.84, -1.4275],
                [-1.9796, -1.1583, -0.3581],
                [-2.5495, -1.6855, -1.129],
                [-2.2897, -1.5412, 0.614],
                [-2.2512, 0.3442, -0.4652],
                [-3.267, 0.5122, -0.8288],
                [-1.5607, 0.7649, -1.1997],
                [-2.0998, 1.0763, 0.8684],
                [-2.2727, 2.1394, 0.6916],
                [-2.8746, 0.7261, 1.5564],
                [-0.7445, 0.8793, 1.5506],
                [-0.6176, -0.1577, 1.8709],
                [-0.7108, 1.4997, 2.4522],
                [0.4756, 1.2458, 0.7265],
                [1.4957, 0.6011, 0.8002],
                [0.3671, 2.4404, -0.1483],
                [0.8563, 2.3527, -1.4491],
                [0.7726, 3.4412, -2.2972],
                [0.226, 4.6317, -1.8465],
                [-0.2379, 4.7321, -0.5448],
                [-0.1761, 3.6404, 0.302],
                [-0.5193, 3.732, 1.3225],
                [-0.6508, 5.6649, -0.1887],
                [0.1662, 5.4847, -2.5066],
                [1.1399, 3.3637, -3.3097],
                [1.2716, 1.4142, -1.7833],
                [1.9903, -1.6968, 0.1339],
                [3.0667, -3.5482, 1.3812],
                [1.7286, -5.5105, 2.0561],
                [-0.6635, -5.6273, 1.4779],
                [-1.7299, -3.7934, 0.2468]
            ]),
            elements=[
                c, c, c, c, c, c, c, o, c, h, h, c, h, h, c, h, h, c, h,
                h, c, o, c, c, c, c, c, c, h, h, h, h, h, h, h, h, h, h
            ]
        )
        self.c1h1f1cl1br1 = Molecule(
            positions=np.array([
                [0.00000000, 0.00000000, 0.00000000],
                [1.01905009, -0.26757550, -0.27935689],
                [0.02762499, 0.81542046, 1.07555861],
                [-0.78300165, 0.83249316, -1.35157817],
                [-0.99521511, -1.60800424, 0.43297720]
            ]),
            elements=[
                c, h, f, cl, br
            ]
        )

    def test_overlay_of_nitrogen_1_with_nitrogen_1_returns_overlayed_with_small_rmsd(self):
        overlayed, order, rmsd = self.nitrogen_1.overlay(self.nitrogen_1, reorder_atoms_to="self")
        np.testing.assert_allclose(
            overlayed.positions, self.nitrogen_1.positions, atol=1e-5
        )
        np.testing.assert_equal(
            overlayed.atomic_numbers, self.nitrogen_1.atomic_numbers
        )
        self.assertAlmostEqual(rmsd, 0.0, 5)

    def test_overlay_of_nitrogen_1_with_nitrogen_2_returns_overlayed_with_small_rmsd(self):
        overlayed, order, rmsd = self.nitrogen_1.overlay(self.nitrogen_2, reorder_atoms_to="self")
        np.testing.assert_allclose(
            overlayed.positions, self.nitrogen_1.positions, atol=1e-5
        )
        np.testing.assert_equal(
            overlayed.atomic_numbers, self.nitrogen_1.atomic_numbers
        )
        self.assertAlmostEqual(rmsd, 0.0, 5)

    def test_overlay_of_nitrogen_1_with_nitrogen_3_returns_overlayed_with_small_rmsd(self):
        overlayed, order, rmsd = self.nitrogen_1.overlay(self.nitrogen_3, reorder_atoms_to="self")
        np.testing.assert_allclose(
            overlayed.positions, self.nitrogen_1.positions, atol=1e-5
        )
        np.testing.assert_equal(
            overlayed.atomic_numbers, self.nitrogen_1.atomic_numbers
        )
        self.assertAlmostEqual(rmsd, 0.0, 5)

    def test_overlay_of_nitrogen_1_with_nitrogen_4_returns_overlayed_with_small_rmsd(self):
        overlayed, order, rmsd = self.nitrogen_1.overlay(self.nitrogen_4, reorder_atoms_to="self")
        np.testing.assert_allclose(
            overlayed.positions, self.nitrogen_1.positions, atol=1e-5
        )
        np.testing.assert_equal(
            overlayed.atomic_numbers, self.nitrogen_1.atomic_numbers
        )
        self.assertAlmostEqual(rmsd, 0.0, 5)

    def test_overlay_of_h2o_1_with_h2o_1_returns_overlayed_with_small_rmsd(self):
        overlayed, order, rmsd = self.h2o_1.overlay(self.h2o_1, reorder_atoms_to="self")
        np.testing.assert_allclose(
            overlayed.positions, self.h2o_1.positions, atol=1e-5
        )
        np.testing.assert_equal(
            overlayed.atomic_numbers, self.h2o_1.atomic_numbers
        )
        self.assertAlmostEqual(rmsd, 0.0, 5)

    def test_overlay_of_h2o_1_with_h2o_2_returns_overlayed_with_small_rmsd(self):
        overlayed, order, rmsd = self.h2o_1.overlay(self.h2o_2, reorder_atoms_to="self")
        np.testing.assert_allclose(
            overlayed.positions, self.h2o_1.positions, atol=1e-5
        )
        np.testing.assert_equal(
            overlayed.atomic_numbers, self.h2o_1.atomic_numbers
        )
        self.assertAlmostEqual(rmsd, 0.0, 5)

    def test_overlay_of_h2o_1_with_h2o_3_returns_overlayed_with_small_rmsd(self):
        overlayed, order, rmsd = self.h2o_1.overlay(self.h2o_3, reorder_atoms_to="self")
        np.testing.assert_allclose(
            overlayed.positions, self.h2o_1.positions, atol=1e-5
        )
        np.testing.assert_equal(
            overlayed.atomic_numbers, self.h2o_1.atomic_numbers
        )
        self.assertAlmostEqual(rmsd, 0.0, 5)

    def test_overlay_of_h2o_1_with_h2o_4_returns_overlayed_with_small_rmsd(self):
        overlayed, order, rmsd = self.h2o_1.overlay(self.h2o_4, reorder_atoms_to="self")
        np.testing.assert_allclose(
            overlayed.positions, self.h2o_1.positions, atol=1e-5
        )
        np.testing.assert_equal(
            overlayed.atomic_numbers, self.h2o_1.atomic_numbers
        )
        self.assertAlmostEqual(rmsd, 0.0, 5)

    def test_overlay_of_c2h4_1_with_c2h4_1_returns_overlayed_with_small_rmsd(self):
        overlayed, order, rmsd = self.c2h4_1.overlay(self.c2h4_1, reorder_atoms_to="self")
        np.testing.assert_allclose(
            overlayed.positions, self.c2h4_1.positions, atol=1e-5
        )
        np.testing.assert_equal(
            overlayed.atomic_numbers, self.c2h4_1.atomic_numbers
        )
        self.assertAlmostEqual(rmsd, 0.0, 5)

    def test_overlay_of_c2h4_1_with_c2h4_2_returns_overlayed_with_small_rmsd(self):
        overlayed, order, rmsd = self.c2h4_1.overlay(self.c2h4_2, reorder_atoms_to="self")
        np.testing.assert_allclose(
            overlayed.positions, self.c2h4_1.positions, atol=1e-5
        )
        np.testing.assert_equal(
            overlayed.atomic_numbers, self.c2h4_1.atomic_numbers
        )
        self.assertAlmostEqual(rmsd, 0.0, 5)

    def test_overlay_of_c2h4_1_with_c2h4_3_returns_overlayed_with_small_rmsd(self):
        overlayed, order, rmsd = self.c2h4_1.overlay(self.c2h4_3, reorder_atoms_to="self")
        np.testing.assert_allclose(
            overlayed.positions, self.c2h4_1.positions, atol=1e-5
        )
        np.testing.assert_equal(
            overlayed.atomic_numbers, self.c2h4_1.atomic_numbers
        )
        self.assertAlmostEqual(rmsd, 0.0, 5)

    def test_overlay_of_aceta_1_with_aceta_1_returns_overlayed_with_small_rmsd_1(self):
        overlayed, order, rmsd = self.aceta_1.overlay(self.aceta_1, reorder_atoms_to="self")
        np.testing.assert_allclose(
            overlayed.positions, self.aceta_1.positions, atol=1e-5
        )
        np.testing.assert_equal(
            overlayed.atomic_numbers, self.aceta_1.atomic_numbers
        )
        self.assertAlmostEqual(rmsd, 0.0, 5)

    def test_overlay_of_aceta_1_with_aceta_2_returns_overlayed_with_small_rmsd(self):
        overlayed, order, rmsd = self.aceta_1.overlay(self.aceta_2, reorder_atoms_to="self")
        np.testing.assert_allclose(
            overlayed.positions, self.aceta_1.positions, atol=1e-5
        )
        np.testing.assert_equal(
            overlayed.atomic_numbers, self.aceta_1.atomic_numbers
        )
        self.assertAlmostEqual(rmsd, 0.0, 5)

    def test_overlay_of_aceta_1_with_aceta_3_returns_overlayed_with_small_rmsd(self):
        overlayed, order, rmsd = self.aceta_1.overlay(self.aceta_3, reorder_atoms_to="self")
        np.testing.assert_allclose(
            overlayed.positions, self.aceta_1.positions, atol=1e-5
        )
        np.testing.assert_equal(
            overlayed.atomic_numbers, self.aceta_1.atomic_numbers
        )
        self.assertAlmostEqual(rmsd, 0.0, 5)

    def test_overlay_of_aceta_1_with_aceta_4_returns_overlayed_with_small_rmsd(self):
        overlayed, order, rmsd = self.aceta_1.overlay(self.aceta_4, reorder_atoms_to="self")
        np.testing.assert_allclose(
            overlayed.positions, self.aceta_1.positions, atol=1e-5
        )
        np.testing.assert_equal(
            overlayed.atomic_numbers, self.aceta_1.atomic_numbers
        )
        self.assertAlmostEqual(rmsd, 0.0, 5)

    def test_overlay_of_aceta_1_with_aceta_5_returns_overlayed_with_small_rmsd(self):
        overlayed, order, rmsd = self.aceta_1.overlay(self.aceta_5, reorder_atoms_to="self")
        np.testing.assert_allclose(
            overlayed.positions, self.aceta_1.positions, atol=1e-5
        )
        np.testing.assert_equal(
            overlayed.atomic_numbers, self.aceta_1.atomic_numbers
        )
        self.assertAlmostEqual(rmsd, 0.0, 5)

    def test_overlay_of_aceta_1_with_aceta_6_returns_overlayed_with_small_rmsd(self):
        overlayed, order, rmsd = self.aceta_1.overlay(self.aceta_6, reorder_atoms_to="self")
        np.testing.assert_allclose(
            overlayed.positions, self.aceta_1.positions, atol=1e-5
        )
        np.testing.assert_equal(
            overlayed.atomic_numbers, self.aceta_1.atomic_numbers
        )
        self.assertAlmostEqual(rmsd, 0.0, 5)

    def test_overlay_of_aceta_1_with_aceta_7_returns_overlayed_with_small_rmsd(self):
        overlayed, order, rmsd = self.aceta_1.overlay(self.aceta_7, reorder_atoms_to="self")
        np.testing.assert_allclose(
            overlayed.positions, self.aceta_1.positions, atol=1e-5
        )
        np.testing.assert_equal(
            overlayed.atomic_numbers, self.aceta_1.atomic_numbers
        )
        self.assertAlmostEqual(rmsd, 0.0, 5)

    def test_overlay_of_galgax_1_with_galgax_2_returns_correct_ordering(self):
        overlayed, order, rmsd = self.galgax_1.overlay(self.galgax_2, reorder_atoms_to="self")
        np.testing.assert_equal(
            overlayed.atomic_numbers, self.galgax_1.atomic_numbers
        )
        # compare against reordering obtained by hand
        np.testing.assert_equal(
            order, np.array([
                4, 3, 2, 1, 0, 5, 6, 7, 8, 11, 14, 17, 20, 21, 22, 23,
                24, 25, 26, 27, 33, 34, 35, 36, 37, 10, 9, 12, 13, 16,
                15, 18, 19, 32, 31, 30, 29, 28
            ])
        )

    def test_overlay_of_galgax_2_with_galgax_1_returns_correct_ordering(self):
        overlayed, order, rmsd = self.galgax_2.overlay(self.galgax_1, reorder_atoms_to="self")
        np.testing.assert_equal(
            overlayed.atomic_numbers, self.galgax_2.atomic_numbers
        )
        # compare against reordering obtained by hand
        np.testing.assert_equal(
            order, np.array([
                4, 3, 2, 1, 0, 5, 6, 7, 8, 26, 25, 9, 27, 28, 10, 30,
                29, 11, 31, 32, 12, 13, 14, 15, 16, 17, 18, 19, 37, 36,
                35, 34, 33, 20, 21, 22, 23, 24
            ])
        )

    def test_overlay_of_galgax_3_with_galgax_4_returns_correct_ordering(self):
        overlayed, order, rmsd = self.galgax_3.overlay(self.galgax_4, reorder_atoms_to="self")
        np.testing.assert_equal(
            overlayed.atomic_numbers, self.galgax_3.atomic_numbers
        )
        # compare against reordering obtained by hand
        np.testing.assert_equal(
            order, np.array([
                23, 24, 25, 26, 27, 22, 20, 21, 17, 14, 11, 8, 6, 7, 5,
                4, 3, 2, 1, 0, 32, 31, 30, 29, 28, 19, 18, 15, 16, 13,
                12, 9, 10, 33, 34, 35, 36, 37
            ])
        )

    def test_overlay_of_galgax_4_with_galgax_3_returns_correct_ordering(self):
        overlayed, order, rmsd = self.galgax_4.overlay(self.galgax_3, reorder_atoms_to="self")
        np.testing.assert_equal(
            overlayed.atomic_numbers, self.galgax_4.atomic_numbers
        )
        # compare against reordering obtained by hand
        np.testing.assert_equal(
            order, np.array([
                19, 18, 17, 16, 15, 14, 12, 13, 11, 31, 32, 10, 30, 29,
                9, 27, 28, 8, 26, 25, 6, 7, 5, 0, 1, 2, 3, 4, 24, 23,
                22, 21, 20, 33, 34, 35, 36, 37
            ])
        )

    def test_overlay_of_galgax_5_with_galgax_6_returns_correct_ordering(self):
        overlayed, order, rmsd = self.galgax_5.overlay(self.galgax_6, reorder_atoms_to="self")
        np.testing.assert_equal(
            overlayed.atomic_numbers, self.galgax_5.atomic_numbers
        )
        # compare against reordering obtained by hand
        np.testing.assert_equal(
            order, np.array([
                0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 14, 17, 20, 21, 22, 27,
                26, 25, 24, 23, 37, 36, 35, 34, 33, 10, 9, 13, 12, 16,
                15, 18, 19, 28, 29, 30, 31, 32
            ])
        )

    def test_overlay_of_galgax_6_with_galgax_5_returns_correct_ordering(self):
        overlayed, order, rmsd = self.galgax_6.overlay(self.galgax_5, reorder_atoms_to="self")
        np.testing.assert_equal(
            overlayed.atomic_numbers, self.galgax_6.atomic_numbers
        )
        # compare against reordering obtained by hand
        np.testing.assert_equal(
            order, np.array([
                0, 1, 2, 3, 4, 5, 6, 7, 8, 26, 25, 9, 28, 27, 10, 30,
                29, 11, 31, 32, 12, 13, 14, 19, 18, 17, 16, 15, 33, 34,
                35, 36, 37, 24, 23, 22, 21, 20
            ])
        )

    def test_overlay_of_galgax_7_with_galgax_8_returns_correct_ordering(self):
        overlayed, order, rmsd = self.galgax_7.overlay(self.galgax_8, reorder_atoms_to="self")
        np.testing.assert_equal(
            overlayed.atomic_numbers, self.galgax_7.atomic_numbers
        )
        # compare against reordering obtained by hand
        np.testing.assert_equal(
            order, np.array([
                0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 14, 17, 20, 21, 22, 23,
                24, 25, 26, 27, 37, 36, 35, 34, 33, 10, 9, 13, 12, 16,
                15, 18, 19, 32, 31, 30, 29, 28
            ])
        )

    def test_overlay_of_galgax_8_with_galgax_7_returns_correct_ordering(self):
        overlayed, order, rmsd = self.galgax_8.overlay(self.galgax_7, reorder_atoms_to="self")
        np.testing.assert_equal(
            overlayed.atomic_numbers, self.galgax_8.atomic_numbers
        )
        # compare against reordering obtained by hand
        np.testing.assert_equal(
            order, np.array([
                0, 1, 2, 3, 4, 5, 6, 7, 8, 26, 25, 9, 28, 27, 10, 30,
                29, 11, 31, 32, 12, 13, 14, 15, 16, 17, 18, 19, 37, 36,
                35, 34, 33, 24, 23, 22, 21, 20
            ])
        )

    def test_overlay_of_galgax_9_with_galgax_10_returns_correct_ordering(self):
        overlayed, order, rmsd = self.galgax_9.overlay(self.galgax_10, reorder_atoms_to="self")
        np.testing.assert_equal(
            overlayed.atomic_numbers, self.galgax_9.atomic_numbers
        )
        # compare against reordering obtained by hand
        np.testing.assert_equal(
            order, np.array([
                27, 26, 25, 24, 23, 22, 20, 21, 17, 14, 11, 8, 6, 7, 5,
                0, 1, 2, 3, 4, 28, 29, 30, 31, 32, 18, 19, 16, 15, 13,
                12, 9, 10, 37, 36, 35, 34, 33
            ])
        )

    def test_overlay_of_galgax_10_with_galgax_9_returns_correct_ordering(self):
        overlayed, order, rmsd = self.galgax_10.overlay(self.galgax_9, reorder_atoms_to="self")
        np.testing.assert_equal(
            overlayed.atomic_numbers, self.galgax_10.atomic_numbers
        )
        # compare against reordering obtained by hand
        np.testing.assert_equal(
            order, np.array([
                15, 16, 17, 18, 19, 14, 12, 13, 11, 31, 32, 10, 30, 29,
                9, 28, 27, 8, 25, 26, 6, 7, 5, 4, 3, 2, 1, 0, 20, 21,
                22, 23, 24, 37, 36, 35, 34, 33
            ])
        )

    def test_get_vdw_volume_of_oxygen_1_returns_correct_value(self):
        vdw_volume = self.oxygen_1.get_vdw_volume(d_grid=0.1)
        self.assertAlmostEqual(vdw_volume, 14.77, 2)

    def test_get_vdw_volume_of_nitrogen_2_returns_correct_value(self):
        vdw_volume = self.nitrogen_2.get_vdw_volume(d_grid=0.1)
        self.assertAlmostEqual(vdw_volume, 28.88, 2)

    def test_get_vdw_volume_of_h2o_1_returns_correct_value(self):
        vdw_volume = self.h2o_1.get_vdw_volume(d_grid=0.1)
        self.assertAlmostEqual(vdw_volume, 23.338, 2)

    def test_rotate_to_best_plane(self):
        temp_h2o = deepcopy(self.h2o_1)
        best_axes = temp_h2o.get_best_plane_axes()
        assert np.linalg.norm(np.abs(best_axes) - np.eye(3)) > 0.1
        temp_h2o.rotate_to_best_plane()
        new_best_axes = temp_h2o.get_best_plane_axes()
        assert np.linalg.norm(np.abs(new_best_axes) - np.eye(3)) < 1e-5

    def test_positions_in_molecular_axis_frame(self):
        methods = ("nc", "pca", "moi")
        molecules = [self.c1h1f1cl1br1, self.h2o_2, self.aceta_1, self.galgax_3]
        for method in methods:
            for mol_og in molecules:
                mol = deepcopy(mol_og)
                mol.positions = mol.positions_in_molecular_axis_frame(method=method)
                _, _, rmsd = mol_og.overlay(mol)
                assert rmsd < 1e-6, f"{method}, {mol}"
