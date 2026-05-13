from unittest import TestCase
from cspy.molbuilder.mobility.transloc import mol_idx


class TestTransientLocalization(TestCase):

    def test_cell_idx_returns_correct_values_1(self):
        self.assertEqual(mol_idx(0, 0, 0, 0, 3, 3, 3), 0)
        self.assertEqual(mol_idx(1, 0, 0, 0, 3, 3, 3), 1)
        self.assertEqual(mol_idx(2, 0, 0, 0, 3, 3, 3), 2)
        self.assertEqual(mol_idx(0, 1, 0, 0, 3, 3, 3), 3)
        self.assertEqual(mol_idx(1, 1, 0, 0, 3, 3, 3), 4)
        self.assertEqual(mol_idx(2, 1, 0, 0, 3, 3, 3), 5)
        self.assertEqual(mol_idx(0, 2, 0, 0, 3, 3, 3), 6)
        self.assertEqual(mol_idx(1, 2, 0, 0, 3, 3, 3), 7)
        self.assertEqual(mol_idx(2, 2, 0, 0, 3, 3, 3), 8)
        self.assertEqual(mol_idx(0, 0, 1, 0, 3, 3, 3), 9)
        self.assertEqual(mol_idx(1, 0, 1, 0, 3, 3, 3), 10)
        self.assertEqual(mol_idx(2, 0, 1, 0, 3, 3, 3), 11)
        self.assertEqual(mol_idx(0, 1, 1, 0, 3, 3, 3), 12)
        self.assertEqual(mol_idx(1, 1, 1, 0, 3, 3, 3), 13)
        self.assertEqual(mol_idx(2, 1, 1, 0, 3, 3, 3), 14)
        self.assertEqual(mol_idx(0, 2, 1, 0, 3, 3, 3), 15)
        self.assertEqual(mol_idx(1, 2, 1, 0, 3, 3, 3), 16)
        self.assertEqual(mol_idx(2, 2, 1, 0, 3, 3, 3), 17)
        self.assertEqual(mol_idx(0, 0, 2, 0, 3, 3, 3), 18)
        self.assertEqual(mol_idx(1, 0, 2, 0, 3, 3, 3), 19)
        self.assertEqual(mol_idx(2, 0, 2, 0, 3, 3, 3), 20)
        self.assertEqual(mol_idx(0, 1, 2, 0, 3, 3, 3), 21)
        self.assertEqual(mol_idx(1, 1, 2, 0, 3, 3, 3), 22)
        self.assertEqual(mol_idx(2, 1, 2, 0, 3, 3, 3), 23)
        self.assertEqual(mol_idx(0, 2, 2, 0, 3, 3, 3), 24)
        self.assertEqual(mol_idx(1, 2, 2, 0, 3, 3, 3), 25)
        self.assertEqual(mol_idx(2, 2, 2, 0, 3, 3, 3), 26)

    def test_cell_idx_returns_correct_values_2(self):
        self.assertEqual(mol_idx(0, 0, 0, 1, 3, 3, 3), 27)
        self.assertEqual(mol_idx(1, 0, 0, 1, 3, 3, 3), 28)
        self.assertEqual(mol_idx(2, 0, 0, 1, 3, 3, 3), 29)
        self.assertEqual(mol_idx(0, 1, 0, 1, 3, 3, 3), 30)
        self.assertEqual(mol_idx(1, 1, 0, 1, 3, 3, 3), 31)
        self.assertEqual(mol_idx(2, 1, 0, 1, 3, 3, 3), 32)
        self.assertEqual(mol_idx(0, 2, 0, 1, 3, 3, 3), 33)
        self.assertEqual(mol_idx(1, 2, 0, 1, 3, 3, 3), 34)
        self.assertEqual(mol_idx(2, 2, 0, 1, 3, 3, 3), 35)
        self.assertEqual(mol_idx(0, 0, 1, 1, 3, 3, 3), 36)
        self.assertEqual(mol_idx(1, 0, 1, 1, 3, 3, 3), 37)
        self.assertEqual(mol_idx(2, 0, 1, 1, 3, 3, 3), 38)
        self.assertEqual(mol_idx(0, 1, 1, 1, 3, 3, 3), 39)
        self.assertEqual(mol_idx(1, 1, 1, 1, 3, 3, 3), 40)
        self.assertEqual(mol_idx(2, 1, 1, 1, 3, 3, 3), 41)
        self.assertEqual(mol_idx(0, 2, 1, 1, 3, 3, 3), 42)
        self.assertEqual(mol_idx(1, 2, 1, 1, 3, 3, 3), 43)
        self.assertEqual(mol_idx(2, 2, 1, 1, 3, 3, 3), 44)
        self.assertEqual(mol_idx(0, 0, 2, 1, 3, 3, 3), 45)
        self.assertEqual(mol_idx(1, 0, 2, 1, 3, 3, 3), 46)
        self.assertEqual(mol_idx(2, 0, 2, 1, 3, 3, 3), 47)
        self.assertEqual(mol_idx(0, 1, 2, 1, 3, 3, 3), 48)
        self.assertEqual(mol_idx(1, 1, 2, 1, 3, 3, 3), 49)
        self.assertEqual(mol_idx(2, 1, 2, 1, 3, 3, 3), 50)
        self.assertEqual(mol_idx(0, 2, 2, 1, 3, 3, 3), 51)
        self.assertEqual(mol_idx(1, 2, 2, 1, 3, 3, 3), 52)
        self.assertEqual(mol_idx(2, 2, 2, 1, 3, 3, 3), 53)

    def test_cell_idx_returns_correct_values_3(self):
        self.assertEqual(mol_idx(0, 0, 0, 2, 3, 3, 3), 54)
        self.assertEqual(mol_idx(1, 0, 0, 2, 3, 3, 3), 55)
        self.assertEqual(mol_idx(2, 0, 0, 2, 3, 3, 3), 56)
        self.assertEqual(mol_idx(0, 1, 0, 2, 3, 3, 3), 57)
        self.assertEqual(mol_idx(1, 1, 0, 2, 3, 3, 3), 58)
        self.assertEqual(mol_idx(2, 1, 0, 2, 3, 3, 3), 59)
        self.assertEqual(mol_idx(0, 2, 0, 2, 3, 3, 3), 60)
        self.assertEqual(mol_idx(1, 2, 0, 2, 3, 3, 3), 61)
        self.assertEqual(mol_idx(2, 2, 0, 2, 3, 3, 3), 62)
        self.assertEqual(mol_idx(0, 0, 1, 2, 3, 3, 3), 63)
        self.assertEqual(mol_idx(1, 0, 1, 2, 3, 3, 3), 64)
        self.assertEqual(mol_idx(2, 0, 1, 2, 3, 3, 3), 65)
        self.assertEqual(mol_idx(0, 1, 1, 2, 3, 3, 3), 66)
        self.assertEqual(mol_idx(1, 1, 1, 2, 3, 3, 3), 67)
        self.assertEqual(mol_idx(2, 1, 1, 2, 3, 3, 3), 68)
        self.assertEqual(mol_idx(0, 2, 1, 2, 3, 3, 3), 69)
        self.assertEqual(mol_idx(1, 2, 1, 2, 3, 3, 3), 70)
        self.assertEqual(mol_idx(2, 2, 1, 2, 3, 3, 3), 71)
        self.assertEqual(mol_idx(0, 0, 2, 2, 3, 3, 3), 72)
        self.assertEqual(mol_idx(1, 0, 2, 2, 3, 3, 3), 73)
        self.assertEqual(mol_idx(2, 0, 2, 2, 3, 3, 3), 74)
        self.assertEqual(mol_idx(0, 1, 2, 2, 3, 3, 3), 75)
        self.assertEqual(mol_idx(1, 1, 2, 2, 3, 3, 3), 76)
        self.assertEqual(mol_idx(2, 1, 2, 2, 3, 3, 3), 77)
        self.assertEqual(mol_idx(0, 2, 2, 2, 3, 3, 3), 78)
        self.assertEqual(mol_idx(1, 2, 2, 2, 3, 3, 3), 79)
        self.assertEqual(mol_idx(2, 2, 2, 2, 3, 3, 3), 80)

    def test_cell_idx_returns_correct_values_4(self):
        self.assertEqual(mol_idx(-1, 0, 0, 0, 3, 3, 3), 2)
        self.assertEqual(mol_idx(-2, 0, 0, 0, 3, 3, 3), 1)
        self.assertEqual(mol_idx(-3, 0, 0, 0, 3, 3, 3), 0)
        self.assertEqual(mol_idx(-4, 0, 0, 0, 3, 3, 3), 2)
        self.assertEqual(mol_idx(-5, 0, 0, 0, 3, 3, 3), 1)
        self.assertEqual(mol_idx(-6, 0, 0, 0, 3, 3, 3), 0)
        self.assertEqual(mol_idx(-7, 0, 0, 0, 3, 3, 3), 2)
        self.assertEqual(mol_idx(-8, 0, 0, 0, 3, 3, 3), 1)
        self.assertEqual(mol_idx(-9, 0, 0, 0, 3, 3, 3), 0)
        self.assertEqual(mol_idx(-10, 0, 0, 0, 3, 3, 3), 2)
        self.assertEqual(mol_idx(-11, 0, 0, 0, 3, 3, 3), 1)

    def test_cell_idx_returns_correct_values_5(self):
        self.assertEqual(mol_idx(0, 0, 0, 0, 3, 3, 3), 0)
        self.assertEqual(mol_idx(1, 0, 0, 0, 3, 3, 3), 1)
        self.assertEqual(mol_idx(2, 0, 0, 0, 3, 3, 3), 2)
        self.assertEqual(mol_idx(3, 0, 0, 0, 3, 3, 3), 0)
        self.assertEqual(mol_idx(4, 0, 0, 0, 3, 3, 3), 1)
        self.assertEqual(mol_idx(5, 0, 0, 0, 3, 3, 3), 2)
        self.assertEqual(mol_idx(6, 0, 0, 0, 3, 3, 3), 0)
        self.assertEqual(mol_idx(7, 0, 0, 0, 3, 3, 3), 1)
        self.assertEqual(mol_idx(8, 0, 0, 0, 3, 3, 3), 2)
        self.assertEqual(mol_idx(9, 0, 0, 0, 3, 3, 3), 0)

    def test_cell_idx_returns_correct_values_6(self):
        self.assertEqual(mol_idx(-1, 0, 0, 0, 4, 4, 4), 3)
        self.assertEqual(mol_idx(-2, 0, 0, 0, 4, 4, 4), 2)
        self.assertEqual(mol_idx(-3, 0, 0, 0, 4, 4, 4), 1)
        self.assertEqual(mol_idx(-4, 0, 0, 0, 4, 4, 4), 0)
        self.assertEqual(mol_idx(-5, 0, 0, 0, 4, 4, 4), 3)
        self.assertEqual(mol_idx(-6, 0, 0, 0, 4, 4, 4), 2)
        self.assertEqual(mol_idx(-7, 0, 0, 0, 4, 4, 4), 1)
        self.assertEqual(mol_idx(-8, 0, 0, 0, 4, 4, 4), 0)
        self.assertEqual(mol_idx(-9, 0, 0, 0, 4, 4, 4), 3)
        self.assertEqual(mol_idx(-10, 0, 0, 0, 4, 4, 4), 2)
        self.assertEqual(mol_idx(-11, 0, 0, 0, 4, 4, 4), 1)

    def test_cell_idx_returns_correct_values_7(self):
        self.assertEqual(mol_idx(-1, 0, 0, 1, 4, 4, 4), 67)
        self.assertEqual(mol_idx(-2, 0, 0, 1, 4, 4, 4), 66)
        self.assertEqual(mol_idx(-3, 0, 0, 1, 4, 4, 4), 65)
        self.assertEqual(mol_idx(-4, 0, 0, 1, 4, 4, 4), 64)
        self.assertEqual(mol_idx(-5, 0, 0, 1, 4, 4, 4), 67)
        self.assertEqual(mol_idx(-6, 0, 0, 1, 4, 4, 4), 66)
        self.assertEqual(mol_idx(-7, 0, 0, 1, 4, 4, 4), 65)
        self.assertEqual(mol_idx(-8, 0, 0, 1, 4, 4, 4), 64)
        self.assertEqual(mol_idx(-9, 0, 0, 1, 4, 4, 4), 67)
        self.assertEqual(mol_idx(-10, 0, 0, 1, 4, 4, 4), 66)
        self.assertEqual(mol_idx(-11, 0, 0, 1, 4, 4, 4), 65)

    def test_cell_idx_returns_correct_values_8(self):
        self.assertEqual(mol_idx(0, 0, 0, 0, 2, 2, 2), 0)
        self.assertEqual(mol_idx(1, 0, 0, 0, 2, 2, 2), 1)
        self.assertEqual(mol_idx(0, 1, 0, 0, 2, 2, 2), 2)
        self.assertEqual(mol_idx(1, 1, 0, 0, 2, 2, 2), 3)
        self.assertEqual(mol_idx(0, 0, 1, 0, 2, 2, 2), 4)
        self.assertEqual(mol_idx(1, 0, 1, 0, 2, 2, 2), 5)
        self.assertEqual(mol_idx(0, 1, 1, 0, 2, 2, 2), 6)
        self.assertEqual(mol_idx(1, 1, 1, 0, 2, 2, 2), 7)
        self.assertEqual(mol_idx(0, 0, 0, 1, 2, 2, 2), 8)
        self.assertEqual(mol_idx(1, 0, 0, 1, 2, 2, 2), 9)
        self.assertEqual(mol_idx(0, 1, 0, 1, 2, 2, 2), 10)
        self.assertEqual(mol_idx(1, 1, 0, 1, 2, 2, 2), 11)
        self.assertEqual(mol_idx(0, 0, 1, 1, 2, 2, 2), 12)
        self.assertEqual(mol_idx(1, 0, 1, 1, 2, 2, 2), 13)
        self.assertEqual(mol_idx(0, 1, 1, 1, 2, 2, 2), 14)
        self.assertEqual(mol_idx(1, 1, 1, 1, 2, 2, 2), 15)
        self.assertEqual(mol_idx(0, 0, 0, 2, 2, 2, 2), 16)
        self.assertEqual(mol_idx(1, 0, 0, 2, 2, 2, 2), 17)
        self.assertEqual(mol_idx(0, 1, 0, 2, 2, 2, 2), 18)
        self.assertEqual(mol_idx(1, 1, 0, 2, 2, 2, 2), 19)
        self.assertEqual(mol_idx(0, 0, 1, 2, 2, 2, 2), 20)
        self.assertEqual(mol_idx(1, 0, 1, 2, 2, 2, 2), 21)
        self.assertEqual(mol_idx(0, 1, 1, 2, 2, 2, 2), 22)
        self.assertEqual(mol_idx(1, 1, 1, 2, 2, 2, 2), 23)

    def test_cell_idx_returns_correct_values_9(self):
        self.assertEqual(mol_idx(0, 0, 0, 0, 4, 3, 2), 0)
        self.assertEqual(mol_idx(1, 0, 0, 0, 4, 3, 2), 1)
        self.assertEqual(mol_idx(2, 0, 0, 0, 4, 3, 2), 2)
        self.assertEqual(mol_idx(3, 0, 0, 0, 4, 3, 2), 3)
        self.assertEqual(mol_idx(0, 1, 0, 0, 4, 3, 2), 4)
        self.assertEqual(mol_idx(1, 1, 0, 0, 4, 3, 2), 5)
        self.assertEqual(mol_idx(2, 1, 0, 0, 4, 3, 2), 6)
        self.assertEqual(mol_idx(3, 1, 0, 0, 4, 3, 2), 7)
        self.assertEqual(mol_idx(0, 2, 0, 0, 4, 3, 2), 8)
        self.assertEqual(mol_idx(1, 2, 0, 0, 4, 3, 2), 9)
        self.assertEqual(mol_idx(2, 2, 0, 0, 4, 3, 2), 10)
        self.assertEqual(mol_idx(3, 2, 0, 0, 4, 3, 2), 11)
        self.assertEqual(mol_idx(0, 0, 1, 0, 4, 3, 2), 12)
        self.assertEqual(mol_idx(1, 0, 1, 0, 4, 3, 2), 13)
        self.assertEqual(mol_idx(2, 0, 1, 0, 4, 3, 2), 14)
        self.assertEqual(mol_idx(3, 0, 1, 0, 4, 3, 2), 15)
        self.assertEqual(mol_idx(0, 1, 1, 0, 4, 3, 2), 16)
        self.assertEqual(mol_idx(1, 1, 1, 0, 4, 3, 2), 17)
        self.assertEqual(mol_idx(2, 1, 1, 0, 4, 3, 2), 18)
        self.assertEqual(mol_idx(3, 1, 1, 0, 4, 3, 2), 19)
        self.assertEqual(mol_idx(0, 2, 1, 0, 4, 3, 2), 20)
        self.assertEqual(mol_idx(1, 2, 1, 0, 4, 3, 2), 21)
        self.assertEqual(mol_idx(2, 2, 1, 0, 4, 3, 2), 22)
        self.assertEqual(mol_idx(3, 2, 1, 0, 4, 3, 2), 23)

    def test_cell_idx_returns_correct_values_10(self):
        self.assertEqual(mol_idx(0, 0, 0, 0, 4, 3, 2), 0)
        self.assertEqual(mol_idx(-1, 0, 0, 0, 4, 3, 2), 3)  # 3 0 0
        self.assertEqual(mol_idx(0, -1, 0, 0, 4, 3, 2), 8)  # 0 2 0
        self.assertEqual(mol_idx(-1, -1, 0, 0, 4, 3, 2), 11)  # 3 2 0
        self.assertEqual(mol_idx(0, 0, -1, 0, 4, 3, 2), 12)  # 0 0 1
        self.assertEqual(mol_idx(-1, 0, -1, 0, 4, 3, 2), 15)  # 3 0 1
        self.assertEqual(mol_idx(0, -1, -1, 0, 4, 3, 2), 20)  # 0 2 1
        self.assertEqual(mol_idx(-1, -1, -1, 0, 4, 3, 2), 23)  # 3 2 1

    def test_cell_idx_returns_correct_values_11(self):
        self.assertEqual(mol_idx(0, 0, 0, 0, 4, 3, 2), 0)
        self.assertEqual(mol_idx(-2, 0, 0, 0, 4, 3, 2), 2)  # 2 0 0
        self.assertEqual(mol_idx(0, -2, 0, 0, 4, 3, 2), 4)  # 0 1 0
        self.assertEqual(mol_idx(-2, -2, 0, 0, 4, 3, 2), 6)  # 2 1 0
        self.assertEqual(mol_idx(0, 0, -2, 0, 4, 3, 2), 0)  # 0 0 0
        self.assertEqual(mol_idx(-2, 0, -2, 0, 4, 3, 2), 2)  # 2 0 0
        self.assertEqual(mol_idx(0, -2, -2, 0, 4, 3, 2), 4)  # 0 1 0
        self.assertEqual(mol_idx(-2, -2, -2, 0, 4, 3, 2), 6)  # 2 1 0

    def test_cell_idx_returns_correct_values_12(self):
        self.assertEqual(mol_idx(0, 0, 0, 1, 4, 3, 2), 24)
        self.assertEqual(mol_idx(1, 0, 0, 1, 4, 3, 2), 25)
        self.assertEqual(mol_idx(2, 0, 0, 1, 4, 3, 2), 26)
        self.assertEqual(mol_idx(3, 0, 0, 1, 4, 3, 2), 27)
        self.assertEqual(mol_idx(0, 1, 0, 1, 4, 3, 2), 28)
        self.assertEqual(mol_idx(1, 1, 0, 1, 4, 3, 2), 29)
        self.assertEqual(mol_idx(2, 1, 0, 1, 4, 3, 2), 30)
        self.assertEqual(mol_idx(3, 1, 0, 1, 4, 3, 2), 31)
        self.assertEqual(mol_idx(0, 2, 0, 1, 4, 3, 2), 32)
        self.assertEqual(mol_idx(1, 2, 0, 1, 4, 3, 2), 33)
        self.assertEqual(mol_idx(2, 2, 0, 1, 4, 3, 2), 34)
        self.assertEqual(mol_idx(3, 2, 0, 1, 4, 3, 2), 35)
        self.assertEqual(mol_idx(0, 0, 1, 1, 4, 3, 2), 36)
        self.assertEqual(mol_idx(1, 0, 1, 1, 4, 3, 2), 37)
        self.assertEqual(mol_idx(2, 0, 1, 1, 4, 3, 2), 38)
        self.assertEqual(mol_idx(3, 0, 1, 1, 4, 3, 2), 39)
        self.assertEqual(mol_idx(0, 1, 1, 1, 4, 3, 2), 40)
        self.assertEqual(mol_idx(1, 1, 1, 1, 4, 3, 2), 41)
        self.assertEqual(mol_idx(2, 1, 1, 1, 4, 3, 2), 42)
        self.assertEqual(mol_idx(3, 1, 1, 1, 4, 3, 2), 43)
        self.assertEqual(mol_idx(0, 2, 1, 1, 4, 3, 2), 44)
        self.assertEqual(mol_idx(1, 2, 1, 1, 4, 3, 2), 45)
        self.assertEqual(mol_idx(2, 2, 1, 1, 4, 3, 2), 46)
        self.assertEqual(mol_idx(3, 2, 1, 1, 4, 3, 2), 47)

    def test_cell_idx_returns_correct_values_13(self):
        self.assertEqual(mol_idx(0, 0, 0, 1, 4, 3, 2), 24)
        self.assertEqual(mol_idx(-1, 0, 0, 1, 4, 3, 2), 27)  # 3 0 0
        self.assertEqual(mol_idx(0, -1, 0, 1, 4, 3, 2), 32)  # 0 2 0
        self.assertEqual(mol_idx(-1, -1, 0, 1, 4, 3, 2), 35)  # 3 2 0
        self.assertEqual(mol_idx(0, 0, -1, 1, 4, 3, 2), 36)  # 0 0 1
        self.assertEqual(mol_idx(-1, 0, -1, 1, 4, 3, 2), 39)  # 3 0 1
        self.assertEqual(mol_idx(0, -1, -1, 1, 4, 3, 2), 44)  # 0 2 1
        self.assertEqual(mol_idx(-1, -1, -1, 1, 4, 3, 2), 47)  # 3 2 1

    def test_cell_idx_returns_correct_values_14(self):
        self.assertEqual(mol_idx(0, 0, 0, 1, 4, 3, 2), 24)
        self.assertEqual(mol_idx(-2, 0, 0, 1, 4, 3, 2), 26)  # 2 0 0
        self.assertEqual(mol_idx(0, -2, 0, 1, 4, 3, 2), 28)  # 0 1 0
        self.assertEqual(mol_idx(-2, -2, 0, 1, 4, 3, 2),30)  # 2 1 0
        self.assertEqual(mol_idx(0, 0, -2, 1, 4, 3, 2), 24)  # 0 0 0
        self.assertEqual(mol_idx(-2, 0, -2, 1, 4, 3, 2), 26)  # 2 0 0
        self.assertEqual(mol_idx(0, -2, -2, 1, 4, 3, 2), 28)  # 0 1 0
        self.assertEqual(mol_idx(-2, -2, -2, 1, 4, 3, 2), 30)  # 2 1 0

    def test_cell_idx_returns_correct_values_15(self):
        self.assertEqual(mol_idx(0, 0, 0, 0, 4, 3, 2), 0)
        self.assertEqual(mol_idx(4, 0, 0, 0, 4, 3, 2), 0)
        self.assertEqual(mol_idx(0, 3, 0, 0, 4, 3, 2), 0)
        self.assertEqual(mol_idx(0, 0, 2, 0, 4, 3, 2), 0)
        self.assertEqual(mol_idx(4, 3, 0, 0, 4, 3, 2), 0)
        self.assertEqual(mol_idx(4, 0, 2, 0, 4, 3, 2), 0)
        self.assertEqual(mol_idx(0, 3, 2, 0, 4, 3, 2), 0)
        self.assertEqual(mol_idx(4, 3, 2, 0, 4, 3, 2), 0)

    def test_cell_idx_returns_correct_values_16(self):
        self.assertEqual(mol_idx(0, 0, 0, 1, 4, 3, 2), 24)
        self.assertEqual(mol_idx(4, 0, 0, 1, 4, 3, 2), 24)
        self.assertEqual(mol_idx(0, 3, 0, 1, 4, 3, 2), 24)
        self.assertEqual(mol_idx(0, 0, 2, 1, 4, 3, 2), 24)
        self.assertEqual(mol_idx(4, 3, 0, 1, 4, 3, 2), 24)
        self.assertEqual(mol_idx(4, 0, 2, 1, 4, 3, 2), 24)
        self.assertEqual(mol_idx(0, 3, 2, 1, 4, 3, 2), 24)
        self.assertEqual(mol_idx(4, 3, 2, 1, 4, 3, 2), 24)
