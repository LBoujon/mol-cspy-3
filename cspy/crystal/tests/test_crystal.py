from unittest import TestCase
from cspy.chem import Element
from cspy.crystal import Crystal, UnitCell, AsymmetricUnit, SpaceGroup
from os.path import abspath, join, dirname
from tempfile import TemporaryDirectory
import os

EXAMPLES_DIR = abspath(join(dirname(__file__), "../../../examples"))
EXPERIMENTAL = join(EXAMPLES_DIR, "experimental_structures")
GENERATED = join(EXAMPLES_DIR, "generated_structures")


class CrystalTests(TestCase):
    nacl = Crystal(
        UnitCell.cubic(5.6402),
        SpaceGroup(225),
        AsymmetricUnit([Element[x] for x in ("Cl", "Na")], [[0.5] * 3, [0.0] * 3]),
    )
    tcyety_sg2 = None

    def test_load(self):
        for ext in (".cif", ".res"):
            c = Crystal.load(join(EXPERIMENTAL, f"ACETAC01{ext}"))

    def test_save(self):
        c = self.nacl
        with TemporaryDirectory() as tmpdirname:
            for ext in (".res",):
                fname = f"{tmpdirname}/NaCl{ext}"
                c.save(fname)
                os.remove(fname)

    def test_connectivity(self):
        conn = self.nacl.unit_cell_connectivity()

    def test_has_close_contacts(self):
        # assert that the NaCl structure has no close contacts
        self.nacl.has_close_contacts()
        assert not self.nacl.has_close_contacts()

        # assert that the close_contact_example.res structure has close contacts
        c = Crystal.load(join(GENERATED, "close_contact_example.res"))
        assert c.has_close_contacts()