import logging
from cspy.util.logging_config import FORMATS, DATEFMT
from cspy.crystal import Crystal
from cspy.minimize import DmacrysMinimizer
from cspy.chem.multipole import DistributedMultipoles
from unittest import TestCase
import pytest
import shutil
import pathlib
import subprocess
import sqlite3
import os


RES_STRING = """TITL ACETAC01 ACETAC01
CELL 1.0 9.92687 6.00001 7.50178 90.0 90.0 90.0
ZERR 1 0.0 0.0 0.0 0.0 0.0 0.0
LATT -1
SFAC  C O H
C1 1     0.048386223    -0.010429983    -0.038806956 
C2 1    -0.053113041     0.051193248     0.097463369 
H1 3     0.071045934    -0.170484716    -0.259437260 
H2 3    -0.001357553     0.142399763     0.207355954 
H3 3    -0.128734057     0.162849729     0.044323980 
H4 3    -0.101917951    -0.090729849     0.153524730 
O1 2    -0.000418687    -0.131969780    -0.170231765 
O2 2     0.166109131     0.047171588    -0.034192052 
END
"""

MULT_STRING = """C_F1_1____  -0.0406904452   0.0703428721  -0.0061084260  Rank 4
  0.2237053149 
 -0.0008989020  -0.1243301783  -0.0445817412 
 -0.5260667863  -0.0119057136   0.0124875997  -0.1335336853  -0.1019408098 
 -0.0526080665  -0.2847650689   0.0428887857  -0.0261007409  -0.0050904193 
  1.1523440412   0.1831375909 
  0.3587605160  -0.1243536814  -0.0048686624  -0.4879510812   0.3332769355 
 -0.0647717619   0.0074811820   0.3560292808   0.3455957895 
C_F1_2____   1.4415176861   0.0703428721  -0.0061084260  Rank 4
 -0.0838682566 
 -0.0056345005  -0.1321791085  -0.0300982209 
 -0.0786771660  -0.0082992695   0.0367097409   0.1571039895  -0.0423793264 
  0.3841897460   1.1866389538  -1.7697606180   0.2856380330   0.0623501731 
 -1.0810428434  -0.5604100362 
 -0.2890305848   0.4735565436   0.0050164272   1.0445135665  -2.3917549465 
  0.4976412712   0.0622121887  -0.6955902720  -1.2231507099 
H_F2_1____  -1.5746820038  -1.0954718059  -0.0061084260  Rank 4
  0.2869091284 
  0.0001085212   0.0381096372   0.0264010920 
 -0.0935114728   0.0004172541   0.0001784780   0.0876202354   0.0127056665 
  0.0002754221   0.0437503242   0.0117099056  -0.0005544268  -0.0003464349 
 -0.0426961261  -0.0521402930 
 -0.0743528809   0.0006539382   0.0005308166   0.0736778929   0.0305837505 
  0.0000290747  -0.0000057393  -0.0447127001   0.0053559335 
H_F1_2____   1.7973560011   1.1241767274  -0.0826813166  Rank 4
  0.1009461830 
  0.0091949547  -0.0920257108  -0.0957971271 
 -0.1268955104  -0.0055250755  -0.0203447708  -0.0920318642   0.0494348975 
  0.0236622359  -0.0646148057  -0.1300765953   0.0123384221  -0.0239365799 
 -0.1304210910   0.0303316356 
 -0.1177605418  -0.0031695209  -0.0314574472  -0.0536934116   0.0053756770 
  0.0042332585  -0.0258787147  -0.1889711872   0.0387902698 
H_F1_3____   1.8439905472  -0.3284903056   0.9159553587  Rank 4
  0.0901679695 
 -0.0978817611  -0.0864957466   0.0394440164 
  0.0875495141   0.0775832929  -0.1048911018   0.0275387630  -0.0225999790 
 -0.0083908527   0.1454455094  -0.0869280269   0.0680230533  -0.1075020989 
  0.0251166668  -0.0443962449 
 -0.0420646834   0.0027182358   0.1304443658   0.1010572708   0.0208206150 
  0.0214866100  -0.0186056617  -0.0232171510  -0.0449999390 
H_F1_4____   1.8484949301  -0.4760586111  -0.8261858001  Rank 4
  0.0837232170 
  0.0904669080  -0.0907281881   0.0538430005 
  0.0479421828  -0.0722367172   0.1339300353   0.0059583831  -0.0328423657 
  0.0528347632   0.1098281072  -0.0871054610  -0.0338768227   0.1421913172 
  0.0049686140  -0.0573712101 
  0.0242508888  -0.0119115276  -0.0964826838   0.1429310475   0.0223831440 
 -0.0290276297   0.0552255112  -0.0324972220  -0.0612823238 
O_F1_1____  -0.5732459779  -1.1358628705  -0.0045534020  Rank 4
 -0.3013109409 
  0.0030649604  -0.1841907590   0.3965269727 
 -0.7329510759   0.0093285698  -0.0051767908   0.1174988661   0.5556084283 
 -0.0089568689   0.1614384776  -0.2319634995   0.0091853718  -0.0156435523 
 -0.8875376123   0.0696138475 
  0.3341087257  -0.0567437994   0.0116676138  -0.9720089943   0.3992477767 
  0.0133157536  -0.0522909861   0.4577772853   0.7095901060 
O_F1_2____  -0.7250003069   1.0791276174   0.0136628291  Rank 4
 -0.4002722750 
 -0.0030377699   0.2353558235  -0.3567598133 
 -0.1742863278  -0.0026231537   0.0219465378  -0.3909881443  -0.8979161421 
  0.0098811630  -0.2262934405   0.4244944403   0.0149205908   0.0324441506 
 -0.1783754303  -0.1216155649 
 -0.2033761770  -0.0392500118   0.0037541019  -0.1628315107  -0.1833201257 
 -0.0371300325   0.0516366885   1.0916804542  -1.4351779444 

#ENDMOL"""


AXIS_STRING = """MOLX 1
X LINE  C_F1_1____ C_F1_2____ 1
Y PLANE C_F1_1____ C_F1_2____ 1 H_F2_1____ 2
ENDS"""


class DmacrysMinimizerTestCase(TestCase):
    crystal = Crystal.from_shelx_string(RES_STRING)
    mults = DistributedMultipoles.from_dma_string(MULT_STRING)

    @pytest.mark.external_binaries
    def test_dmacrys_minimize(self):
        minimizer = DmacrysMinimizer(self.mults, AXIS_STRING, name="TEST ACETAC01")
        final = minimizer.minimize(self.crystal)
        # check the lattice energy is within 0.1 kJ/mol of expected
        assert abs(final.properties["lattice_energy"] + 47.0862) < 0.1


class DmacrysReoptimizeTestCase(TestCase):
    dir_location = str(pathlib.Path(__file__).parent.resolve())
    data_location = dir_location + "/data"

    @pytest.mark.external_binaries
    def test_reoptimize(self):
        reoptimise_command = (
            "mpiexec -np 2 cspy-reoptimize "
            + self.data_location
            + "/acetic_fine10.db -x"
            + self.data_location
            + "/acetic.xyz -c"
            + self.data_location
            + "/acetic_rank0.dma -m"
            + self.data_location
            + "/acetic.dma -a"
            + self.data_location
            + "/acetic.mols"
        )

        shutil.copyfile(
            self.data_location + "/acetic_fine10_singlepoint_cspy.toml", "cspy.toml"
        )
        subprocess.run(reoptimise_command, shell=True)

    @pytest.mark.external_binaries
    def test_spg2_energy(self):
        con = sqlite3.connect("acetic_fine10.opt.db")
        cursor = con.cursor()
        sql_query = "select energy from crystal where id = 'acetic-QR-2-3-1-OPT-1';"
        cursor.execute(sql_query)
        fetch = cursor.fetchall()
        assert len(fetch) > 0
        energy = float(fetch[0][0])
        cursor.close()

        assert abs(energy - 216.4792) < 0.1

    @pytest.mark.external_binaries
    def test_spg4_energy(self):
        con = sqlite3.connect("acetic_fine10.opt.db")
        cursor = con.cursor()
        sql_query = "select energy from crystal where id = 'acetic-QR-4-1-1-OPT-1';"
        cursor.execute(sql_query)
        fetch = cursor.fetchall()
        assert len(fetch) > 0
        energy = float(fetch[0][0])
        cursor.close()

        assert abs(energy - 243.0728) < 0.1

    @pytest.mark.external_binaries
    def test_spg5_energy(self):
        con = sqlite3.connect("acetic_fine10.opt.db")
        cursor = con.cursor()
        sql_query = "select energy from crystal where id = 'acetic-QR-5-3-1-OPT-1';"
        cursor.execute(sql_query)
        fetch = cursor.fetchall()
        assert len(fetch) > 0
        energy = float(fetch[0][0])
        cursor.close()

        assert abs(energy - 730.3071) < 0.1

    @pytest.mark.external_binaries
    def test_spg9_energy(self):
        con = sqlite3.connect("acetic_fine10.opt.db")
        cursor = con.cursor()
        sql_query = "select energy from crystal where id = 'acetic-QR-9-1-1-OPT-1';"
        cursor.execute(sql_query)
        fetch = cursor.fetchall()
        assert len(fetch) > 0
        energy = float(fetch[0][0])
        cursor.close()

        assert abs(energy - 374.5782) < 0.1

    @pytest.mark.external_binaries
    def test_spg14_energy(self):
        con = sqlite3.connect("acetic_fine10.opt.db")
        cursor = con.cursor()
        sql_query = "select energy from crystal where id = 'acetic-QR-14-3-1-OPT-1';"
        cursor.execute(sql_query)
        fetch = cursor.fetchall()
        assert len(fetch) > 0
        energy = float(fetch[0][0])
        cursor.close()

        assert abs(energy - 347.8426) < 0.1

    @pytest.mark.external_binaries
    def test_spg15_energy(self):
        con = sqlite3.connect("acetic_fine10.opt.db")
        cursor = con.cursor()
        sql_query = "select energy from crystal where id = 'acetic-QR-15-4-1-OPT-1';"
        cursor.execute(sql_query)
        fetch = cursor.fetchall()
        assert len(fetch) > 0
        energy = float(fetch[0][0])
        cursor.close()

        assert abs(energy - 1156.0548) < 0.1

    @pytest.mark.external_binaries
    def test_spg19_energy(self):
        con = sqlite3.connect("acetic_fine10.opt.db")
        cursor = con.cursor()
        sql_query = "select energy from crystal where id = 'acetic-QR-19-1-1-OPT-1';"
        cursor.execute(sql_query)
        fetch = cursor.fetchall()
        assert len(fetch) > 0
        energy = float(fetch[0][0])
        cursor.close()

        assert abs(energy - 323.2417) < 0.1

    @pytest.mark.external_binaries
    def test_spg29_energy(self):
        con = sqlite3.connect("acetic_fine10.opt.db")
        cursor = con.cursor()
        sql_query = "select energy from crystal where id = 'acetic-QR-29-1-1-OPT-1';"
        cursor.execute(sql_query)
        fetch = cursor.fetchall()
        assert len(fetch) > 0
        energy = float(fetch[0][0])
        cursor.close()

        assert abs(energy - 653.2653) < 0.1

    @pytest.mark.external_binaries
    def test_spg33_energy(self):
        con = sqlite3.connect("acetic_fine10.opt.db")
        cursor = con.cursor()
        sql_query = "select energy from crystal where id = 'acetic-QR-33-1-1-OPT-1';"
        cursor.execute(sql_query)
        fetch = cursor.fetchall()
        assert len(fetch) > 0
        energy = float(fetch[0][0])
        cursor.close()

        assert abs(energy - 650.0137) < 0.1

    @pytest.mark.external_binaries
    def test_spg61_energy(self):
        con = sqlite3.connect("acetic_fine10.opt.db")
        cursor = con.cursor()
        sql_query = "select energy from crystal where id = 'acetic-QR-61-5-1-OPT-1';"
        cursor.execute(sql_query)
        fetch = cursor.fetchall()
        assert len(fetch) > 0
        energy = float(fetch[0][0])
        cursor.close()

        assert abs(energy - 480.0746) < 0.1

    @pytest.mark.external_binaries
    def test_tidy_up(self):
        for item in ["acetic_fine10.opt.db", "acetic_fine10.opt.db-shm", "acetic_fine10.opt.db-wal", "errors.txt", "cspy.toml"]:
            if os.path.isfile(item):
                os.remove(item)
