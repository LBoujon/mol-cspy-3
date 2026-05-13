from unittest import TestCase
from cspy.cspympi.reopt_tasks import ReoptWorker
from cspy.distributed.reoptimization import MinimizationStructure
from cspy.configuration import CONFIG
from cspy.configuration import configure

LJ_inp = "TITL LJ2\nCELL 1 10 10 10 90 90 90\nLATT -1\nSFAC H\nH1      1  0.0  0.0  0.0  1.000000\nH2      1  0.45 0.5  0.5  1.000000"
#LJ_out_expected = "TITL LJ2_opt_by_ase\nCELL 0.7 1.28238 1.28238 1.141557 90 90 90\nLATT -1\nSFAC H\nH1    1       0.500000000000      -0.000000000000       0.750000000239\nH2    1      -0.000000000000       0.500000000000       0.249999999761"
LJ_latt_out_expected = [1.28238, 1.28238, 1.14156, 90.0, 90.0, 90.0]
LJ_at0_out_expected = [0.5, -0.0, 0.75]
LJ_at1_out_expected = [-0.0, 0.5, 0.25]
LJ_energy_expected = -16.48832

def check_cell(cell_line):
    """ We don't need full accuracy so lets round """
    
    _, _, A, B, C, alpha, beta, gamma =  cell_line.split()
    LJ_latt_out_actual = [round(float(A), 5), round(float(B), 5), round(float(C), 5), round(float(alpha), 5), round(float(beta), 5), round(float(gamma), 5)]
    assert LJ_latt_out_actual == LJ_latt_out_expected


def check_coords(coord_line0, coord_line1):
    """ We don't need full accuracy so lets round """

    _, _, X, Y, Z =  coord_line0.split()
    LJ_at0_out_actual = [round(float(X), 5), round(float(Y), 5), round(float(Z), 5)]
    _, _, X, Y, Z =  coord_line1.split()
    LJ_at1_out_actual = [round(float(X), 5), round(float(Y), 5), round(float(Z), 5)]

    assert LJ_at0_out_actual == LJ_at0_out_expected
    assert LJ_at1_out_actual == LJ_at1_out_expected


def override_config():
    # NOTE: LJ does not play well with LBFGS. It just pushes the atoms to the cutoff distance
    # BFGS is fine
    configure({"csp_minimization_step": [{'kind': 'ase', 
                                        'calculator_type' : 'ase_builtin',
                                        'fmax' : 0.000001,
                                        'optimizer' : 'BFGS',
                                        'ase' : {'model_units' : {'energy' : 'kJ/mol',
                                                                  'length' : 'Ang',
                                                                  'normalisation' : 'molecular_formula_unit',
                                                                  'energy_corr' : None},
                                                'ase_builtin' : {'model' : 'Lennard-Jones',
                                                                'sigma' : 1, 
                                                                'epsilon' : 1, 
                                                                'rc' : 10,
                                                                'smooth' : False}
                                                 }
                                        }]})


class CompositeASELJ(TestCase):
    structure = MinimizationStructure(id="1",
                            filename="LJ2",
                            energy=0,
                            file_content=LJ_inp,
                            spacegroup="1",
                            trial_number="0",)
    
    descriptors = CONFIG.get("descriptors")
    worker_data  = {
            "charges": None,
            "multipoles": None,
            "axis": None,
            "bondlength_cutoffs": None,
            "asymmetric_unit": ['lj.xyz', 'lj.xyz'],
            "minimization": {
                "minimization_steps": [{'kind': 'ase', 
                                        'model' : 'Lennard-Jones', 
                                        'sigma' : 1, 
                                        'epsilon' : 1, 
                                        'rc' : 10,
                                        'fmax' : 0.000001,
                                        'smooth' : False,
                                        'optimizer' : 'BFGS'
                                        }]
            },
            "check_single_point_energy": False,
            "descriptors" : {"pxrd" : None},
            "keep_files" : False,
            "Zp" : 2,
        }
    
    reoptworker = ReoptWorker(worker_data)

    def test_optimisation(self):
        override_config()
        tag, data, mtime = self.reoptworker.optimize_structure(self.structure)
        valid, crystals = data
        assert valid == True
        crystal = crystals[1]

        LJ_energy_actual = round(crystal.energy, 5)
        assert LJ_energy_actual == LJ_energy_expected

        LJ_out_actual = crystal.file_content
        LJ_out_actual_split = LJ_out_actual.split('\n')

        check_cell(LJ_out_actual_split[1])
        check_coords(LJ_out_actual_split[4], LJ_out_actual_split[5])