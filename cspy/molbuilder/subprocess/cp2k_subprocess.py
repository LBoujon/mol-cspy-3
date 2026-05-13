import subprocess
import time
import logging
from pathlib import Path
from cspy.molbuilder.exceptions import CP2KCalculationFailure
from cspy.pyAOMlite.pyAOM_utils import single_molecule


LOG = logging.getLogger(__name__)


class CP2KSub:

    def __init__(self, config):
        """Subprocess for calling cp2k. This has been only tested to run
        on the sopt (serial) and ssmp parallel (only OpenMP) versions of
        cp2k. The ssmp parallel will run on the maximum number of
        threads that the job is set to be able to run on.

        :param config: config parser object
        """
        self.cp2k = config['CP2K']['cp2k']
        self.xc_functional = config['CP2K']['XC_FUNCTIONAL']
        self.basis_set_file_name = config['CP2K']['BASIS_SET_FILE_NAME']
        self.potential_file_name = config['CP2K']['POTENTIAL_FILE_NAME']
        self.mgrid_cutoff = config['CP2K']['MGRID_CUTOFF']
        self.mgrid_rel_cutoff = config['CP2K']['MGRID_REL_CUTOFF']
        self.scf_max_scf = config['CP2K']['SCF_MAX_SCF']
        self.mo_cubes_nhomo = config['CP2K']['MO_CUBES_NHOMO']
        self.mo_cubes_nlumo = config['CP2K']['MO_CUBES_NLUMO']
        self.mo_cubes_stride = config['CP2K']['MO_CUBES_STRIDE']
        self.template = config['CP2K']['template']
        self.basis = config['CP2K']['basis']
        self.potential = config['CP2K']['potential']
        LOG.warning(
            '\n--------------------------------------------------------\n'
            'WARNING CP2K MAY GIVE INCORRECT RESULTS!!!\n'
            'SEE MESSAGE FROM ORESTIS ABOUT EXAMPLE FAILURE\n'
            'SEEMS TO WORK FINE FOR LUMO THOUGH\n'
            '--------------------------------------------------------\n'
            'In my calculations for this particular molecule,\n'
            'CP2K seems to get the ordering of frontier occupied\n'
            'states wrong: if you check the FMO of 63 (see attached),\n'
            'you\'ll see that this is not the typical pi-conjugated\n'
            'symmetry you might expect (compare with MO 89 from NWChem:\n'
            'B3LYP/6-311g(d)). I don\'t know if you encountered such\n'
            'issues with CP2K as well. In my calculations, I always\n'
            'double-check with all-electron DFT using NWChem.\n\n'
            'Cheers,\n'
            'Orestis\n'
            '--------------------------------------------------------\n'
            'With that HOMO/LUMO is there a way I can do\n'
            'this checking automatically with a script or something?\n'
            'I have way too many molecules to do this by hand.\n\n'
            'Chi Cheng\n'
            '--------------------------------------------------------\n'
            'There\'s a way: if your PAH molecules are aligned\n'
            'parallel to the x-y plane, you can filter out MOs with\n'
            'diminished pz contributions from the MO information in\n'
            'the CP2K output file. I made a small change to\n'
            'get_cp2k_MO() in pyAOM_utils in order to get this\n'
            'information. Check out the attached ipynb. MOs 61 and 63\n'
            'don\'t show typical pi-conjugation, since the 3pz\n'
            'contribution is close to zero with a large histogram\n'
            'peak value.\n\n'
            'You can go for a non-diagrammatic approach as well:\n'
            'since you have the px, py, and pz information, there\'re\n'
            'python modules that automatically calculate distribution\n'
            'kernels, curve maxima etc... This way, if the unsigned\n'
            'mean value of the pz contribution is close to zero with\n'
            'a peak height larger than px and py, this could be a\n'
            'strong indication that the MO lacks pi-conjugation.\n\n'
            'In case your molecules are not aligned, there\'s an\n'
            'automated way to align them. Let me know and I\n'
            'can help!\n\n'
            'Hope this helps!\n\n'
            'Cheers,\n'
            'Orestis\n'
            '--------------------------------------------------------'
        )

    def run_cp2k(self, xyz_file):
        """Runs the cp2k subprocess.

        :param xyz_file: the XYZ file
        """
        label = Path(xyz_file).stem
        input_file_name = self.write_input_file(xyz_file)

        start = time.time()
        LOG.debug('Started CP2K calculation with input file %s',
                  input_file_name)

        try:
            subprocess.run(
                [
                    self.cp2k,
                    '-i', input_file_name,
                    '-o', label + '_PBE_DZVP-GTH.out'
                ],
                check=True
            )
        except Exception as e:
            raise CP2KCalculationFailure('cp2k failure: %s', e)

        LOG.debug('Completed CP2K calculation for input file %s'
                  ' in %ss', input_file_name, time.time() - start)

    def write_input_file(self, xyz_file):
        """Writes the input file for the cp2k calculation.

        :param xyz_file: the XYZ file
        """
        mol = single_molecule(xyz_file)
        label = Path(xyz_file).stem

        qs_gga_params = {
            'PROJECT_NAME': label + '_PBE_DZVP-GTH',
            'XC_FUNCTIONAL': self.xc_functional,
            'BASIS_SET_FILE_NAME': self.basis_set_file_name,
            'POTENTIAL_FILE_NAME': self.potential_file_name,
            'MGRID_CUTOFF': self.mgrid_cutoff,
            'MGRID_REL_CUTOFF': self.mgrid_rel_cutoff,
            'SCF_MAX_SCF': self.scf_max_scf,
            'MO_CUBES_NHOMO': self.mo_cubes_nhomo,
            'MO_CUBES_NLUMO': self.mo_cubes_nlumo,
            'MO_CUBES_STRIDE': self.mo_cubes_stride,
        }

        mol.prep_cp2k_single(
            qs_gga_params, label, self.template, self.basis,
            self.potential
        )
        return 'output/' + label + '/' + label + '_PBE_DZVP-GTH' + '.inp'
