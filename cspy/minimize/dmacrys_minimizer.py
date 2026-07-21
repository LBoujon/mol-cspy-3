import logging
from cspy.executable import Neighcrys, Dmacrys, ReturnCodeError, TimeoutExpired
from cspy.potentials import available_potentials, PotentialData
from cspy.formats import DmacrysSummary, DmacrysOutput
from os.path import exists, basename, join
from cspy.crystal import Crystal
from cspy.chem.multipole import DistributedMultipoles
import sys
import os
from cspy.util.timer import Timer
from cspy.util.path import Path

if sys.version.startswith("2"):
    from backports.tempfile import TemporaryDirectory
else:
    from tempfile import TemporaryDirectory

LOG = logging.getLogger("dmacrys_minimizer")
BOND_CUTOFFS = """
C_  S_  1.9
I_  Hg  2.8
I_  C_  2.3
B_  F_  1.45
O_  H_  1.3
C_  H_  1.28
O_  S_  1.5
C_  Br  1.95
N_  S_  1.7
C_  C_  1.65
N_  N_  1.55
C_  Cl  1.95
B_  C_  1.65
S_  S_  2.5
S_  Cl  2.2
Br  Hg  2.5
N_  O_  1.55
C_  O_  1.55
C_  F_  1.45
S_  F_  1.8
N_  Hg  2.8
N_  H_  1.2
C_  N_  1.58
ENDS
"""


class DmacrysMinimizer:
    def __init__(
        self,
        multipoles,
        axis,
        reorder_atoms_if_high_rmsd=False,
        reorder_method='molecular_axis',
        name="Unknown",
        potential="fit",
        bondlengths=None,
        keep_files=False,
        **kwargs
    ):
        
        self.multipoles = multipoles
        self.axis = axis
        self.reorder_atoms_if_high_rmsd = reorder_atoms_if_high_rmsd
        self.reorder_method = reorder_method
        self.name = name
        self.kwargs = kwargs
        self.bondlengths = bondlengths
        if not self.bondlengths:
            self.bondlengths = BOND_CUTOFFS
        self.keep_files = keep_files
        self.potential = potential
        self.potential_type, self.potential_filename = available_potentials[potential]
        if self.potential_type == "W":
            self.kwargs["foreshorten_hydrogens"] = self.kwargs.get(
                "foreshorten_hydrogens", True
            )
        self._last_dmain = None
        self._last_symmetry = None
        self._last_summary = None
        self._last_dmaout = None
        self.rmsd_error_threshold = kwargs.get("rmsd_error_threshold", None)

    @property
    def res_filename(self):
        return "{}.res".format(self.name)

    @property
    def multipole_filename(self):
        return "{}.dma".format(self.name)

    @property
    def axis_filename(self):
        return "{}.mols".format(self.name)

    @property
    def potential_basename(self):
        return basename(self.potential_filename)
    
    def interpret_dmacrys_error(self, summary_contents):
        error = "string"
        if 'too many iteration cycles' in summary_contents:
            error = "MaxIts"
            LOG.exception("Error in Dmacrys: Total GO iterations exceeds max allowed GO iterations.", stack_info=False, exc_info=False)

        return error

    def check_multipole_mappings(self, mappings):
        if self.rmsd_error_threshold is None:
            return

        for idx, rmsd, inverted in mappings:
            if rmsd > self.rmsd_error_threshold:
                raise RuntimeError("RMSD exceeded threshold for mapping")

    def setup_minimization(self, crystal, working_directory, reorder_atoms_if_high_rmsd=False, reorder_method='molecular_axis'):
        """ Setup the minimization in the working directory 
        
        Args:
            crystal (cspy.Crystal): The crystal to minimize
            working_directory (str): The directory to run the minimization
            reorder_atoms_if_high_rmsd (bool): If True, reorder atoms if the RMSD of multipole matching is high
            reorder_method (str): The method to use for reordering atoms if the RMSD of multipole matching is high ('rdkit' or 'molecular_axis')

        Returns:
            None        
        """
        LOG.debug("Setting up minimization in %s", working_directory)
        res = join(working_directory, self.res_filename)
        crystal.save(res)
        assert exists(res)

        axis = Path(working_directory, self.axis_filename)
        axis.write_text(self.axis)
        LOG.debug("wrote axis: to %s\n%s", self.axis_filename, axis)

        nc_axis = crystal.neighcrys_setup(
            potential_type=self.potential_type, file_content=self.axis
        )

        bonds = Path(working_directory, "bondlengths")
        if "bondlength_cutoffs" in crystal.properties:
            bondlengths = (
                "\n".join(
                    "  ".join((a.ljust(2, "_"), b.ljust(2, "_"), str(d)))
                    for (a, b), d in crystal.properties["bondlength_cutoffs"].items()
                )
                + "\nENDS\n"
            )
        else:
            bondlengths = self.bondlengths
        bonds.write_text(bondlengths)
        assert bonds.exists()

        pots = join(working_directory, self.potential_basename)
        elements = set(x.symbol for x in crystal.asymmetric_unit.elements)
        pot_data = PotentialData(self.potential)
        Path(pots).write_text(pot_data.neighcrys_string_form(elements=elements))
        assert exists(pots)

        mults = Path(working_directory, self.multipole_filename)
        reduction = 0.1 if self.kwargs.get("foreshorten_hydrogens") else None
        crystal.map_multipoles(self.multipoles,
                               foreshorten_hydrogens=reduction,
                               reorder_atoms_if_high_rmsd=reorder_atoms_if_high_rmsd,
                               reorder_method=reorder_method)
        if self.reorder_atoms_if_high_rmsd:
            crystal.save(res)
        adjusted_mults = DistributedMultipoles.from_molecules(
            crystal.symmetry_unique_molecules()
        )
        LOG.debug("Adjusted mults:\n%s", adjusted_mults.to_dma_string())
        mults.write_text(adjusted_mults.to_dma_string())
        assert exists(mults)

        if "hessian" in crystal.properties:
            with open(working_directory + '/dmahessian', 'w') as h:
                h.writelines(crystal.properties["hessian"])

    def run_neighcrys(self, working_directory):
        exe = Neighcrys(
            self.res_filename,
            potential_type=self.potential_type,
            axis_filename=self.axis_filename,
            multipole_filename=self.multipole_filename,
            name=self.name,
            working_directory=working_directory,
            potential_filename=self.potential_basename,
            keep_files = self.keep_files,
            **self.kwargs
        )

        timing = Timer()
        with timing:
            exe.run()

        success = exe.dmain_contents is not None
        LOG.debug(
            "NEIGHCRYS setup %s in %.2fs",
            "succeeded" if success else "failed",
            timing.elapsed,
        )
        return exe

    def run_dmacrys(self, nc, working_directory):
        exe = Dmacrys(
            nc.dmain_contents,
            nc.symmetry_contents,
            name=self.name,
            working_directory=working_directory,
            keep_files = self.keep_files,
            **self.kwargs
        )
        timing = Timer()
        with timing:
            returncode = exe.run_checked()
        success = exe.shelx_contents is not None
        LOG.debug(
            "DMACRYS minimization %s in %.2fs",
            "succeeded" if success else "failed",
            timing.elapsed,
        )
        if not success:
            LOG.debug(
                "Last 20 lines in DMACRYS output: %s\n",
                "\n".join(exe.output_contents.splitlines()[-20:]),
            )
            LOG.debug("DMACRYS input:\n%s", exe.input_contents)
        return exe
    
    def _directory_minimize(self, crystal, directory):
        try:
            self.setup_minimization(crystal,
                                directory,
                                reorder_atoms_if_high_rmsd=self.reorder_atoms_if_high_rmsd,
                                reorder_method=self.reorder_method)
        
            nc = self.run_neighcrys(directory)
        except (ReturnCodeError, TimeoutExpired, ValueError, KeyError, IndexError) as exc:
            LOG.exception("Error in Neighcrys: %s", exc)
            return None
        self._last_ncin = nc.input_contents
        if not nc.dmain_contents:
            LOG.error(
                "Neighcrys problem, \nLast 5 fort.21 lines:%s\n",
                "\n".join(nc.fort21_contents.splitlines()[-5:]),
            )
            return None
        self._last_dmain = nc.dmain_contents
        self._last_symmetry = nc.symmetry_contents
        try:
            dm = self.run_dmacrys(nc, directory)
        except (ReturnCodeError) as exc:
            LOG.exception("Error in Dmacrys: %s", exc, stack_info=False, exc_info=False)
            return None
        except (TimeoutExpired) as exc:
            LOG.exception("Error in Dmacrys: %s", exc, stack_info=False, exc_info=False)
            return 'Timeout'
        except Exception as exc:
            LOG.exception("Unknown error in Dmacrys: %s", exc)
            return None
        if dm.shelx_contents is None:
            LOG.exception("No shelx contents in dmacrys output")
            return None
        self._last_summary = DmacrysSummary(dm.summary_contents)
        self._last_dmaout = DmacrysOutput(dm.output_contents)
        self._last_dmain = dm.input_contents
        if dm.error:
            return dm.error
        if self._last_summary.error:
            return self._last_summary.error
        
        try:
            new_crystal = Crystal.from_shelx_string(dm.shelx_contents)
        except (ValueError) as exc:
            LOG.exception("Error in DMACRYS output: %s", exc)
            return None
        crystal.properties["lattice_energy"] = self.summary.initial_energy
        crystal.properties["density"] = self.summary.initial_density
        new_crystal.properties["initial_energy"] = self.summary.initial_energy
        new_crystal.properties["initial_density"] = self.summary.initial_density
        new_crystal.properties["lattice_energy"] = self.summary.final_energy
        new_crystal.properties["cell_volume"] = self.summary.final_volume
        new_crystal.properties["density"] = self.summary.final_density
        new_crystal.properties["dmacrys_summary"] = self._last_summary
        if hasattr(self.summary, "phonon_frequencies"):
            new_crystal.properties["phonon_frequencies"] \
                = self.summary.phonon_frequencies
            new_crystal.properties["mass_and_moments"] \
                = self.dmaout.mass_and_moments()
            new_crystal.properties["phonon_eigenvectors"] \
                = self.dmaout.phonon_eigenvectors(new_crystal)
            new_crystal.properties["transform_cspy_global"] \
                = self.dmaout.transform_cspy_global(new_crystal)
            
        if 'HESS' in self.kwargs:
            with open(directory + '/dmahessian', 'r') as h:
                new_hessian = h.readlines()
                new_crystal.properties["hessian"] = new_hessian

        return new_crystal

    def minimize(self, crystal):
        LOG.debug('Minimizing crystal "%s"', crystal.titl)
        timing = Timer()

        if not 'HESS' in self.kwargs and "hessian" in crystal.properties:
            del crystal.properties["hessian"]

        with timing:
            if self.keep_files:
                import time
                import random
                import string
                while True:
                    seconds = str(int(time.time())) 
                    letters = string.ascii_lowercase
                    rnd_string = ''.join(random.choice(letters) for i in range(10))
                    workdir = seconds + rnd_string
                    try:
                        os.mkdir(workdir)
                    except:
                        continue
                    break
                LOG.info('Performing DMACRYS minimization in "%s"', workdir)
                new_crystal = self._directory_minimize(crystal, workdir)
            else:
                with TemporaryDirectory(prefix="/dev/shm/") as tmpdirname: 
                    LOG.debug('Performing DMACRYS minimization in "%s"', tmpdirname)
                    new_crystal = self._directory_minimize(crystal, tmpdirname)
                
        LOG.debug(
            'Minimization of "%s" complete in %.2fs', crystal.titl, timing.elapsed
        )
        return new_crystal


    @property
    def neighcrys_input(self):
        return self._last_ncin

    @property
    def symmetry_contents(self):
        return self._last_symmetry

    @property
    def dmain_contents(self):
        return self._last_dmain

    @property
    def dmaout_contents(self):
        return self._last_dmaout.contents

    @property
    def summary_contents(self):
        return self._last_summary.contents

    @property
    def dmaout(self):
        return self._last_dmaout

    @property
    def summary(self):
        return self._last_summary

    def __repr__(self):
        return "DmacrysMinimizer(for='{}')".format(self.name)

    def __call__(self, crystal):
        return self.minimize(crystal)

    @classmethod
    def from_axis_and_dma_files(cls, axis_filename, dma_filename, **kwargs):
        from cspy.chem.multipole import DistributedMultipoles

        axis = Path(axis_filename).read_text()
        mults = DistributedMultipoles.from_dma_file(dma_filename)
        return cls(mults, axis, **kwargs)
