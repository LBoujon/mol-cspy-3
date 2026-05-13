import logging
from cspy.executable import Gulp, ReturnCodeError, TimeoutExpired
from cspy.potentials import available_potentials, PotentialData
from cspy.crystal import Crystal
from cspy.chem.multipole import DistributedMultipoles
import sys
import numpy as np
from cspy.crystal.util import find_formula_unit
from cspy.util.timer import Timer
from cspy.util.path import Path
from cspy.util.constants import EV2KJ_PER_MOL
from tempfile import TemporaryDirectory
import os
import sys
from typing import Tuple, Union, Optional
import shutil

LOG = logging.getLogger("gulp_minimizer")

class GulpMinimizer:
    # does not include opti or mole as these are hardcoded
    _supported_keywords = (
        "conj", # conjugate gradient minimisation
        "lbfgs", # LBFGS minimisation
        "rfo", # rational function optimisation minimisation
        "unit", # BFGS minimisation start from a unit Hessian
        "force_minimisation", # minimise force, not energy
        "rigid", # rigid molecules
        "cellonly", # fix molecules, optimise cell parameters
        "fix_molecule", # fix molecule connectivity
        "nosymmetry" # ignore symmetry
    )

    _supported_options = (
        "stepmx", # max step size taken during a GO iteration
        "maxcyc", # max number of GO iterations
        "xtol", # distance tolerance
        "ftol", # force tolerance
        "gtol", # gradient tolerance
        "gmax", # max allowed gradient
        "accuracy" # accuracy of ewald summation
    )

    def __init__(
        self,
        multipoles : Optional[str] = None,
        axis : Optional[str] = None,
        reorder_atoms_if_high_rmsd : Optional[bool] = False,
        reorder_method : Optional[str] = 'molecular_axis',
        name : Optional[str] = 'Unknown',
        potential: Optional[str] = 'gfn-ff',
        keep_files : Optional[bool] = False,
        **kwargs
    ):
        """Gulp minimizer class for performing geometry optimisations
        and singlepoints with GULP.

        Parameters
        ----------
        contents : string
            string containing the contents of the gulp standard out
        parse : bool
            whether or not to parse the output
        singlepoint : bool
            whether or not the output is from a singlepoint calculation

        """
        self.charges = multipoles 
        self.axis = axis
        self.reorder_atoms_if_high_rmsd = reorder_atoms_if_high_rmsd
        self.reorder_method = reorder_method
        self.name = name
        self.kwargs = kwargs
        self.keep_files = keep_files
        self.potential = potential
        self.temp_prefix = kwargs.pop("temp_prefix", "/dev/shm/")
        if not self.is_gfnff:
            if potential in available_potentials.keys():
                self.potential_type, self.potential_filename = available_potentials[potential]
            else:
                pot_path = Path(potential + '.lib')
                try:
                    assert pot_path.exists()
                    self.potential_filename = potential + '.lib'
                    self.potential_type = "Unknown"
                    #raise Exception("No valid potential")
                except AssertionError as e:
                    LOG.exception('Cannot find %s in mol-CSPy potential dictionary. Nor could an associated .lib file be found locally.', potential)
                    sys.exit()

        self._last_input = None
        self._last_output = None
        self._last_cif = None
        self.singlepoint = False

    @property
    def input_file(self):
        return "{}.gin".format(self.name)

    @property
    def cif_filename(self):
        return "{}.cif".format(self.name)

    @property
    def potential_basename(self):
        return Path(self.potential_filename).with_suffix(".lib").name

    @property
    def is_gfnff(self):
        return self.potential.lower() in ("gfn-ff", "gfnff", "gfn")

    def _gfnff_input_contents(self, crystal : Crystal, **kwargs) -> None:
        """Setup input for gfnff

        Parameters
        ----------
        crystal : Crystal
            CSPy Crystal object
        parse : bool
            whether or not to parse the output
        singlepoint : bool
            whether or not the output is from a singlepoint calculation

        """
        conp = "conp"
        opt = "opti"
        header = f"{opt} {conp} gfnff prop phon gwolf noden"
        params = " ".join(f"{x:12.6f}" for x in crystal.unit_cell.parameters)
        cell = f"cell\n{params}"
        atom_lines = []
        crystal.neutralise_cell_charge()
        asym = crystal.asymmetric_unit
        for i in range(len(asym)):
            a, b, c = asym.positions[i, :]
            atom_lines.append(f"{asym.elements[i]} core {a:12.8f} {b:12.8f} {c:12.8f}")
        atoms = "frac\n" + "\n".join(atom_lines) + "\n"
        sg = f"space\n{crystal.space_group.international_tables_number}"
        return "\n".join((
            header, cell, atoms, sg,
            f"output cif {self.cif_filename}\noutput drv {self.name}.drv"
        ))
    

    def _input_keywords(self, settings : str) -> str:
        """ Configure keyword settings for GULP input.
        See GULP's help.html file for more info.

        Parameters
        ----------
        settings : string
            WIP string of settings for GULP input file (.gin)

        Returns
        -------
        settings : string
            WIP string of settings for GULP input file (.gin)
        """
        # we will leave this on by default for now
        settings += " kjmol"
        for k in self._supported_keywords:
            if k in self.kwargs:
                if self.kwargs[k] == "True":
                    settings += " " + k

        settings += " prt_var\n"
        return settings


    def _input_options(self, settings : str) -> str:
        """ Configure options settings for GULP input.
        See GULP's help.html file for more info.

        Parameters
        ----------
        settings : string
            WIP string of settings for GULP input file (.gin)

        Returns
        -------
        settings : string
            WIP string of settings for GULP input file (.gin)
        """
        settings += "cutp 15 mdf 2\n"
        #settings += "cutp 30 mdf 4\n"
        for k in self._supported_options:
            if k =="maxcyc" and k in self.kwargs:
                if self.kwargs[k] == "0":
                    settings = settings.replace('opti', '')
                    self.singlepoint = True
                else:
                    settings += "maxcyc " + self.kwargs[k] + "\n"
            elif k in self.kwargs:
                settings += k + " " + self.kwargs[k] + "\n"

        # for debug purposes
        #settings += "dump every 1 gulp.res"

        return settings
    

    def get_rigid_molecule_atoms(self, crystal : Crystal) -> list:
        """ 
        Extract the lines needed to define the atoms for a GULP input
        file from a mol-CSPy crystal object. Define atoms for a 
        rigid potential.

        Parameters
        ----------
        crystal : Crystal
            Crystal represented as a mol-CSPy Crystal object to
            be converted to a GULP input file.

        Returns
        -------
        atom_lines : list of strings
            Lines that define the atoms for the GULP input file
        """
        atom_lines = []
        for mol in crystal.symmetry_unique_molecules():
            frac_pos = crystal.to_fractional(mol.positions)
            crystal.neutralise_cell_charge()
            charges = [x.charge for x in mol.multipoles]
            potential_labels = mol.properties["neighcrys_potential_label"]

            for el, p, (a, b, c), charge in zip(
            mol.elements,
            potential_labels,
            frac_pos,
            charges,
            ):
                atom_lines.append(f"{el}{p[1]} core {a:12.8f} {b:12.8f} {c:12.8f} {charge:12.8f}")

        return atom_lines
    

    def get_flexible_molecule_atoms(self, crystal : Crystal)  -> list:
        """ 
        Extract the lines needed to define the atoms for a GULP input
        file from a mol-CSPy crystal object. Define atoms for a 
        flexible potential.
        
        Parameters
        ----------
        crystal : Crystal
            Crystal represented as a mol-CSPy Crystal object to
            be converted to a GULP input file.

        Returns
        -------
        atom_lines : list of strings
            Lines that define the atoms for the GULP input file
        """
        atom_lines = []
        GULP_labels_types = dict()
        for mol in crystal.symmetry_unique_molecules():
            frac_pos = crystal.to_fractional(mol.positions)
            crystal.neutralise_cell_charge()
            charges = [x.charge for x in mol.multipoles]
            atom_labels = mol.properties["GULP_atom_labels"]
            atom_types = mol.properties["potential_atom_types"]

            for g, p, (a, b, c), charge in zip(
            atom_labels,
            atom_types,
            frac_pos,
            charges,
            ):
                atom_lines.append(f"{g} core {a:12.8f} {b:12.8f} {c:12.8f} {charge:12.8f}")
                if not g in GULP_labels_types.keys():
                    GULP_labels_types[g] = p

        species_lines = ["\nSpecies"]
        for label in GULP_labels_types.keys():
            species_lines.append(label + ' core ' + GULP_labels_types[label])

        atom_lines += species_lines

        return atom_lines


    def input_contents(self, crystal : Crystal, flexible : Optional[bool] = False) -> None:
        """ Setup up contents for GULP input file (.gin) for GULP
        geometry optimisation and singlepoints.

        Parameters
        ----------
        crystal : Crystal
            CSPy Crystal object

        Returns
        -------
         : string
            Contents for GULP input file (.gin) for GULP
        geometry optimisation and singlepoints.
        """
        if self.is_gfnff:
            return self._gfnff_input_contents(crystal)
        settings = f"opti molecule unfix conp"
        #settings = f"opti molecule conp c6"
        settings = self._input_keywords(settings)
        settings = self._input_options(settings) 
        settings += f"\ntitle\n{self.name}\nend\n" #cap off the settings

        params = " ".join(f"{x:12.6f}" for x in crystal.unit_cell.parameters)
        cell = f"cell\n{params}"
        atom_lines = []
        crystal.neutralise_cell_charge()
        for mol in crystal.symmetry_unique_molecules():
            potential_labels = mol.properties["neighcrys_potential_label"]
            frac_pos = crystal.to_fractional(mol.positions)
            charges = [x.charge for x in mol.multipoles]
            for el, p, (a, b, c), charge in zip(
                mol.elements,
                potential_labels,
                frac_pos,
                charges,
            ):
                atom_lines.append(f"{el}{p[1]} core {a:12.8f} {b:12.8f} {c:12.8f} {charge:12.8f}")
        atoms = "frac\n" + "\n".join(atom_lines) + "\n"
        if flexible:
            atom_lines = self.get_flexible_molecule_atoms(crystal)
        else:
            atom_lines = self.get_rigid_molecule_atoms(crystal)        

        atoms = "frac\n" + "\n".join(atom_lines) + "\n"

        #sg = f"space\n{crystal.space_group.international_tables_number}"
        sg = f"space\n{crystal.space_group.gulp_full_symbol()}"
        return "\n".join((
            settings, cell, atoms, sg, "library " + self.potential_basename,
            f"output cif {self.cif_filename}\noutput drv {self.name}.drv"
        ))

    def setup_minimization(self, crystal : Crystal, working_directory : str) -> None:
        """ Setup potentials and GULP input file (.gin) for GULP geometry optimisation
        and singlepoints.

        Parameters
        ----------
        crystal : Crystal
            CSPy Crystal object

        working_directory : string
            Directory to perform calculation in
        """
        LOG.debug("Setting up minimization in %s", working_directory)

        if not self.is_gfnff:
            pot_dest = Path(working_directory, self.potential_basename)
            if '.lib' in self.potential_filename:
                pot_orig = Path(self.potential_basename)
            else:
                pot_orig = available_potentials[self.potential][1]
            file_ext = str(pot_orig).split('.')[-1]
            # if potential is in DMACRYS format, we must convert for GULP
            if file_ext == 'pots':
                elements = set(x.symbol for x in crystal.asymmetric_unit.elements)
                pot_data = PotentialData(self.potential)
                pot_dest.write_text(pot_data.gulp_string_form(elements=elements))
                crystal.neighcrys_setup(
                    potential_type=self.potential_type, file_content=self.axis
                )
                crystal.map_multipoles(self.charges,
                                    reorder_atoms_if_high_rmsd=self.reorder_atoms_if_high_rmsd,
                                    reorder_method=self.reorder_method)
                flexible=False
            else:
                # this glob is a temporary development solution
                # we should get a better solution
                from glob import glob
                typing_files = glob('*.typing')
                crystal.assign_ff_atom_typing(typing_files[0], get_GULP_labels=True)
                shutil.copyfile(pot_orig, pot_dest)
                flexible = True
            assert pot_dest.exists()                

        self._last_input = self.input_contents(crystal, flexible)

    def run_gulp(self, working_directory : str) -> Gulp:
        """ Setup GULP executable and run it.

        Parameters
        ----------
        working_directory : string
            Directory to perform calculation in

        Returns
        -------
        exe : string
            GULP executable
        """
        exe = Gulp(
            self._last_input,
            name=self.name,
            working_directory=working_directory,
            **self.kwargs
        )
        timing = Timer()
        with timing:
            returncode = exe.run()
        self.minimization_time = timing.elapsed
        success = exe.output_contents is not None
        LOG.info(
            "Gulp minimization %s in %.2fs",
            "completed" if success else "failed",
            timing.elapsed,
        )
        if not success:
            LOG.debug(
                "Last 20 lines in Gulp output: %s\n",
                "\n".join(exe.output_contents.splitlines()[-20:]),
            )
            LOG.debug("Gulp input:\n%s", exe.input_contents)
        return exe
    

    def _directory_minimize(self, crystal : Crystal, directory : str):
        """ Perform a GULP minimization or singlepoint inside
        a provided directory.

        Parameters
        ----------
        crystal : Crystal
            CSPy Crystal object

        working_directory : string
            Directory to perform calculation in

        Returns
        -------
        new_crystal : Crystal
            CSPy Crystal object
        """
        from cspy.formats.gulp_output import GulpOutput
        self.setup_minimization(crystal, directory)
        try:
            gulp = self.run_gulp(directory)
        except (ReturnCodeError, TimeoutExpired) as exc:
            LOG.exception("Error in Gulp: %s", exc, stack_info=False, exc_info=False)
            return 'Timeout'
        except Exception as exc:
            LOG.exception("Unknown error in Gulp: %s", exc)
            return None
        try:
            self._last_output = gulp.output_contents
            last_cif_path = Path(directory, self.cif_filename)
            if not last_cif_path.is_file():
                error = "Unknown"
                LOG.error("Error in Gulp: No cif file returned by output. Something went wrong.")

                return error
            self._last_cif = last_cif_path.read_text()
            self._last_drv = Path(directory, f"{self.name}.drv").read_text()
            go = GulpOutput(self._last_output, singlepoint=self.singlepoint)
            new_crystal = Crystal(
                go.parsed_contents["unit cell"],
                crystal.space_group,
                go.parsed_contents["asymmetric unit"]
            )
            if go.reached_maxcyc and not self.kwargs.get("allow_maxcyc") == True:
                error = "MaxIts"
                LOG.error("Error in Gulp: Total GO iterations (cycles) exceeds max allowed GO iterations (cycles).")

                return error

            # GULP reports energies per unit cell. We want per formula unit
            formula_unit, unique_molecules, Z = find_formula_unit(new_crystal.unit_cell_molecules())
            energy = float(self._last_drv.splitlines()[0].split()[1]) * EV2KJ_PER_MOL / Z
        except Exception as exc:
            LOG.exception("Error reading Gulp outputs: %s", exc)
            return None
        new_crystal.properties["lattice_energy"] = energy
        new_crystal.properties["density"] = new_crystal.density
        new_crystal.properties["minimization_time"] = self.minimization_time

        return new_crystal


    def minimize(self, crystal : Crystal):
        """ Decide where to perform a GULP minimization or singlepoint
        and then run a minimization in that location via _directory_minimize

        Parameters
        ----------
        crystal : Crystal
            CSPy Crystal object

        Returns
        -------
        new_crystal : Crystal
            CSPy Crystal object
        """
        LOG.debug('Minimizing crystal "%s"', crystal.titl)
        timing = Timer()
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
                LOG.info('Performing GULP minimization in "%s"', workdir)
                new_crystal = self._directory_minimize(crystal, workdir)
            else:
                with TemporaryDirectory(prefix=self.temp_prefix) as tmpdirname:
                    LOG.debug('Performing GULP minimization in "%s"', tmpdirname)
                    new_crystal = self._directory_minimize(crystal, tmpdirname)

        LOG.debug(
            'Minimization of "%s" complete in %.2fs', crystal.titl, timing.elapsed
        )
        return new_crystal

    def __repr__(self):
        return "GulpMinimizer(for='{}')".format(self.name)

    def __call__(self, crystal : Crystal):
        return self.minimize(crystal)
