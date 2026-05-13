import logging
from cspy.util.timer import Timer
from cspy.crystal.util import crystal_expansions
from cspy.crystal import Crystal

LOG = logging.getLogger(__name__)


class CompositeMinimizer:

    def __init__(self, minimization_steps, minimization_type,
                 bondlength_cutoffs=None, keep_files=False):
        self.minimization_steps = minimization_steps
        self.bondlength_cutoffs = bondlength_cutoffs
        self.minimization_type = minimization_type
        self.keep_files = keep_files

    def minimize(self, crystal):
        results = []
        timer = Timer()
        axis = None
        mults = None
        previous_step = None
        cleanup_files = None
        for i, f in enumerate(self.minimization_steps, start=1):
            step_name = f.__name__ if hasattr(f, "__name__") else f.__class__.__name__
            LOG.debug("Starting minimization step %d: %s", i, step_name)
            with timer:
                #try:
                # remove symmetries and set new axis for expanded crystal
                if "reduction" in self.minimization_type[i - 1]:
                    expansion = crystal_expansions(
                        crystal, self.minimization_type[i - 1][1])
                    crystal = crystal.as_P1_supercell(expansion)
                    crystal.neighcrys_setup(
                        "F" if f.potential == "fit" else "W")
                    axis = crystal.neighcrys_axis.to_string()
                if axis is not None:
                    f.axis = axis

                # maintain the bondlength_cutoffs throughout the minimization
                if self.bondlength_cutoffs is not None:
                    crystal.properties["bondlength_cutoffs"] \
                        = self.bondlength_cutoffs
                
                if i>0 and previous_step == "DftbMinimizer": #if other flex methods added in future this will need editing to account for these
                    LOG.debug("Recalculating multipoles for %s", results[0].titl)                        
                    from cspy.apps.dma import generate_multipoles
                    from cspy.potentials import available_potentials
                    from cspy.chem.multipole import DistributedMultipoles
                    from cspy.configuration import CONFIG
                    from cspy.minimize import DmacrysMinimizer
                    from pathlib import Path
                    from scipy.spatial.distance import pdist
                    import numpy as np
                    import os
                    
                    mf_format = results[0].titl + "{}.dma"
                    mf_suffix = "_rank0"
                    multipole_filename = mf_format.format(mf_suffix)
                    basis_set = CONFIG.get("gaussian.basis")
                    method = CONFIG.get("gaussian.method")
                    potential_type, potential = available_potentials[CONFIG.get("csp.potential")]
                    foreshorten_hydrogens = 0.1 if potential_type == "W" else None
                    mult_kwargs = {"nprocshared": 1,
                                    "mem": "2GB",
                    }
                    work_dir = os.getcwd()
                    os.mkdir("{}".format(results[0].titl)) #.split()[0]?
                    os.chdir("/".join((work_dir, results[0].titl)))

                    #create new axis
                    crystal.neighcrys_setup(potential_type=potential_type)
                    axis_file = Path("{}.mols".format(results[0].titl))
                    axis_file.write_text(crystal.neighcrys_axis.to_string())
                    axis = axis_file.read_text()

                    cleanup_files = generate_multipoles(crystal,
                                                        ranks=[0],
                                                        mf_format=mf_format,
                                                        potential_type=potential_type,
                                                        basis_set=basis_set,
                                                        method=method,
                                                        dma_switch=4,
                                                        foreshorten_hydrogens=foreshorten_hydrogens,
                                                        charges=None,
                                                        multiplicities=None,
                                                        **mult_kwargs)
                    
                    mults = DistributedMultipoles.from_dma_file(multipole_filename)
                    os.chdir(work_dir)
                    LOG.debug("new multipoles generated for %s", results[0].titl)

                    for mol in crystal.symmetry_unique_molecules():
                        cutoff = max(15.0, 1.5 * np.max(pdist(mol.positions)))

                    if CONFIG.get("pilot_timeout"):
                        pilot_timeout = True
                        crystal_info = {"spacegroup" : crystal.space_group.international_tables_number}
                    else:
                        pilot_timeout = False
                    dmacrys_kwargs = {
                        "potential": CONFIG.get("csp.potential"),
                        "timeout": CONFIG.get("dmacrys.timeout"),
                        "MAXI": CONFIG.get("neighcrys.max_iterations"),
                        "vdw_cutoff": cutoff,
                        "pilot_timeout": pilot_timeout,
                        "crystal_info" : crystal_info
                    }
                    f = DmacrysMinimizer(multipoles=mults, 
                                            axis=axis,
                                            keep_files=self.keep_files,
                                            **dmacrys_kwargs)

                crystal = self.minimize_with_sanity_check(f, crystal, step_name)
                if cleanup_files:
                        try:
                            import shutil # alternative is Path().unlink
                            shutil.rmtree("/".join((work_dir, results[0].titl)), ignore_errors=True)
                        except Exception as e:
                            LOG.error("Exception: %s\nCannot remove cleanup files", e)
                       
                # except Exception as e:
                #     LOG.error("Error in minimization step %d: %s: %s", i, step_name, e)
                #     crystal = None
            success = crystal is not None and not isinstance(crystal, str)
            LOG.debug(
                "Finished minimization step %d: %s", i, "OK" if success else "FAIL"
            )
            if not success:
                if isinstance(crystal, str):
                    # 'crystal' is actually a string of an exception
                    results.append(crystal)
                    return results
                break
            crystal.properties["minimization_time"] = timer.elapsed
            results.append(crystal)
            previous_step = step_name #for dftb check (line 43)

        return results

    def minimize_with_sanity_check(self, minimizer, crystal, step):

        new_crystal = minimizer(crystal)
        LOG.debug(f'minimizer is of type - {step}')
        if new_crystal is None or isinstance(new_crystal, str):
            return new_crystal
        elif step == "DftbMinimizer":
            LOG.debug(f'skipping sanity check for {step}')
            return new_crystal

        if self.bondlength_cutoffs is not None:
            new_crystal.properties["bondlength_cutoffs"] = self.bondlength_cutoffs
        # convert crystal to shelx string and back again
        # this ensures it's readable from shelx format and
        # sometimes there are buck cats before the conversion but not after
        # not sure why
        crystal_properties = new_crystal.properties
        new_crystal = Crystal.from_shelx_string(new_crystal.to_shelx_string())
        new_crystal.properties = crystal_properties
        supercell = new_crystal.as_P1_supercell((3, 3, 3))
        if self.bondlength_cutoffs is not None:
            supercell.properties["bondlength_cutoffs"] = self.bondlength_cutoffs

        nuc_mols_sc = len(supercell.unit_cell_molecules())
        if step == "VaspMinimizer":
            # crystals are standardised before being passed to VASP
            # the number of molecules may have fallen during this step
            nuc_mols_crys = len(crystal.standardized().unit_cell_molecules())
        else:
            nuc_mols_crys = len(crystal.unit_cell_molecules())

        if nuc_mols_sc == 27 * nuc_mols_crys:
            return new_crystal
        else:
            LOG.exception(
                "%s - number of molecules in a 3x3x3 supercell of the "
                "minimized crystal (%s) is not equal to 27 times the "
                "number of molecules in the unit cell of the input "
                "crystal (%s) there must be a collision between "
                "molecules after minimization, treating minimization "
                "as unsuccessful", crystal.titl, nuc_mols_sc, 27 * nuc_mols_crys
            ) 

            return "MolClash"

    @property
    def step_count(self):
        return len(self.minimization_steps)

    def __call__(self, crystal):
        return self.minimize(crystal)

    @classmethod
    def from_defaults(cls, **kwargs):
        bondlength_cutoffs = kwargs.pop("bondlength_cutoffs", None)

        minimizers = []
        minimizer_types = []
        from cspy.configuration import CONFIG
        from copy import deepcopy
        from cspy.minimize import MinimizerFactory

        steps = deepcopy(CONFIG["csp_minimization_step"])
        for i, step_conf in enumerate(steps, start=1):
            if not "MAXI" in step_conf:
                step_conf["MAXI"] = CONFIG.get("neighcrys.max_iterations")
            if kwargs.get("crystal_info") and not "crystal_info" in step_conf:
                step_conf["crystal_info"] = kwargs.get("crystal_info")
            if CONFIG.get("dmacrys.pilot_timeout"):
                step_conf["pilot_timeout"] = True
            else:
                step_conf["pilot_timeout"] = False
            kind = step_conf.pop("kind", "dmacrys")
            LOG.debug(
                "Min step %d (%s) %s",
                i,
                kind.upper(),
                ", ".join(f"{k}: {v}" for k, v in step_conf.items()),
            )
            step_conf.update(**kwargs)
            if kind == "dftb":
                dftb_keywords = {k:v for k, v in CONFIG.get("dftb").items()}
                #JG - need to define keywords below for now as if they are 
                #     included in the config file, they seem to break the
                #     command line args for cspy-opt.
                dftb_keywords["single_point"] = False
                dftb_keywords["fixed_lattice_opt"] = True
                dftb_keywords['lattice_opt'] = CONFIG.get("dftb.lattice_opt", False)
                
                if dftb_keywords['lattice_opt']:
                    dftb_keywords['fixed_lattice_opt'] = False
                factory = MinimizerFactory(kind=kind, **dftb_keywords)
            elif kind == "vasp":
                vasp_keywords = {k:v for k, v in CONFIG.get("vasp").items()}
                factory = MinimizerFactory(kind=kind, **vasp_keywords)
            elif kind == "ase":
                ase_keywords = {k:v for k, v in CONFIG.get("ase").items()}
                factory = MinimizerFactory(kind=kind, **ase_keywords)
            else:
                factory = MinimizerFactory(kind=kind)
            required_args = factory.required_args
            if "multipoles" in required_args:
                electrostatics = step_conf.pop("electrostatics", "multipoles")
                step_conf["multipoles"] = kwargs[electrostatics]
            if "potential" not in step_conf:
                step_conf["potential"] = CONFIG.get("neighcrys.potential")

            if kind == "dmacrys" and "reduction" in step_conf:
                target = step_conf.pop("reduction")
                minimizer_types.append(("reduction", target))
            else:
                minimizer_types.append("minimize")

            minimizers.append(factory(**step_conf))
        return cls(minimizers, minimizer_types, bondlength_cutoffs, kwargs.get("keep_files"))





