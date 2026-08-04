import logging
import time
from collections import namedtuple
from enum import IntEnum
from cspy.chem.multipole import DistributedMultipoles
from cspy.crystal import Crystal
from cspy.db.key import CspDatabaseId
from cspy.minimize import CompositeMinimizer
from cspy.minimize import single_point_evaluation
from cspy.ml.descriptors import calculate_crystal_powder_pattern
from .worker import Worker
from cspy.configuration import CspyConfiguration
from cspy.crystal.util import find_formula_unit

LOG = logging.getLogger(__name__)
CONFIG = CspyConfiguration()

MinimizedStructure = namedtuple(
	"MinimizedStructure",
	"id spacegroup trial_number filename energy density file_content xrd time minimization_step",
)

MC_MinimizedStructure = namedtuple(
	"MC_MinimizedStructure",
	"name id spacegroup trial_number minimization_step mc_step unique_index "
	"energy density file_content initial_res xrd time accept",
)


class ReoptTaskTag(IntEnum):
	OPT = 1
	TA_OPT = 2

class ReoptWorker(Worker):

    def __init__(self, data):
        """The Reoptimization worker class.

        Parameters
        ----------
        data: dict
            Dictionary containing all the data required to run the
            structure generation or minimisation tasks.
        """
        super().__init__()
        self.data = data

    def calculate(self, data):
        """Override of the calculate method for Reoptimisation tasks.

        Parameters
        ----------
        data: tuple
            Tuple containing the task tag to be done and args for
            that task.

        Returns
        -------
        tuple
            Tuple of the results which contains the task tag, results
            and time taken.
        """
        task, args = data
        results = None
        if task == ReoptTaskTag.OPT:
            results = self.optimize_structure(args)

        if task == ReoptTaskTag.TA_OPT:
            LOG.info("optimizing threshold structure: %s", args.id)
            results = self.optimize_threshold_structure(args)

        return results

    def optimize_structure(self, structure):
        t1 = time.time()
        from cspy.configuration import configure

        name = structure.id
        sg = structure.spacegroup
        Zp = self.data["Zp"]
        if Zp == "calculate":
            c = Crystal.from_shelx_string(structure.file_content)
            mols = c.symmetry_unique_molecules()
            _, _, Zp = find_formula_unit(mols)
        crystal_info = {"spacegroup" : sg, "Zp" : Zp}
        kwargs = {}
        kwargs["crystal_info"] = crystal_info

        kwargs["charges"] = None
        if self.data["charges"] is not None:
            kwargs["charges"] = DistributedMultipoles.from_dma_string(
                self.data["charges"]
            )            

        kwargs["multipoles"] = None
        if self.data["multipoles"] is not None:
            kwargs["multipoles"] = DistributedMultipoles.from_dma_string(
                self.data["multipoles"]
            )

        kwargs["axis"] = None
        if self.data["axis"] is not None:
            kwargs["axis"] = self.data["axis"]

        if self.data["bondlength_cutoffs"] is not None:
            kwargs["bondlength_cutoffs"] = self.data["bondlength_cutoffs"]
        kwargs["keep_files"] = self.data["keep_files"]
        minimization_settings = self.data.get("minimization", {})
        check_spe = self.data["check_single_point_energy"]
        configure(minimization_settings)

        structure_filename = structure.filename
        structure_spacegroup = structure.spacegroup
        structure_seed = structure.trial_number
        if hasattr(structure, 'energy'):
            energy = structure.energy
        else:
            energy = None
        structure = Crystal.from_shelx_string(structure.file_content)
        structure.properties["lattice_energy"] = energy
        structure.properties["Zp"] = self.data["Zp"]
        structure_id = CspDatabaseId.from_components(name, "OPT", 0)
        res = structure.to_shelx_string(titl=structure_id)

        crystals = [
            MinimizedStructure(
                id=structure_id,
                spacegroup=structure_spacegroup,
                trial_number=structure_seed,
                filename=structure_filename,
                energy=float("nan"),
                density=float("nan"),
                time=float("nan"),
                xrd=None,
                file_content=res,
                minimization_step=0,
            )
        ]

        minimizer = CompositeMinimizer.from_defaults(**kwargs)
        minimized_crystals = minimizer(structure)
        valid = False

        for minimization_step, crystal in enumerate(
            minimized_crystals, start=1):
            if isinstance(crystal, str):
                valid = crystal
            else:
                structure_id = CspDatabaseId.from_components(
                    name, "OPT", minimization_step)
                mtime = crystal.properties["minimization_time"]
                xrd = None
                # try-except block below is currently used to distinquish between dftb/vasp and dmacrys opt
                try:
                    energy = crystal.properties["lattice_energy"]
                    density = crystal.properties["density"]
                except: 
                    energy = crystal.properties["final_energy"]
                    density = crystal.density
                res = crystal.to_shelx_string(titl=f"{structure_id} {energy} {density}")
                if minimization_step == minimizer.step_count:
                    check_through = True
                    if check_spe:
                        energy_check, _ = single_point_evaluation(
                            crystal,
                            kwargs["multipoles"],
                            kwargs["axis"],
                            potential=self.data["minimization"]["neighcrys"]["potential"],
                            bondlength_cutoffs=kwargs["bondlength_cutoffs"],
                            name=name,
                            **self.data["minimization"]["dmacrys"],
                        )
                        if abs(energy - energy_check) > 10.0:
                            check_through = False
                            LOG.info(
                                "Different energy from single point "
                                "evaluation id %s, energy %s, energy_check %s",
                                structure_id, energy, energy_check
                            )
                    if check_through:
                        valid = True
                        xrd = calculate_crystal_powder_pattern(
                            crystal, self.data["descriptors"].get("pxrd")
                        )

                crystals.append(
                    MinimizedStructure(
                        id=structure_id,
                        spacegroup=structure_spacegroup,
                        trial_number=structure_seed,
                        filename=structure_filename,
                        energy=energy,
                        density=density,
                        time=mtime,
                        xrd=xrd,
                        file_content=res,
                        minimization_step=minimization_step,
                    )
                )

        return ReoptTaskTag.OPT, (valid, crystals), time.time() - t1

    def optimize_threshold_structure(self, structure):
        t1 = time.time()
        from cspy.configuration import configure

        filename = structure.name
        name = structure.id
        name, task_type, sg, trial_num, original_minimization_step, mc_step = name.split("-")

        kwargs = {}
        if self.data["charges"] is not None:
            kwargs["charges"] = DistributedMultipoles.from_dma_string(
                self.data["charges"]
            )
        if self.data["multipoles"] is not None:
            kwargs["multipoles"] = DistributedMultipoles.from_dma_string(
                self.data["multipoles"]
            )
        if self.data["axis"] is not None:
            kwargs["axis"] = self.data["axis"]
        if self.data["bondlength_cutoffs"] is not None:
            kwargs["bondlength_cutoffs"] = self.data["bondlength_cutoffs"]
        minimization_settings = self.data.get("minimization", {})
        check_spe = self.data["check_single_point_energy"]
        configure(minimization_settings)

        minimizer = CompositeMinimizer.from_defaults(**kwargs)

		#structure_filename = structure.filename
		#structure_spacegroup = structure.spacegroup
        trial_number = structure.trial_number
        u_index = structure.unique_index
        ini_res = structure.file_content
        accept = structure.accept
        structure = Crystal.from_shelx_string(structure.file_content)
        structure_id = CspDatabaseId.from_components(
            name, task_type, sg, trial_num, 0, mc_step
        )
        res = structure.to_shelx_string(titl=structure_id)

        crystals = [
            MC_MinimizedStructure(
                name=filename,
                id=structure_id,
                spacegroup=sg,
                trial_number=trial_number,
                minimization_step=0,
                mc_step=mc_step,
                unique_index=u_index,
                energy=float("nan"),
                density=float("nan"),
                file_content=res,
                initial_res=ini_res,
                xrd=None,
                time=float("nan"),
                accept=accept,
            )
        ]

        minimized_crystals = minimizer(structure)
        valid = False

        for minimization_step, crystal in enumerate(minimized_crystals, start=1):
            structure_id = CspDatabaseId.from_components(
                name, task_type, sg, trial_num, minimization_step, mc_step
            )
            energy = crystal.properties["lattice_energy"]
            density = crystal.properties["density"]
            mtime = crystal.properties["minimization_time"]
            res = crystal.to_shelx_string(
                titl=f"{structure_id} {energy} {density}")
            xrd = None

            if minimization_step == minimizer.step_count:
                check_through = True
                if check_spe:
                    energy_check, _ = single_point_evaluation(
                        crystal,
                        kwargs["multipoles"],
                        kwargs["axis"],
                        potential=self.data["minimization"]["neighcrys"]["potential"],
                        bondlength_cutoffs=kwargs["bondlength_cutoffs"],
                        name=name,
                        **self.data["minimization"]["dmacrys"],
                    )
                    if abs(energy - energy_check) > 10.0:
                        check_through = False
                        LOG.info(
                            "Different energy from single point "
                            "evaluation id %s, energy %s, energy_check %s",
                            structure_id, energy, energy_check
                        )
                if check_through:
                    valid = True
                    xrd = calculate_crystal_powder_pattern(
                        crystal, self.data["descriptors"].get("pxrd")
                    )

            crystals.append(
                MC_MinimizedStructure(
                    name=filename,
                    id=structure_id,
                    spacegroup=sg,
                    trial_number=trial_number,
                    minimization_step=minimization_step,
                    mc_step=mc_step,
                    unique_index=u_index,
                    energy=energy,
                    density=density,
                    file_content=res,
                    initial_res=ini_res,
                    xrd=xrd,
                    time=mtime,
                    accept=accept,
                )
            )

        return ReoptTaskTag.TA_OPT, (valid, crystals), time.time() - t1
