import copy
import logging
import time
import numpy as np
from collections import namedtuple
from enum import IntEnum
from cspy import Molecule
from cspy.chem.multipole import DistributedMultipoles
from cspy.crystal import Crystal
from cspy.db.key import CspDatabaseId
from cspy.minimize import CompositeMinimizer
from cspy.minimize import single_point_evaluation
from cspy.ml.descriptors import calculate_crystal_powder_pattern
from cspy.sample.mc.mc_change import MC
from cspy.cspympi.cspy_tasks import GeneratedStructure, CSPyWorker, CSPyTaskTag
from .worker import Worker


LOG = logging.getLogger(__name__)


PerturbedStructure = namedtuple(
    "PerturbedStructure", "name spacegroup trial_number mc_step file_content"
)
MC_MinimizedStructure = namedtuple(
    "MC_MinimizedStructure",
    "name id spacegroup trial_number minimization_step mc_step unique_index "
    "energy density file_content initial_res xrd time accept molecule_id",
)


class QRBHWorker(CSPyWorker):
    def __init__(self, data):
        """The CSP worker class.

        Parameters
        ----------
        data: dict
            Dictionary containing all the data required to run the
            structure generation or minimisation tasks.
        """
        super().__init__(data)
        self.task_name = "QRBH"

    def calculate(self, data):
        """Override of the calculate method for CSPy tasks.

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

        if task == CSPyTaskTag.QR:
            results = self.structure_generation(args)
        elif task == CSPyTaskTag.OPT:
            results = self.minimize_structure(args)
        elif task == CSPyTaskTag.PER_OPT:
            results = self.perturb_and_minimize(args)

        return results

    def minimize_structure(self, structure):
        from cspy.configuration import configure

        t1 = time.time()
        if isinstance(structure, list):
            structure = structure[0]
        if type(structure).__name__ == "GeneratedStructure":
            mc_step = 0
        elif type(structure).__name__ == "PerturbedStructure":
            mc_step = structure.mc_step

        name = structure.name
        sg = structure.spacegroup
        Zp = self.data["Zp"]
        crystal_info = {"spacegroup" : sg, "Zp" : Zp}
        trial_number = structure.trial_number
        ini_res = structure.file_content
        u_index = float("nan")

        charges = DistributedMultipoles.from_dma_string(self.data["charges"])
        multipoles = DistributedMultipoles.from_dma_string(self.data["multipoles"])
        axis = self.data["axis"]
        bondlength_cutoffs = self.data["bondlength_cutoffs"]
        minimization_settings = self.data["minimization"]
        check_spe = self.data["check_single_point_energy"]
        configure(minimization_settings)

        minimizer = CompositeMinimizer.from_defaults(
            charges=charges, axis=axis, multipoles=multipoles,
            bondlength_cutoffs=bondlength_cutoffs, keep_files=self.data["keep_files"], 
            crystal_info = crystal_info
        )

        structure = Crystal.from_shelx_string(structure.file_content)
        structure_id = CspDatabaseId.from_components(
            name, self.task_name, sg, trial_number, 0, mc_step
        )
        res = structure.to_shelx_string(titl=structure_id)
        crystals = [
            MC_MinimizedStructure(
                name=name,
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
                accept=False,
                molecule_id=name
            )
        ]
        minimized_crystals = minimizer(structure)
        valid = False

        for minimization_step, crystal in enumerate(minimized_crystals, start=1):
            if isinstance(crystal, str):
                valid = crystal
            else:
                structure_id = CspDatabaseId.from_components(
                    name, self.task_name, sg, trial_number, minimization_step, mc_step
                )
                energy = crystal.properties["lattice_energy"]
                density = crystal.properties["density"]
                mtime = crystal.properties["minimization_time"]
                res = crystal.to_shelx_string(titl=f"{structure_id} {energy} {density}")
                xrd = None

                if minimization_step == minimizer.step_count:
                    check_through = True
                    if check_spe:
                        energy_check, _ = single_point_evaluation(
                            crystal,
                            multipoles,
                            axis,
                            bondlength_cutoffs=bondlength_cutoffs,
                            potential=self.data["minimization"]["neighcrys"]["potential"],
                            name=name,
                            **self.data["minimization"]["dmacrys"],
                        )
                        if abs(energy - energy_check) > 10.0:
                            check_through = False
                            LOG.info(
                                "Different energy from single point evaluation id %s, "
                                "energy %s, energy_check %s",
                                structure_id, energy, energy_check
                            )
                    if check_through:
                        valid = True
                        if self.data["mc"]["niggli_s"]:
                            crystal = crystal.standardized()
                            res = crystal.to_shelx_string(titl=f"{structure_id} {energy} {density}")
                        xrd = calculate_crystal_powder_pattern(
                            crystal, self.data["descriptors"].get("pxrd")
                        )

                crystals.append(
                    MC_MinimizedStructure(
                        name=name,
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
                        accept=False,
                        molecule_id=name,
                    )
                )
        return CSPyTaskTag.OPT, (valid, crystals), time.time() - t1

    def perturb_structure(self, args, tol=1e-3):
        sg, seed, mc_step, name, res_in = args
        mc_control = MC(
            self.data["mc"]["move"],
            self.data["mc"]["auto_prob"],
            self.data["mc"]["auto_cutoff"],
            all_s=self.data["mc"]["move_all"],
            sat_s=self.data["mc"]["sat_expand"],
        )
        mc_control.update_sg(res_in)
        while True:
            res_new, move = mc_control.mc_move(res_in)
            crys_tmp = Crystal.from_shelx_string(res_new)
            molecules = crys_tmp.symmetry_unique_molecules()
            if sum(len(x) for x in molecules) != len(crys_tmp.asymmetric_unit):
                continue
            valid = True
            for mol in molecules:
                for bond in self.data["mc"]["bonds"]:
                    try:
                        if np.allclose(mol.bonds.todense(), bond, atol=tol):
                            break
                    except ValueError:
                        continue
                else:
                    valid = False
            if valid:
                return PerturbedStructure(
                    name=name,
                    trial_number=seed,
                    spacegroup=sg,
                    mc_step=mc_step,
                    file_content=res_new,
                )
    
    def perturb_and_minimize(self, args):
        t1 = time.time()
        structure = self.perturb_structure(args)
        if not structure.file_content:
            n = float("nan")
            return (
                [
                    MC_MinimizedStructure(
                        name="",
                        id=None,
                        file_content="",
                        spacegroup=structure.spacegroup,
                        trial_number=structure.trial_number,
                        energy=n,
                        density=n,
                        time=n,
                        xrd=None,
                        minimization_step=-1,
                        mc_step=0,
                        initial_res="",
                        accept=False,
                        unique_index=n,
                        molecule_id=""
                    )
                ],
                False,
                time.time() - t1,
            )
        task, results, mtime = self.minimize_structure(structure)
        return task, results, time.time() - t1
    
