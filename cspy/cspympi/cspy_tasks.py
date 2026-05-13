import copy
import logging
import time
import random
from mpi4py import MPI
import traceback
from collections import namedtuple
from enum import IntEnum
from cspy import Molecule
from cspy.chem.multipole import DistributedMultipoles
from cspy.crystal import Crystal
from cspy.crystal.generate_crystal import CrystalGenerator
from cspy.db.key import CspDatabaseId
from cspy.minimize import CompositeMinimizer
from cspy.minimize import single_point_evaluation
from cspy.ml.descriptors import PowderPattern
from .worker import Worker
from cspy.util.path import Path
from cspy.db import CspDataStoreFlex
from cspy.potentials import available_potentials
import random
from cspy.apps.dma import (
    generate_combined_name,
    generate_combined_res,
    calculate_equivalent_molecules,
    fit_multipoles_with_mulfit,
)
from os import listdir
from typing import Union, NamedTuple

LOG = logging.getLogger(__name__)


GeneratedStructure = namedtuple(
    "GeneratedStructure", "name spacegroup trial_number file_content"
)
GeneratedStructure_flex = namedtuple(
    "GeneratedStructure_flex",
    "name spacegroup trial_number file_content file_mults file_axis file_charge file_bonds conf_energy molecule_id",
)
MinimizedStructure = namedtuple(
    "MinimizedStructure",
    "id spacegroup trial_number minimization_step energy density file_content xrd time molecule_id",
)


class CSPyTaskTag(IntEnum):
    """
    QR = Quasi-Random Structure Generation
    OPT = Geometry Optimisation
    PER_OPT = Something about QRBH?
    EAU = Extract Asymmetric Unit
    """
    QR = 1
    OPT = 2
    PER_OPT = 3
    EAU = 4


class CSPyWorker(Worker):
    def __init__(self, data: dict) -> None:
        """Initialise the mol-CSPy worker class.

        Args:
            data (dict): A dictionary containing all the data required
            to run the structure generation or minimisation tasks.
        """
        super().__init__()
        self.data = data
        self.task_name = "QR"
        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()

    def calculate(self, data: tuple) -> tuple:
        """Override of the calculate method for CSPy tasks.

        Args:
            data (tuple): A tuple containing the task tag to be done
            and args for that task.

        Returns:
            tuple: A tuple of the results which contains the task tag,
            results and time taken.
        """

        task, args, overrides = data
        results = None

        if task == CSPyTaskTag.QR:
            LOG.debug("Rank " + str(self.rank) + '- QR')
            if 'clg' in overrides.keys():
                clg_alg = overrides['clg']
                if clg_alg == "aut":
                    results = self.structure_generation_aut(args, overrides)
                elif clg_alg == "mps":
                    results = self.structure_generation_mps(args)
                elif clg_alg == "flex":
                    results = self.structure_generation_flex(args)
                elif clg_alg == 'clg2.0' or clg_alg == 'clg2' or clg_alg == 'clg':
                    results = self.structure_generation(args, overrides)
                else:
                    LOG.error("Unrecognised CLG algorithm, %s", clg_alg)
                    return CSPyTaskTag.QR, (args[0], []),0
            else:
                if self.data["aut"]:
                    results = self.structure_generation_aut(args, overrides)
                elif self.data["mps"]:
                    results = self.structure_generation_mps(args)
                elif self.data["flex"]:
                    results = self.structure_generation_flex(args)
                else:
                    results = self.structure_generation(args, overrides)
        elif task == CSPyTaskTag.OPT:
            LOG.debug("Rank " + str(self.rank) + '- OPT')
            results = self.minimize_structure(args, overrides)
        elif task == CSPyTaskTag.EAU:
            LOG.debug("Rank " + str(self.rank) + '- EAU')
            results = self.extract_asymmetric_unit(args, overrides)

        return results

    def check_multipoles_valid_flex(self, xyz_files: list[str], potential: str) -> bool:
        """Checks that the specified multipoles overlay with the crystal created from
        the xyz files

        Args:
            xyz_files (list[str]): A list containing xyz file names
            potential (str): The name of an intermolecular potential

        Returns:
            Bool: True if multipoles valid (i.e. low rmsd) else False
        """
        c = DistributedMultipoles.from_dma_string(self.data["charges"])
        m = DistributedMultipoles.from_dma_string(self.data["multipoles"])
        axis_contents = self.data["axis"]
        success = True
        if len(xyz_files) == 1:
            crys = Crystal.from_xyz_files(xyz_files, titl="tmp")
        else:
            #better to use the strings from the rearranged xyz
            molecules = [Molecule.from_xyz_string(xyz_string) for xyz_string in self.data["asymmetric_unit"]]
            crys = Crystal.from_molecules(molecules, titl="tmp")
        potential_type = available_potentials[potential][0]
        crys.neighcrys_setup(potential_type=potential_type, file_content=axis_contents)
        reduction = 0.1 if potential_type == "W" else None
        for rank, multipoles in (("charges", c), ("multipoles", m)):
            mapping = crys.map_multipoles(multipoles, foreshorten_hydrogens=reduction)
            for i, (j, rmsd, inv) in enumerate(mapping):
                if rmsd > 1e-4:
                    LOG.error(
                        "%s: %5s idx=%d%s (%.3g): %s",
                        xyz_files[i],
                        rank,
                        j,
                        "'" if inv else "",
                        rmsd, "failed",
                    )
                    success = False

        return success

    def create_axis_charges_mults_from_conformations(
        self, xyz_files: list[str], mult_files: list[str], cleanup_files: list[str]
    ) -> list[str]:
        """Function to create dmacrys axis, multipole and charge file for
        Z'>1 calculations which contain multiple conformations

        Args:
            xyz_files (list[str]): A list containing xyz file names of each molecular conformation
            mult_files (list[str]): A list containing multipole file names of each molecular conformation
            cleanup_files (list[str]): A list of the current files to be cleaned up

        Returns:
            list[str]: An updated list of the files to be cleaned up
        """
        import os
        import time
        LOG.error(mult_files)
        # with open(mult_files[0], 'r') as f:
        #     flines = f.readlines()
        #     LOG.error(flines)
        LOG.error(os.getcwd())
        for file in os.listdir(os.getcwd()):
            LOG.error(file)
        time.sleep(10000)
        title = generate_combined_name(xyz_files)
        combined_res_filename = "{title}.res".format(title=title)
        mf_format = title + "{}.dma"
        crystal = generate_combined_res(
            xyz_files, title, combined_res_filename, sort=True
        )
        potential = self.data["minimization"]["neighcrys"]["potential"]
        potential_type = available_potentials[potential][0]
        crystal.neighcrys_setup(potential_type=potential_type)
        # might be able to remove writing axis to file if not used by other functions
        # as we store this directly to self.data on line 191
        axis_file = Path("{}.mols".format(crystal.titl))
        axis_file.write_text(crystal.neighcrys_axis.to_string())
        mp_filename = mf_format.format("")
        oriented_mols = [mol.oriented() for mol in crystal.symmetry_unique_molecules()]
        equivalent_to = calculate_equivalent_molecules(oriented_mols)
        for i, _ in equivalent_to.items():
            mol = oriented_mols[i]
            mults = DistributedMultipoles.from_dma_file(mult_files[i])
            labels = mol.properties["neighcrys_label"]
            mults.molecules[0].labels = labels
            with open(mp_filename, "a") as f:
                f.write(mults.to_dma_string())
                f.write("\n#ENDMOL\n")
            fit_multipoles_with_mulfit(
                mult_files[i], [0], labels, mf_format, mulfit_method="cumulative"
            )  # This appends the rank_0 charge file to rank_0

        self.data["axis"] = crystal.neighcrys_axis.to_string()

        for key, value in {
            'multipoles':mp_filename,
            'charges':mf_format.format("_rank0")
        }:
            with open(value,"r") as f:
                self.data[key] = f.read()
        cleanup_files += [
            mp_filename,
            mf_format.format("_rank0"),
            "{}.mols".format(crystal.titl),
            combined_res_filename,
        ]
        return cleanup_files
    
    def merge_conformation_axis_charges_mults(self, xyz_files: list[str]) -> None:
        """Function to create dmacrys axis, multipole and charge file for
        Z'>1 calculations which contain multiple conformations

        Args:
            xyz_files (list[str]): A list containing xyz file names of each molecular conformation
            charges_files (list[str]): A list containing charges file names of each molecular conformation
            mult_files (list[str]): A list containing multipole file names of each molecular conformation
            cleanup_files (list[str]): A list of the current files to be cleaned up

        Returns:
            list[str]: An updated list of the files to be cleaned up
        """

        from cspy.apps.gplus import scrape_molecular_data, merge_neighcrys_labels, write_xyz_files, merge_charges_files, merge_mults_files, merge_axes_files

        combined_name = None
        file_extensions = ['.xyz', '_rank0.dma', '.dma', '.mols']

        molecule_seeds = []
        for xyz in xyz_files:
            molecule_seeds.append(xyz.split('.xyz')[0])

        molecules_data = dict()
        all_charges_content = []
        all_mults_content = []
        all_axes_content = []

        molecules_data = scrape_molecular_data(molecule_seeds, file_extensions)

        for seed in molecule_seeds:
            # xyz_coordinates, charges, mults, axes
            all_charges_content.append(molecules_data[seed][2])
            all_mults_content.append(molecules_data[seed][3])
            all_axes_content.append(molecules_data[seed][4])

        # get the new neighcrys labels
        relabel_dicts, element_ordering = merge_neighcrys_labels(all_charges_content)

        # finally, write input files for CSPy
        self.data["asymmetric_unit"] = write_xyz_files(molecule_seeds, molecules_data, element_ordering, combined_name, overwrite=True, skip_writing=True)
        self.data["charges"] = merge_charges_files(all_charges_content, relabel_dicts, element_ordering, combined_name, skip_writing=True)
        self.data["multipoles"] = merge_mults_files(all_mults_content, relabel_dicts, element_ordering, combined_name, skip_writing=True)
        self.data["axis"] = merge_axes_files(all_axes_content, relabel_dicts, combined_name, skip_writing=True)

    def setup(self, seed: int, spacegroup: int) -> Union[None, tuple]:
        """Runs the initial setup for the crystal generation and minimizaton steps.

        Args:
            seed (int): The current sobol seed used for setting the random number generator.
            spacegroup (int): The space group of which a crystal will be generated in.

        Returns:
            Union[None,tuple]: Normal termination results in returning None. If an error is raised
            will return a tuple so that the csp manager can process the error.
        """
        # initialised a try block so that we can make sure file cleanup is always run inside the
        # finally block, whether an error is raised or not.
        try:
            random.seed(seed)
            cleanup_files = []
            xyz_files = []
            charges_files = []
            mult_files = []
            axis_files = []
            # charges, axis and multipoles added to this loop as they can be occupied from a previous
            # job which can then lead to RMSD errors when mapping the multipoles.
            for property in [
                "energy",
                "molecule_id",
                "asymmetric_unit",
                "charges",
                "axis",
                "multipoles",
                "charges",
            ]:
                self.data[property] = []

            for id, item in enumerate(self.data["dbs"]):
                db = CspDataStoreFlex(item + ".db")
                elimit = self.data["min_energy"][id] + self.data["conf_energy_window"]
                # get the id of all conformations below elimit
                ids = [
                    row[0]
                    for row in db.select(
                        "distorted_molecules",
                        ["id", "energy"],
                        "where energy <= " + str(elimit),
                    )
                ]
                # select a random conformation from the ids list and extract
                # data such as coordinates, multipoles, energy, e.t.c from db
                choice = random.choice(ids)
                for row in db.select(
                    "distorted_molecules",
                    ["id", "axes", "xyz_coordinates", "charges", "mults", "energy"],
                    f'where id = "{choice}"',
                ):
                    file_name = str(spacegroup) + "_" + str(seed) + "_" + row[0]
                    xyz_files += [file_name + ".xyz"]
                    charges_files += [file_name + "_rank0.dma"]
                    mult_files += [file_name + ".dma"]

                    self.data["asymmetric_unit"] += [row[2]]
                    self.data["energy"] += [row[5]]
                    self.data["molecule_id"] += [row[0]]

                    # if we only have one db, we can extract the axis, charges and mults from the
                    # conformation db as these have already been calculated in the mol-dis step.
                    if len(self.data["dbs"]) == 1:
                        self.data["axis"] = row[1]
                        self.data["multipoles"] = row[4]
                        self.data["charges"] = row[3]
                        bondlength_cutoffs = {}
                        mol = Molecule.from_xyz_string(row[2])
                        for k, v in mol.neighcrys_bond_cutoffs().items():
                            if k not in bondlength_cutoffs or v > bondlength_cutoffs[k]:
                                bondlength_cutoffs[k] = v
                        self.data["bondlength_cutoffs"] = bondlength_cutoffs
                        with open(f"{file_name}.xyz",'w') as f:
                            f.write(row[2])
                        cleanup_files += [file_name + ".xyz"]

                    # write xyz, charges and mults to file so that we can create combined axis, charge
                    # and multipole files for crystals containig random conformations from each db.
                    else:
                        files_to_write = {
                            ".mols" : row[1],
                            ".xyz": row[2],
                            ".dma": row[4],
                            "_rank0.dma": row[3]
                        }
                        for ext, contents in files_to_write.items():
                            with open(file_name + ext, 'w') as f:
                                f.write(contents)

                        cleanup_files += [
                            file_name + ".mols",
                            file_name + ".xyz",
                            file_name + ".dma",
                            file_name + "_rank0.dma",
  
                        ]
                db.disconnect()

            # If any of the axis, charge or multipoles in self.data are not populated,
            # then we create these. Currently, this should trigger for G>1 calculations
            if not all(
                [self.data["axis"], self.data["multipoles"], self.data["charges"]]
            ):
                # cleanup_files = self.create_axis_charges_mults_from_conformations(
                #     xyz_files=xyz_files,
                #     mult_files=mult_files,
                #     cleanup_files=cleanup_files,
                # )
                self.merge_conformation_axis_charges_mults(xyz_files=xyz_files)
                if not self.data["bondlength_cutoffs"]:
                    bondlength_cutoffs = {}
                    for idx, xyz in enumerate(xyz_files):
                        mol = Molecule.load(xyz)
                        for k, v in mol.neighcrys_bond_cutoffs().items():
                            if k not in bondlength_cutoffs or v > bondlength_cutoffs[k]:
                                bondlength_cutoffs[k] = v

                    self.data["bondlength_cutoffs"] = bondlength_cutoffs

            if not self.check_multipoles_valid_flex(
                xyz_files=xyz_files,
                potential=self.data["minimization"]["neighcrys"]["potential"],
            ):
                LOG.error(
                    "Mapping the multipoles to their respective molecules gave a larger than expected RMSD."
                )

        except Exception:
            var = traceback.format_exc()
            LOG.error(f"The following error was raised during setup:\n{var}\n")
            return (
                "Error",
                None,
                None,
            )  # When processed by the csp manager, this will stop all workers and end the simulation.

        finally:
            # if error was raised during create_axis_charges_mults_from_conformations
            # then it is possible some files were not passed to cleanup.
            prefix = str(spacegroup) + "_" + str(seed)
            cleanup_files += [
                item
                for item in listdir()
                if prefix in item and item.endswith((".dma", ".res", ".mols", ".xyz"))
            ]
            for fname in cleanup_files:
                f = Path(fname)
                if f.exists():
                    f.unlink()

    def worker_data_aut(self, seed : int) -> None:
        """Randomly (not quasi) selects an asymmetric unit from a dictionary
        and store it in self.data.
        Random choice is weighted.

        Parameters
        ----------
        seed : int
            Random seed

        Returns
        -------
        None
        """
        aus = self.data["clusters"]
        properties = self.data["cluster_properties"]
        weights = self.data["cluster_weights"]

        random.seed(seed)

        au_id = random.choices(aus, weights = weights, k = 1)[0]
        xyz_string = properties[au_id]['xyz_coordinates']

        self.data["asymmetric_unit"] = [Molecule.from_xyz_string(xyz_string)]
        self.data["asymmetric_unit_id"] = au_id

    def structure_generation_aut(self, 
                                 args : dict = dict(), 
                                 overrides : dict = dict()
                                 ) -> tuple[CSPyTaskTag.QR, tuple[int, list[GeneratedStructure], list[int]], float]:
        """Iterate over a set of seeds and generate crystals
            for each seed using asymmetric unit transplant (SAUCE).
            For each seed, an asymmetric unit is pseudo-randomly selected
            and placed into a quasi-random cell lattice.

            Return the generated crystal structures.

        Parameters
        ----------
        args : dict
            Dictionary of arguments

        overrides : dict
            Dictionary that allows for overriding values set 
            in self.data

        Returns
        -------
        CSPyTaskTag.QR : CSPyTaskTag.QR
            Task tag used for mpi workers

        spacegroup : int
            Space group of generated crystals

        crystals : list[GeneratedStructure]
            list of crystals generated by the CLG

        au_ids : list[int]
            list of indices of asymmetric units used
            to generate crystals

        time : float
            Time spent on CLG
        """
        # only works for non-flex systems right now because the databases can't handle variable multipoles
        t1 = time.time()
        spacegroup, (min_seed, max_seed), name = args
        LOG.debug("min seed: %d, max seed: %d", min_seed, max_seed)

        self.data["clusters"] = overrides.get('aus', self.data["clusters"])
        self.data["cluster_properties"] = overrides.get('au_properties', self.data["cluster_properties"])
        self.data["cluster_weights"] = overrides.get('au_weights', self.data["cluster_weights"])

        crystals = []
        au_ids = []
        for seed in range(min_seed, max_seed):
            self.worker_data_aut(seed)
            molecules = copy.deepcopy(self.data["asymmetric_unit"])
            au_id = self.data["asymmetric_unit_id"]
            au_ids.append(au_id)
            clg = CrystalGenerator(molecules, spacegroup, nudge=self.data["nudge"],  adaptcell=self.data["adaptcell"], asi=self.data["asi"])
            generated = clg.generate(seed)
            if generated is None:
                LOG.debug("Failed to generate AUT crystal")
                crystals.append(None)
                continue
            res = generated.to_shelx_string(titl=str(
                CspDatabaseId.from_components(name, "QR", spacegroup, seed, 0)
            ))
            crystals.append(GeneratedStructure(
                    name=name, trial_number=seed, spacegroup=spacegroup,
                    file_content=res
            ))

        return CSPyTaskTag.QR, (spacegroup, crystals, au_ids), time.time() - t1

    def structure_generation_mps(self, args : dict
                                 ) -> tuple[CSPyTaskTag.QR, tuple[int, list[GeneratedStructure]], float]:
        """Iterate over a set of seeds and generate crystals
            for each seed using molecular pair seeding (SAUCE).
            For each seed, molecular pairs are pseudo-randomly 
            chained together into an asymmetric unit and 
            placed into a quasi-random cell lattice.

            Return the generated crystal structures.

        Parameters
        ----------
        args : dict
            Dictionary of arguments

        overrides : dict
            Dictionary that allows for overriding values set 
            in self.data

        Returns
        -------
        CSPyTaskTag.QR : CSPyTaskTag.QR
            Task tag used for mpi workers

        spacegroup : int
            Space group of generated crystals

        crystals : list[GeneratedStructure]
            list of crystals generated by the CLG

        time : float
            Time spent on CLG
        """
        # only works for non-flex systems right now because the databases can't handle variable multipoles

        from cspy.sauce.clg_mps import generate_multi_pair_crystal
        from cspy.formats.dma import parse_dma_string

        t1 = time.time()
        spacegroup, (min_seed, max_seed), name = args
        LOG.debug("min seed: %d, max seed: %d", min_seed, max_seed)
        reference_molecules = [
            Molecule.from_xyz_string(x) for x in self.data["asymmetric_unit"]
        ]

        unshuffled_xyz = [xyz.strip('.xyz') for xyz in self.data["xyz_names"]]

        dma_data = parse_dma_string(self.data["charges"])
        atom_types = dict()
        charges = dict()
        for ind, molecule_dma in enumerate(dma_data):
            atom_types[unshuffled_xyz[ind]] = molecule_dma['atom_types']
            charges[unshuffled_xyz[ind]] = molecule_dma['charges']

        crystals = []
        for seed in range(min_seed, max_seed):
            molecules = copy.deepcopy(reference_molecules)
            LOG.debug("Attempting to generate MPS crystal")
            clg = CrystalGenerator(molecules, spacegroup, nudge=self.data["nudge"],  adaptcell=self.data["adaptcell"], asi=self.data["asi"])
            clg, overrides = generate_multi_pair_crystal(pair_types=self.data["clusters"], properties=self.data["cluster_properties"], clg=clg, molecules=self.data["molecule_properties"], unshuffled_xyz=unshuffled_xyz, spg=spacegroup, weights=self.data["cluster_weights"], sample_size=100, seed=seed)

            if clg:
                generated = clg.generate(seed, overrides=overrides)
            else:
                LOG.error("Failed when calling MPS CLG")
                generated = None

            if generated is None:
                crystals.append(None)
                continue
            res = generated.to_shelx_string(titl=str(
                CspDatabaseId.from_components(name, "QR", spacegroup, seed, 0)
            ))
            LOG.info("Adding valid MPS crystal to list")
            crystals.append(GeneratedStructure(
                    name=name, trial_number=seed, spacegroup=spacegroup,
                    file_content=res
            ))

        return CSPyTaskTag.QR, (spacegroup, crystals), time.time() - t1

    def structure_generation_flex(self, args: tuple) -> tuple:
        """Crystal structure generation function adapted for flexible-molecule CSP
        using CLG 2.0

        Args:
            args (tuple): A tuple containing space group, seed and name information

        Returns:
            tuple: A tuple containing the job tag, spacegroup, generated crystals
            and time taken
        """

        t1 = time.time()
        spacegroup, (min_seed, max_seed), name = args
        LOG.debug("min seed: %d, max seed: %d", min_seed, max_seed)
        crystals = []
        for seed in range(min_seed, max_seed):
            error_code = self.setup(seed, spacegroup)
            if error_code:
                return error_code
            molecules = [
                Molecule.from_xyz_string(x) for x in self.data["asymmetric_unit"]
            ]
            clg = CrystalGenerator(
                molecules,
                spacegroup,
                nudge=self.data["nudge"],
                adaptcell=self.data["adaptcell"],
                asi=self.data["asi"],
            )
            generated = clg.generate(seed)
            if generated is None:
                crystals.append(None)
                continue
            res = generated.to_shelx_string(
                titl=str(CspDatabaseId.from_components(name, "QR", spacegroup, seed, 0))
            )
            crystals.append(
                GeneratedStructure_flex(
                    name=name,
                    trial_number=seed,
                    spacegroup=spacegroup,
                    file_content=res,
                    file_mults=self.data["multipoles"],
                    file_axis=self.data["axis"],
                    file_charge=self.data["charges"],
                    file_bonds=self.data["bondlength_cutoffs"],
                    conf_energy=self.data["energy"],
                    molecule_id=self.data["molecule_id"],
                )
            )
        return CSPyTaskTag.QR, (spacegroup, crystals), time.time() - t1

    def structure_generation(self, 
                            args : dict = dict(), 
                            overrides : dict = dict()
                            ) -> tuple[CSPyTaskTag.QR, tuple[int, list[GeneratedStructure]], float]:
        """Iterate over a set of seeds and generate crystals
            For each seed, quasi-randomly select molecule 
            rotations and translations, and unit cell parameters.

            Return the generated crystal structures.

        Parameters
        ----------
        args : dict
            Dictionary of arguments

        overrides : dict
            Dictionary that allows for overriding values set 
            in self.data

        Returns
        -------
        CSPyTaskTag.QR : CSPyTaskTag.QR
            Task tag used for mpi workers

        spacegroup : int
            Space group of generated crystals

        crystals : list[GeneratedStructure]
            list of crystals generated by the CLG

        time : float
            Time spent on CLG
        """
        
        t1 = time.time()
        spacegroup, (min_seed, max_seed), name = args
        LOG.debug("min seed: %d, max seed: %d", min_seed, max_seed)
        if "asymmetric_unit" in overrides.keys():
            self.data["asymmetric_unit"] = overrides["asymmetric_unit"]
        molecules = [
            Molecule.from_xyz_string(x) for x in self.data["asymmetric_unit"]
        ]
        clg = CrystalGenerator(molecules, spacegroup, nudge=self.data["nudge"],  adaptcell=self.data["adaptcell"], asi=self.data["asi"])
        crystals = []
        for seed in range(min_seed, max_seed):
            generated = clg.generate(seed)
            if generated is None:
                crystals.append(None)
                continue
            res = generated.to_shelx_string(
                titl=str(CspDatabaseId.from_components(name, "QR", spacegroup, seed, 0))
            )
            crystals.append(
                GeneratedStructure(
                    name=name,
                    trial_number=seed,
                    spacegroup=spacegroup,
                    file_content=res,
                )
            )

        return CSPyTaskTag.QR, (spacegroup, crystals), time.time() - t1

    def minimize_structure(
        self, structure: Union[list, NamedTuple], overrides: dict
    ) -> tuple:
        """Crystal structure minimization function

        Args:
            structure (Union[list, NamedTuple]): A list of NamedTuples or single NamedTuple object
            containing the necessary info to minimize the crystal(s).
            overrides (Dict): A dictionary containing overrides for the minimization settings typically
            determined when calculating automatic timeouts.

        Returns:
            tuple: A tuple containing the job tag, whether the minimization was successful, minimized
            crystals and time taken.
        """
        from cspy.configuration import configure, CONFIG

        t1 = time.time()

        if isinstance(structure, list):
            structure = structure[0]

        name = structure.name
        sg = structure.spacegroup
        Zp = self.data["Zp"]
        crystal_info = {"spacegroup" : sg, "Zp" : Zp}
        trial_number = structure.trial_number
        if self.data["flex"]:
            charges = DistributedMultipoles.from_dma_string(structure.file_charge)
            multipoles = DistributedMultipoles.from_dma_string(structure.file_mults)
            axis = structure.file_axis
            bondlength_cutoffs = structure.file_bonds
            ei = structure.conf_energy
            molecule_id = structure.molecule_id
        else:
            if self.data["charges"]:
                charges = DistributedMultipoles.from_dma_string(self.data["charges"])
            else:
                charges = None
            if self.data["multipoles"]:
                multipoles = DistributedMultipoles.from_dma_string(self.data["multipoles"])
            else:
                multipoles = None
            if self.data["charges"]:
                axis = self.data["axis"]
            else:
                axis = None
            bondlength_cutoffs = self.data["bondlength_cutoffs"]
        minimization_settings = self.data["minimization"]
        for key in overrides.keys():
            minimization_settings[key] = overrides[key]
        check_spe = self.data["check_single_point_energy"]
        configure(minimization_settings)

        crystal_info = {"spacegroup": sg}
        minimizer = CompositeMinimizer.from_defaults(
            charges=charges,
            axis=axis,
            multipoles=multipoles,
            bondlength_cutoffs=bondlength_cutoffs,
            keep_files=self.data["keep_files"],
            crystal_info=crystal_info,
        )

        structure_id = CspDatabaseId.from_components(name, self.task_name, sg, trial_number, 0)
        try:
            structure = Crystal.from_shelx_string(structure.file_content)
            res = structure.to_shelx_string(titl=structure_id)
            input_valid = True
        except:
            LOG.error("Input crystal structure %s cannot be interpreted from shelx string: %s", structure_id, structure.file_content)
            res = None
            input_valid = False

        

        if self.data["flex"]:
            molecule_id = "-".join(molecule_id)
            crystals = [
                MinimizedStructure(
                    id=structure_id,
                    energy=float("nan"),
                    density=float("nan"),
                    spacegroup=sg,
                    trial_number=trial_number,
                    time=float("nan"),
                    xrd=None,
                    file_content=res,
                    minimization_step=0,
                    molecule_id=molecule_id,
                )
            ]
        else:
            molecule_id = name
            crystals = [
                MinimizedStructure(
                    id=structure_id,
                    energy=float("nan"),
                    density=float("nan"),
                    spacegroup=sg,
                    trial_number=trial_number,
                    time=float("nan"),
                    xrd=None,
                    file_content=res,
                    minimization_step=0,
                    molecule_id=molecule_id,
                )
            ]
        if input_valid:
            # try:
            #     minimized_crystals = minimizer(structure)
            # except Exception:
            #     print(traceback.format_exc())
            #     LOG.error("Captured unknown error in minimiser in process %s", str(self.rank))
            #     minimized_crystals = ["unknown"]
            minimized_crystals = minimizer(structure)
        else:
            minimized_crystals = ["unknown"]

        valid = False
        for minimization_step, crystal in enumerate(minimized_crystals, start=1):
            if isinstance(crystal, str):
                valid = crystal
            else:
                structure_id = CspDatabaseId.from_components(
                    name, self.task_name, sg, trial_number, minimization_step
                )
                try:
                    energy = crystal.properties["lattice_energy"]
                except KeyError:
                    # Labelled as final_energy for dftb calculations - consider refactor
                    energy = crystal.properties["final_energy"]
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
                            potential=self.data["minimization"]["neighcrys"][
                                "potential"
                            ],
                            name=name,
                            **self.data["minimization"]["dmacrys"],
                        )
                        if abs(energy - energy_check) > 10.0:
                            check_through = False
                            LOG.info(
                                "Different energy from single point evaluation id %s, "
                                "energy %s, energy_check %s",
                                structure_id,
                                energy,
                                energy_check,
                            )
                    if check_through:
                        valid = True
                        # pxrd is only supported descriptor atm but this can be expanded on
                        # Platon is only supported method for pxrd, but we can add others (e.g. pymatgen)
                        if self.data["descriptors"].get("pxrd", None) in ['Platon']:
                            pp = PowderPattern.from_cif_string(crystal.to_cif_string())
                            if pp is not None:
                                xrd = pp.pattern
                        else:
                            xrd = None

                if self.data["flex"]:
                    energy_correction = 0.0
                    for idx, ener in enumerate(ei):
                        energy_correction += energy_correction + (
                            ener - self.data["min_energy"][idx]
                        )
                    energy = energy + (energy_correction / len(ei))
                    res = crystal.to_shelx_string(
                        titl=f"{structure_id} {energy} {density}"
                    )

                crystals.append(
                    MinimizedStructure(
                        id=structure_id,
                        energy=energy,
                        density=density,
                        spacegroup=sg,
                        trial_number=trial_number,
                        time=mtime,
                        xrd=xrd,
                        file_content=res,
                        minimization_step=minimization_step,
                        molecule_id=molecule_id,
                    )
                )

        return CSPyTaskTag.OPT, (valid, crystals), time.time() - t1


    def extract_asymmetric_unit(self, 
                                args : dict = dict(), 
                                overrides : dict = dict()
                                ) -> tuple[CSPyTaskTag.EAU, tuple[bool, dict, int], None]:
        """Take a crystal structure and extract the lowest
        energy representation of the asymmetric unit.

        Parameters
        ----------
        args : dict
            Dictionary of arguments

        overrides : dict
            Dictionary that allows for overriding values set 
            in self.data

        Returns
        -------
        CSPyTaskTag.EAU : CSPyTaskTag.EAU
            Task tag used for mpi workers

        valid : bool
            Whether or not the extraction worked

        aus_data : dict
            Dictionary of data describing the asymmetric unit.
            See extract_min_energy_au for more details.

        spacegroup : int
            Space group of crystal structure that the asymmetric
            unit was sourced from

        None : None
            When the master worker receives this data, it expects 3
            elements. The last element is use dby other functions but
            not this one.
        """
        from cspy.sauce.extract_aut import extract_min_energy_au

        structure = args["structure"]
        spacegroup = args["spacegroup"]
        G = len(self.data["xyz_names"])
        sorted_molecules = self.data["xyz_names"]
        molecules_elements = dict()
        for molecule in sorted_molecules:
            molecules_elements[molecule] = self.data["molecule_properties"][molecule]["atom_elements"]
        pot = self.data["minimization"]["neighcrys"]["potential"]
        atom_types = self.data["pot_atom_types"]
        charges = self.data["pot_charges"]
        Zp = self.data["Zp"]
        atom_ids = self.data["atom_ids"]

        aus_data = extract_min_energy_au(structure, G, spacegroup, sorted_molecules, molecules_elements, Zp, pot, atom_types, charges, atom_ids)
        if aus_data:
            valid = True
        else:
            valid = False
        
        return CSPyTaskTag.EAU, (valid, aus_data, spacegroup), None
