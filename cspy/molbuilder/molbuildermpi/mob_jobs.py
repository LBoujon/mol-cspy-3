import logging
import os
import re
import time
import shutil
import ast
import glob
import numpy as np
from collections import namedtuple
from scipy.spatial.distance import pdist
from enum import IntEnum
from pathlib import Path
from rdkit import Chem
from cspy.chem import Molecule
from cspy.crystal import Crystal
from cspy.apps.dma import generate_combined_name
from cspy.apps.dma import generate_combined_res
from cspy.apps.dma import generate_multipoles
from cspy.crystal.generate_crystal import CrystalGenerator
from cspy.cspympi.cspy_tasks import GeneratedStructure
from cspy.cspympi.cspy_tasks import MinimizedStructure
from cspy.db.key import CspDatabaseId
from cspy.db import CspDataStore
from cspy.db.clustering import find_equivalent_structures
from cspy.configuration import configure
from cspy.chem.multipole import DistributedMultipoles
from cspy.configuration import CspyConfiguration
from cspy.minimize import CompositeMinimizer
from cspy.minimize import single_point_evaluation
from cspy.ml.descriptors import PowderPattern
from cspy.molbuilder.exceptions import PropertyCalculationFailure
from cspy.molbuilder.subprocess import coordinates_from_uff
from cspy.molbuilder.subprocess import CP2KSub
from cspy.molbuilder.mobility import AOM
from cspy.molbuilder.mobility import Marcus
from cspy.pyAOMlite.pyAOM_utils import single_molecule
from .gaussian_jobs import GaussianWorker


LOG = logging.getLogger(__name__)


k_b_ev = 8.617333262145e-5
MinimizationStructure = namedtuple(
    'MinimizationStructure', 'id energy file_content'
)
ClusterArgs = namedtuple(
    'ClusterArgs',
    'output method cluster_energy_threshold '
    'cluster_density_threshold cluster_xrdcdtw_threshold '
    'cluster_xrdcos_threshold cluster_from_equivalent'
)


class MobTaskTag(IntEnum):
    SLEEP = 0
    REORG = 1
    MULT = 2
    QR = 3
    OPT = 4
    CP2K = 5
    MOB = 6
    FAILED = 7


class MobWorker(GaussianWorker):

    def __init__(self, config, sub_node_comm):
        super().__init__(config)
        self.config = config
        self.sub_node_comm = sub_node_comm
        self.len_target = float(config['MOBBUILDER']['len_target'])
        self.vdw_buffer = float(config['MOBBUILDER']['vdw_buffer'])
        self.energy_range = float(config['MOBBUILDER']['energy_range'])
        self.temp = float(config['MOBBUILDER']['temperature'])
        self.g09_method = config['GAUSSIAN']['method']
        self.g09_basis_set = config['GAUSSIAN']['basis_set']
        self.cp2k_basis_file = config['AOM']['cp2k_basis_file']
        self.cp2k_basis = config['AOM']['basis']
        self.sto_proj_dict = ast.literal_eval(config['AOM']['STO_proj_dict'])
        self.aom_dict = ast.literal_eval(config['AOM']['AOM_dict'])
        self.c = float(config['AOM']['C'])
        self.space_groups = ast.literal_eval(config['CSP']['space_groups'])
        self.sampling = ast.literal_eval(config['CSP']['sampling'])
        self.cspy_config = CspyConfiguration()
        self.cp2k_sub = CP2KSub(config)
        self.marcus = Marcus(self.config)
        self.parent_dir = os.getcwd() + '/molbuilderlogs'

    def calculate(self, job):
        os.chdir(self.parent_dir)
        task, (inchi, data) = job

        if task == MobTaskTag.SLEEP:
            self.barrier(self.sub_node_comm, tag=0)
            self.barrier(self.sub_node_comm, tag=1)
            return MobTaskTag.SLEEP, (None, None)

        inchi_key = Chem.InchiToInchiKey(inchi)
        if task == MobTaskTag.REORG:
            self.barrier(self.sub_node_comm, tag=0)
            try:
                reorg_dir = Path(inchi_key + '/reorg')
                reorg_dir.mkdir(exist_ok=True, parents=True)
                os.chdir(reorg_dir)

                molecule = Chem.AddHs(Chem.MolFromInchi(inchi))
                coordinates = coordinates_from_uff(molecule)
                results_1 = self.get_gaussian_results(
                    coordinates, 0, 1, 'opt',
                    inchi_key + '_STAGE_1.com', inchi_key + '_STAGE_1.log'
                )
                results_2 = self.get_gaussian_results(
                    coordinates, self.charge, 2, 'opt',
                    inchi_key + '_STAGE_2.com', inchi_key + '_STAGE_2.log'
                )
                results_3 = self.get_gaussian_results(
                    results_1.coordinates, self.charge, 2, 'sp',
                    inchi_key + '_STAGE_3.com', inchi_key + '_STAGE_3.log'
                )
                results_4 = self.get_gaussian_results(
                    results_2.coordinates, 0, 1, 'sp',
                    inchi_key + '_STAGE_4.com', inchi_key + '_STAGE_4.log'
                )
                e_0_r_0 = results_1.energy
                e_n_r_n = results_2.energy
                e_n_r_0 = results_3.energy
                e_0_r_n = results_4.energy
                reorg = (e_n_r_0 - e_0_r_0 + e_0_r_n - e_n_r_n) * 27.2113961
                affinity = (e_0_r_0 - e_n_r_n) * 27.2113961
                results = (reorg, affinity)
                return MobTaskTag.REORG, (inchi, results)

            except PropertyCalculationFailure as e:
                LOG.warning('REORG calculation failed with '
                            'exception (%s) for molecule %s', e, inchi)
                return MobTaskTag.FAILED, (inchi, None)

            finally:
                self.barrier(self.sub_node_comm, tag=1)

        elif task == MobTaskTag.MULT:
            mult_dir = Path(inchi_key + '/mult')
            mult_dir.mkdir(exist_ok=True, parents=True)
            os.chdir(mult_dir)

            # Z'=1 only
            mol = Molecule.load('../reorg/' + inchi_key + '_STAGE_1.log')
            mol.save(inchi_key + '.xyz')
            xyz_files = [inchi_key + '.xyz']

            title = generate_combined_name(xyz_files)
            res_filename = title + '.res'
            mf_format = title + '{}.dma'
            crystal = generate_combined_res(
                xyz_files, title, res_filename, True
            )
            try:
                cleanup_files = generate_multipoles(
                    crystal,
                    [0],
                    mf_format,
                    basis_set=self.g09_basis_set,
                    method=self.g09_method,
                    xyz_filenames=xyz_files,
                    nprocshared=1
                )
                cleanup_files.append(res_filename)
                for fname in cleanup_files:
                    f = Path(fname)
                    if f.exists():
                        f.unlink()
            except RuntimeError as e:
                LOG.warning('MULT calculation failed with '
                            'exception (%s) for molecule %s', e, inchi)
                return MobTaskTag.FAILED, (inchi, None)

            # setup the csp and cp2k directories
            os.chdir(self.parent_dir)
            csp_dir = Path(inchi_key + '/csp')
            csp_dir.mkdir(exist_ok=True, parents=True)
            cp2k_dir = Path(inchi_key + '/cp2k')
            cp2k_dir.mkdir(exist_ok=True, parents=True)

            shutil.copyfile(
                inchi_key + '/mult/' + inchi_key + '.xyz',
                inchi_key + '/csp/' + inchi_key + '.xyz'
            )
            shutil.copyfile(
                inchi_key + '/mult/' + inchi_key + '.dma',
                inchi_key + '/csp/' + inchi_key + '.dma'
            )
            shutil.copyfile(
                inchi_key + '/mult/' + inchi_key + '_rank0.dma',
                inchi_key + '/csp/' + inchi_key + '_rank0.dma'
            )
            shutil.copyfile(
                inchi_key + '/mult/' + inchi_key + '.mols',
                inchi_key + '/csp/' + inchi_key + '.mols'
            )
            shutil.copyfile(
                inchi_key + '/mult/' + inchi_key + '.xyz',
                inchi_key + '/cp2k/' + inchi_key + '.xyz'
            )

            os.chdir(csp_dir)
            for sg in self.space_groups:
                db_filename = '{}_{}.db'.format(inchi_key, sg)
                ds = CspDataStore.create_and_connect(db_filename)
                ds.close()

            return MobTaskTag.MULT, (inchi, None)

        elif task == MobTaskTag.QR:
            logs_dir = Path(inchi_key + '/csp')
            os.chdir(logs_dir)

            sg, (min_seed, max_seed) = data
            mols = [Molecule.load(inchi_key + '.xyz')]
            clg = CrystalGenerator(mols, sg)

            crystals = []
            for seed in range(min_seed, max_seed + 1):
                generated = clg.generate(seed)
                if generated is None:
                    crystals.append(None)
                    continue
                res = generated.to_shelx_string(
                    titl=str(CspDatabaseId.from_components(
                        inchi_key, 'QR', sg, seed, 0
                    ))
                )
                crystals.append(GeneratedStructure(
                    name=inchi_key, trial_number=seed, spacegroup=sg,
                    file_content=res
                ))

            return MobTaskTag.QR, (inchi, (sg, crystals))

        elif task == MobTaskTag.OPT:
            logs_dir = Path(inchi_key + '/csp')
            os.chdir(logs_dir)

            structure = data
            name = structure.name
            sg = structure.spacegroup
            trial_number = structure.trial_number

            charges = DistributedMultipoles.from_dma_file(
                inchi_key + '_rank0.dma')
            multipoles = DistributedMultipoles.from_dma_file(
                inchi_key + '.dma')
            axis = Path(inchi_key + '.mols').read_text()

            bondlength_cutoffs = {}
            mol = Molecule.load(inchi_key + '.xyz')
            for k, v in mol.neighcrys_bond_cutoffs().items():
                if k not in bondlength_cutoffs or v > bondlength_cutoffs[k]:
                    bondlength_cutoffs[k] = v

            vdw_cutoff = 15.0
            vdw_cutoff = max(vdw_cutoff, 1.5 * np.max(pdist(mol.positions)))

            minimization_settings = {
                'neighcrys': {
                    'vdw_cutoff': vdw_cutoff,
                    'potential': 'fit',
                },
                'pmin': {'timeout': self.cspy_config.get('pmin.timeout')},
                'dmacrys': {'timeout': self.cspy_config.get('dmacrys.timeout')},
                'minimization_steps': self.cspy_config.get('csp_minimization_step'),
            }
            check_spe = self.cspy_config.get("csp.check_single_point_energy")
            configure(minimization_settings)

            minimizer = CompositeMinimizer.from_defaults(
                charges=charges, axis=axis, multipoles=multipoles,
                bondlength_cutoffs=bondlength_cutoffs
            )

            structure = Crystal.from_shelx_string(structure.file_content)
            structure_id = CspDatabaseId.from_components(
                name, 'QR', sg, trial_number, 0
            )
            res = structure.to_shelx_string(titl=structure_id)
            crystals = [
                MinimizedStructure(
                    id=structure_id,
                    energy=float('nan'),
                    density=float('nan'),
                    spacegroup=sg,
                    trial_number=trial_number,
                    time=float('nan'),
                    xrd=None,
                    file_content=res,
                    minimization_step=0
                )
            ]
            minimized_crystals = minimizer(structure)

            valid = False
            for minimization_step, crystal in enumerate(
                    minimized_crystals, start=1):

                structure_id = CspDatabaseId.from_components(
                    name, 'QR', sg, trial_number, minimization_step
                )
                energy = crystal.properties['lattice_energy']
                density = crystal.properties['density']
                mtime = crystal.properties['minimization_time']
                res = crystal.to_shelx_string(titl=structure_id)
                xrd = None

                if minimization_step == minimizer.step_count:
                    check_through = True
                    if check_spe:
                        energy_check, _ = single_point_evaluation(
                            crystal,
                            multipoles,
                            axis,
                            bondlength_cutoffs=bondlength_cutoffs,
                            potential=minimization_settings['neighcrys']['potential'],
                            name=name,
                            **minimization_settings['dmacrys'],
                        )
                        if abs(energy - energy_check) > 10.0:
                            check_through = False
                            LOG.info(
                                'Different energy from single point '
                                'evaluation id %s, energy %s, '
                                'energy_check %s',
                                structure_id, energy, energy_check
                            )
                    if check_through:
                        valid = True
                        pp = PowderPattern.from_cif_string(
                            crystal.to_cif_string())
                        if pp is not None:
                            xrd = pp.pattern

                crystals.append(MinimizedStructure(
                    id=structure_id,
                    energy=energy,
                    density=density,
                    spacegroup=sg,
                    trial_number=trial_number,
                    time=mtime,
                    xrd=xrd,
                    file_content=res,
                    minimization_step=minimization_step
                ))

            if valid:
                db_id = os.getcwd() + '/{}_{}.db'.format(inchi_key, sg)
                return MobTaskTag.OPT, (inchi, (sg, db_id, crystals, valid))
            else:
                return MobTaskTag.OPT, (inchi, (sg, None, None, valid))

        elif task == MobTaskTag.CP2K:
            cp2k_dir = Path(inchi_key + '/cp2k')
            os.chdir(cp2k_dir)
            self.cp2k_sub.run_cp2k(inchi_key + '.xyz')
            for cube in glob.glob('*.cube'):
                os.remove(cube)

            # cluster dbs for mobility evaluation
            os.chdir(self.parent_dir)
            csp_dir = Path(inchi_key + '/csp')
            os.chdir(csp_dir)

            output_db = CspDataStore('{}.db'.format(inchi_key))
            output_db.close()

            args = ClusterArgs(
                output='{}.db'.format(inchi_key),
                method='cdtw',
                cluster_energy_threshold=1.0,
                cluster_density_threshold=0.05,
                cluster_xrdcdtw_threshold=10.00,
                cluster_xrdcos_threshold=0.80,
                cluster_from_equivalent=False
            )

            for i, sg in enumerate(self.space_groups):
                db_filename = '{}_{}.db'.format(inchi_key, sg)
                db = CspDataStore(db_filename)
                # wait for the db thread to commit all crystal
                # structures before clustering
                while db.number_of_final_minimizations() != self.sampling[i]:
                    time.sleep(1)
                db.close()
                find_equivalent_structures(
                    db_filename, args, calculate_missing=False
                )
            find_equivalent_structures(
                '{}.db'.format(inchi_key), args, calculate_missing=False
            )

            os.chdir(self.parent_dir)
            mob_dir = Path(inchi_key + '/mob')
            mob_dir.mkdir(exist_ok=True, parents=True)
            shutil.move(
                inchi_key + '/csp/' + inchi_key + '.db',
                inchi_key + '/mob/' + inchi_key + '.db'
            )
            shutil.copyfile(
                inchi_key + '/mult/' + inchi_key + '.xyz',
                inchi_key + '/mob/' + inchi_key + '.xyz'
            )
            shutil.copyfile(
                inchi_key + '/cp2k/' + inchi_key + '_PBE_DZVP-GTH.out',
                inchi_key + '/mob/' + inchi_key + '_PBE_DZVP-GTH.out',
            )
            os.chdir(mob_dir)

            with open(inchi_key + '_PBE_DZVP-GTH.out', 'r') as cp2k:
                cp2k_out = cp2k.read()
            mo_number = int(
                re.findall(' *Number of electrons: +([0-9]+) *', cp2k_out)[0]
            )
            if self.carrier == 'electron':
                mo_number += 1

            mol = single_molecule(inchi_key + '.xyz')
            mol.get_cp2k_info(
                mo_number, inchi_key + '_PBE_DZVP-GTH.out',
                self.cp2k_basis_file, self.cp2k_basis
            )
            mol.initialize_STOs(self.sto_proj_dict)
            mol.project()
            os.makedirs(os.getcwd() + '/output/{}'.format(inchi_key))
            mol.save_AOM(inchi_key)
            LOG.info('STO projection completeness: %s', mol.orb_compl_dict)

            db = CspDataStore(inchi_key + '.db')
            contents = db.unique_structures().fetchall()
            db.close()
            structures = []
            min_energy = min([row[3] for row in contents])
            max_energy = min_energy + self.energy_range
            for row in contents:
                cid, _, _, energy, file_content = row
                if energy > max_energy:
                    continue
                structures.append(
                    MinimizationStructure(
                        id=cid,
                        energy=energy,
                        file_content=file_content
                    )
                )

            return MobTaskTag.CP2K, (inchi, (mo_number, structures))

        elif task == MobTaskTag.MOB:
            reorg, mo_number, structure = data
            mob_dir = Path(inchi_key + '/mob')
            os.chdir(mob_dir)

            crystal = Crystal.from_shelx_string(structure.file_content)
            aom_coeff_path \
                = os.getcwd() + '/output/{}/AOM_COEFF.dat'.format(inchi_key)
            aom = AOM(self.config, mo_number, aom_coeff_path)
            mob_eig = self.marcus.mobility(crystal, aom, reorg, self.temp)[0]
            mobility = np.sum(mob_eig) / 3
            return MobTaskTag.MOB, (inchi, (structure.id, mobility))

    def barrier(self, comm, tag=0, sleep=1):
        """Fix for issue where the default mpi barrier method causes
        near 100% cpu usage. Code was taken from,

        https://groups.google.com/forum/#!topic/mpi4py/nArVuMXyyZI

        "Below, an scalable point-to-point based implementation of
        barrier() with the sleep() trick you need. A standard
        implementation would just merge send() and recv() on a single
        sendrecv() call. Just may need to tweak the sleep interval, and
        perhaps use a different tag value to avoid previous on-going
        communication."

        :param comm: mpi comm object
        :param tag: the tag for the communication
        :param sleep: the sleep time to check for whether all workers
            have reached the barrier
        """
        size = comm.Get_size()
        if size == 1:
            return
        rank = comm.Get_rank()
        mask = 1
        while mask < size:
            dst = (rank + mask) % size
            src = (rank - mask + size) % size
            req = comm.isend(None, dst, tag)
            while not comm.Iprobe(src, tag):
                time.sleep(sleep)
            comm.recv(None, src, tag)
            req.Wait()
            mask <<= 1
