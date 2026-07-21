import hashlib
import os
import toml
import logging
from cspy.potentials import available_potentials
from cspy.util import recursive_dict_update, nested_dict_delete

try:
    from collections import ChainMap
except ImportError:
    from chainmap import ChainMap

MOST_COMMON_SPACEGROUPS = {
    "single": [
        61,
        14,
        19,
        2,
        4,
        15,
        33,
        9,
        29,
        5,
        1,
        60,
        7,
        18,
        96,
        76,
        145,
        43,
        56,
        13,
        169,
        88,
        20,
        86,
        148,
    ],
    "co-crystal": [2, 14, 15, 4, 19, 61, 1, 33, 5, 9],
}

CONFIG = None
LOG = logging.getLogger(__name__)

KNOWN_HOSTS = {
    "iridis4": {
        "hostname": "iridis4_a.soton.ac.uk",
        "queue_system": "pbs",
        "backup_hosts": ["iridis4_b.soton.ac.uk", "iridis4_c.soton.ac.uk"],
        "remote_dir": "/scratch/{user}/{procid}",
        "concurrency": 16,
    },
    "iridis5": {
        "hostname": "iridis5_a.soton.ac.uk",
        "queue_system": "slurm",
        "backup_hosts": ["iridis5_b.soton.ac.uk", "iridis5_c.soton.ac.uk"],
        "remote_dir": "/scratch/{user}/{procid}",
        "concurrency": 40,
    },
}


DEFAULTS = {
    "TIMEOUT": "1500.0",
    "NUMBER_STRUCTURES": 10000,
    "VDW_CUTOFF": "15.0",
    "BACKUPDIR": "/research/GMDayGroup/daygroup/${USER}",
    "PROCID": os.getpid(),
    "REDISPASS": "",
    # New style dynamic defaults
    "celery": {"worker": {"workdir": "/dev/shm/{user}/{pid}"}},
    "gaussian": {
        "nprocs": 1,
        "memory": "2GB",
        "method": "B3LYP",
        "basis": "6-311G**",
        "basis_file": None,
        "chelpg": None,
        "cleanup": None,
        "calculate_molecular_volume": None,
        "polarizable_continuum_model": None,
        "external_iteration_pcm": None,
        "dielectric_constant": None,
        "optimisation_options": None,
        "iso_value": None,
        "frequency_options": None,
        "all_molecules": None,
        "extra_arguments": None,
        "molecular_states": None,
        "force_rerun": None,
    },
    "neighcrys": {
        "extended_mode": True,
        "bondlength_filename": "bondlengths",
        "max_intermolecular_distance": 4.0,
        "insert_bond_centre_sites": False,
        "standardise_bonds": False,
        "wcal_ngcv": True,
        "max_iterations": 1000,
        "step_size": 0.5,
        "connectivity_search_max_cells": 3,
        "potential": "fit",
        "potential_type": available_potentials["fit"][0],
        "labels_filename": "",
        "foreshorten_hydrogens": False,
        "symmetry_subgroup": [0],
        "vdw_cutoff": 15.0,
        "multipole_filename": "",
        "polarizabilities_filename": "",
        "axis_filename": "",
        "paste_molecular_structure_filename": "",
        "potential_filename": available_potentials["fit"][1],
        "hessian_filename": ""
    },
    "pmin": {
        "iswitch": 2,
        "irosen": 2,
        "iuser_e": 0,
        "irot": 0,
        "isystem": 0,
        "nmol": 1,
        "sthl": 0.5,
        "ck": 0.25,
        "dmax": 10.0,
        "ddmax": 50,
        "icross": 0,
        "allow_change": 0,
        "coefftable": 0,
        "ls_cycles": 50,
        "rss_cycles": 50,
        "steps": [1e-5, 1e-5, 1e-5, 1e-5],
        "frac_changes": [1.0, 1.0, 1.0, 1.0],
        "timeout": 1200.0,
    },
    "dmacrys": {
        "timeout": 1200.0,
        "timeout_auto": False,
        "timeout_auto_cap": 1800.0,
        "timeout_auto_sample": 1000.0,
        "force_field_potentials" : "fit",
        "force_field_type" : "F",
        "foreshorten_hydrogens" :False
    },
    "csp_minimization_step": [
        {"kind": "pmin", "electrostatics": "charges"},
        {
            "kind": "dmacrys",
            "CONP": True,
            "PRES": "0.1 GPa",
            "electrostatics": "charges",
        },
        {"kind": "dmacrys", "electrostatics": "multipoles"},
    ],
    "csp": {
        "spacegroups": "fine10",
        "number_structures": None,
        "conf_energy_window": 22.0,
        "potential": "fit",
        "cutoff": "calculate",
        "status_file": "status.txt",
        "charges_file": None,
        "multipoles_file": None,
        "log_level": "INFO",
        "check_single_point_energy": False,
        "backup_interval": 10800
    },
    "fmcsp_wf": {
        "spacegroups": "fine5-wf",
        "number_of_crystals": None,
        "conf_energy_window": 30.0,
        "partition": "batch",
        "basis_set": "6-311G**",
        "method": "PBE1PBE",
    },
    "dftb": {
        "skf_set":"3ob-3-1",
        "skf_path":"/research/GMDayGroup/daygroup/dftbplus/",
        "disp_coeff":"3ob_brandenburg",
        "groups":1,
        "max_steps":2500,
        "max_force":0.00058,
        "alg":"LBFGS",
        "output_prefix":"geom.out",
        "kpoint_spacing":0.05,
        "scc_tol":1e-5,
        "timeout":1800,
        "single_point": False,
        "fixed_lattice_opt": False,
        "lattice_opt": True,
    },
    "vasp": {
        "potcar_path":"/home/mmm0823/pot/PBE",
        "kspacing":0.05,
        "timeout":3600,
        "incar_settings":{
            "ENCUT":"500",
            "PREC":"Normal",
            "LREAL":"Auto",
            "ISIF":"2",
            "IVDW":"12",
            "NELMIN":"4",
            "EDIFF":"1E-07",
            "EDIFFG":"-0.03" ,
            "NSW":"300",
            "ISYM":"2",
            "IBRION":"1",
            "POTIM":"0.3",
            "ISMEAR":"0",
            "SIGMA":"0.05",
            "KPAR":"4",
            "NCORE":"20",
            "LWAVE":".FALSE",
            "LCHARGE":".FALSE",
        },
        "mpi_settings":{
            "cores_per_structure":80,
        },
    },
    "ase": {
        "optimizer": "LBFGS",
        "fmax": 0.05,
        "steps" : 1000,
    },
    "landscape": {
            "number_structures" : 1000,
            "spacegroup_set" : "default"
    },
    "reoptimization": {
        "smart_optimization_energy_window": 10
    },
    "compack": {
        "allow_artificial_inversion": True,
        "allow_molecular_differences": False,
        "angle_tolerance": 20,
        "distance_tolerance": 0.2,
        "ignore_bond_counts": False,
        "ignore_bond_types": True,
        "ignore_hydrogen_counts": False,
        "ignore_hydrogen_positions": True,
        "ignore_smallest_components": False,
        "match_entire_packing_shell": False,
        "molecular_similarity_threshold": 0.2,
        "packing_shell_size": 30,
        "show_highest_similarity_result": True,
        "skip_when_identifiers_equal": True
    },
    "descriptors" : {"pxrd" : "Platon"},
    "crest" : {},
}

OPT_SCHEMES = {
    # DMACRYS optimisation schemes from fastest (top) to most accurate (bottom)
    "fast&loosest" : {
        "LIMI" : "0.01",
        "LIMG" : "0.01",
        "MAXD"  : "1.0",
        "UDTE" : "1000000000"
    },
    "fast&very_loose" : {
        "LIMI" : "0.001",
        "LIMG" : "0.001",
        "MAXD"  : "1.0",
        "UDTE" : "1000000000"
    },
    "fast&loose" : {
        "LIMI" : "0.0001",
        "LIMG" : "0.0001",
        "MAXD"  : "1.0",
        "UDTE" : "1000000000"
    },
    "loose" : {
        "LIMI" : "0.0001",
        "LIMG" : "0.0001",
        "MAXD"  : "0.5",
        "UDTE" : "1000000000"
    },
    "precise" : {
        "LIMI" : "0.00001",
        "LIMG" : "0.00001",
        "MAXD"  : "0.5",
        "UDTE" : "0"
    },
    "very_precise" : {
        "LIMI" : "0.000001",
        "LIMG" : "0.0000000001",
        "MAXD"  : "0.5",
        "UDTE" : "0"
    }
}



COMMON_SAMPLING_SETTINGS = {
    "efficientz1": {
        # efficient sampling for organic semiconductors Z'=1 
        "space_group": {14, 19, 2, 4, 61, 15, 33, 9, 29},
        "number_structures": {
            14: 5000,
            19: 1000,
            2: 1000,
            4: 1000,
            61: 1000,
            15: 1000,
            33: 1000,
            9: 1000,
            29: 1000
        },
    },
    "efficientz2": {
        # efficient sampling for organic semiconductors Z'=2 
        "space_group": {2, 14, 4},
        "number_structures": {
            2: 12500,
            14: 7500,
            4: 7500
        },
    },
    "coarse10": {
        # 10 most common spacegroups
        "space_group": MOST_COMMON_SPACEGROUPS["single"][:10],
        "number_structures": {x: 1000 for x in MOST_COMMON_SPACEGROUPS["single"]},
    },
    "coarse25": {
        # 25 most common spacegroups
        "space_group": MOST_COMMON_SPACEGROUPS["single"],
        "number_structures": {x: 1000 for x in MOST_COMMON_SPACEGROUPS["single"]},
    },
    "fine10": {
        # 10 most common spacegroups
        "space_group": MOST_COMMON_SPACEGROUPS["single"][:10],
        "number_structures": {x: 10000 for x in MOST_COMMON_SPACEGROUPS["single"][:10]},
    },
    "fine25": {
        # 25 most common spacegroups
        "space_group": MOST_COMMON_SPACEGROUPS["single"],
        "number_structures": {x: 10000 for x in MOST_COMMON_SPACEGROUPS["single"]},
    },
    "co-crystals_fine": {
        # 10 most common spacegroups for co-crystals
        "space_group": {2, 14, 15, 4, 19, 61, 1, 33, 5, 9},
        "number_structures": {
            2: 10000,
            19: 10000,
            4: 20000,
            61: 20000,
            14: 50000,
            15: 50000,
            1: 10000,
            33: 20000,
            5: 20000,
            9: 20000,
        },
    },
    "fine5-wf": {
        # 5 most common spacegroups for fmcsp-wf
        "space_group": MOST_COMMON_SPACEGROUPS["single"][:5],
        "number_structures": {
            61: 100000,
            14: 100000,
            19: 100000,
            2: 100000,
            4: 100000,
        },
    },
    "nextfine5-wf": {
        # 5 next most common spacegroups for fmcsp-wf
        "space_group": MOST_COMMON_SPACEGROUPS["single"][5:10],
        "number_structures": {
            15: 50000,
            33: 50000,
            9: 50000,
            29: 50000,
            5: 50000,
        },
    },
    "fine10-wf": {
        # 10 most common spacegroups for fmcsp-wf
        "space_group": MOST_COMMON_SPACEGROUPS["single"][:10],
        "number_structures": {
            61: 100000,
            14: 100000,
            19: 100000,
            2: 100000,
            4: 100000,
            15: 50000,
            33: 50000,
            9: 50000,
            29: 50000,
            5: 50000,
        },
    },
    "co-crystals_coarse": {
        # 6 most common spacegroups for co-crystals
        "space_group": {2, 14, 15, 4, 19, 61},
        "number_structures": {2: 2500, 14: 5000, 15: 5000, 4: 2500, 19: 3000, 61: 3000},
    },
    "poroussalt": {
        "space_group": {1, 2, 33, 4, 5, 9, 14, 15, 19, 148, 61},
        "number_structures": {
            1: 10000,
            2: 10000,
            33: 20000,
            4: 20000,
            5: 20000,
            9: 20000,
            14: 50000,
            15: 50000,
            19: 10000,
            148: 50000,
            61: 20000,
        },
    },
    "btz2": {
        # Z'=2 sampling for blind test 2021
        "space_group": {2, 14, 4, 19, 1, 29, 33, 15, 61},
        "number_structures": {
            2: 20000,
            14: 50000,
            4: 20000,
            19: 20000,
            1: 10000,
            29: 5000,
            33: 5000,
            15: 5000,
            61: 5000,
        },
    },
    "chiral_fine": {
        # 10 most common spacegroups for enantiomerically pure crystals
        "space_group": {19, 4, 5, 1, 18, 76, 96, 92, 78, 145},
        "number_structures": {
            19: 50000,
            4: 40000,
            5: 20000,
            1: 20000,
            18: 15000,
            76: 10000,
            96: 10000,
            92: 10000,
            78: 10000,
            145: 10000,
        },
    },
    "chiral_finer": {
        # 10 most common spacegroups for enantiomerically pure crystals
        "space_group": {19, 4, 5, 1, 18, 76, 96, 92, 78, 145},
        "number_structures": {
            19: 100000,
            4: 80000,
            5: 40000,
            1: 40000,
            18: 30000,
            76: 20000,
            96: 20000,
            92: 20000,
            78: 20000,
            145: 20000,
        },
    },
    "chiral_fine_1_1": {
        # 10 most common spacegroups for enantiomerically pure crystals with Z'=2
        "space_group": {4, 19, 1, 5, 18, 76, 78, 144, 96, 145},
        "number_structures": {
            4: 50000,
            19: 40000,
            1: 20000,
            5: 20000,
            18: 15000,
            76: 10000,
            78: 10000,
            144: 10000,
            96: 10000,
            145: 10000,
        },
    },
    "chiral_fine_3": {
        # 10 most common spacegroups for enantiomerically pure crystals with Z'>2
        "space_group": {4, 19, 1, 5, 18, 76, 78, 144, 96, 145},
        "number_structures": {
            4: 50000,
            19: 40000,
            1: 20000,
            5: 20000,
            18: 15000,
            76: 10000,
            78: 10000,
            144: 10000,
            96: 10000,
            145: 10000,
        },
    },
}


OLD_ARGS_TO_NEW = {
    "gaussian": {
        "nprocs": "gaussian.nprocs",
        "functional": "gaussian.method",
        "basis_set": "gaussian.basis",
        "basis_file": "gaussian.basis_file",
        "memory": "gaussian.memory",
        "chelpg": "gaussian.chelpg",
        "gaussian_cleanup": "gaussian.cleanup",
        "molecular_volume": "gaussian.calculate_molecular_volume",
        "vol": "gaussian.calculate_molecular_volume",
        "polarizable_continuum": "gaussian.polarizable_continuum_model",
        "external_iteration_pcm": "gaussian.external_iteration_pcm",
        "dielectric_constant": "gaussian.dielectric_constant",
        "esp": "gaussian.dielectric_constant",
        "opt": "gaussian.optimisation_options",
        "gopt": "gaussian.optimisation_options",
        "iso": "gaussian.iso_value",
        "freq": "gaussian.frequency_options",
        "gaussian_all_molecules": "gaussian.all_molecules",
        "additional_args": "gaussian.extra_arguments",
        "set_molecular_states": "gaussian.molecular_states",
        "force_rerun": "gaussian.force_rerun",
    },
    "neighcrys": {
        "potential_file": "neighcrys.potential_filename",
        "bondlength_file": "neighcrys.bondlength_filename",
        "multipole_file": "neighcrys.multipole_filename",
        "axis_file": "neighcrys.axis_filename",
        "lattice_factor": "neighcrys.lattice_factor",
        "step_size": "neighcrys.step-size",
        "standardise_bonds": "neighcrys.standardise_bonds",
        "intermolecular_distance": "neighcrys.max_intermolecular_distance",
        "max_iterations": "neighcrys.max_iterations",
        "vdw_cutoff": "neighcrys.vdw_cutoff",
        "foreshorten_hydrogens": "neighcrys.foreshorten_hydrogens",
        "wcal_ngcv": "neighcrys.wcal_ngcv",
        "force_field_type": "neighcrys.forcefield_type",
        "neighcrys_input_mode": "neighcrys.input_mode",
        "max_search": "neighcrys.connectivity_search_max_cells",
        "check_anisotropy": "neighcrys.check_anisotropy",
        "custom_atoms_anisotropy": "neighcrys.custom_anisotropy",
        "custom_labels": "neighcrys.custom_labels",
        "pressure": "neighcrys.pressure",
        "limi": "neighcrys.limi",
        "limg": "neighcrys.limg",
        "gdma_dist_tol": "neighcrys.gdma_dist_tol",
        "paste_coords": "neighcrys.paste_coords",
        "paste_coords_file": "neighcrys.paste_coords_filename",
        "raise_before_neighcrys_call": "neighcrys.raise_before_call",
        "remove_symmetry_subgroup": "neighcrys.remove_symmetry_subgroup",
        "dummy_atoms": "neighcrys.dummy_atoms",
    },
    "dmacrys": {
        "assisted_convergence": "dmacrys.assisted_convergence",
        "clean_level": "dmacrys.clean_level",
        "zip_level": "dmacrys.zip_level",
        "zip_unpack": "dmacrys.zip_unpack",
        "constant_volume": "dmacrys.constant_volume",
        "dmacrys_timeout": "dmacrys.timeout",
        "remove_negative_eigens": "dmacrys.remove_negative_eigenvalues",
        "cutm": "dmacrys.cutm",
        "nbur": "dmacrys.nbur",
        "auto_neighbour_setting": "dmacrys.auto_neighbour",
        "check_z_value": "dmacrys.check_z",
        "new_ac": "dmacrys.use_new_assisted_convergence",
        "puc": "dmacrys.platon_unit_cell",
        "lj": "dmacrys.fit_lennard_jones",
        "lennard_jones": "dmacrys.fit_lennard_jones",
        "ljp": "dmacrys.lennard_jones_potential_filename",
        "lennard_jones_potential": "dmacrys.lennard_jones_potential_filename",
        "seig1": "dmacrys.seig1",
        "exact_prop": "dmacrys.exact_property",
        "raise_before_dmacrys_call": "dmacrys.raise_before_call",
        "setup_off": "dmacrys.setup_off",
        "setup_on": "dmacrys.setup_on",
        "spline_off": "dmacrys.spline_off",
        "spline_on": "dmacrys.spline_on",
        "set_real_space": "dmacrys.real_space",
        "platon_unit_cell": "dmacrys.platon_unit_cell",
        "timeout": 1800.0,
    },
    "gdma": {"multipole_limit": "gdma.l_max", "multipole_switch": "gdma.version_flag"},
    "crystal_generator": {
        "space_group": "crystal_generator.spacegroup",
        "number_structures": "crystal_generator.structure_count",
        "input_file": "crystal_generator.input_filename",
        "output_file": "crystal_generator.output_filename",
        "min_packing": "crystal_generator.minimum_packing",
        "frozen_position": "crystal_generator.freeze_position",
        "frozen_rotation": "crystal_generator.freeze_rotation",
        "frozen_angles": "crystal_generator.freeze_angles",
        "dummy_symmetry_operation": "crystal_generator.dummy_symop",
        "sobol_seed": "crystal_generator.sobol_seed",
        "min_angle": "crystal_generator.minimum_unit_cell_angle",
        "max_angle": "crystal_generator.maximum_unit_cell_angle",
        "max_repulsion_energy": "crystal_generator.maximum_repulsion_energy",
        "separating_axis_theorem": "crystal_generator.separating_axis_theorem",
        "target_volume": "crystal_generator.target_volume",
        "boxes": "crystal_generator.boxes",
        "max_volume": "crystal_generator.max_volume",
        "expand_cell": "crystal_generator.expand_cell",
        "number_processors": "crystal_generator.nprocs",
        "automatic_sobol_seed": "crystal_generator.automatic_sobol_seed",
        "squashing_param": "crystal_generator.squashing_parameter",
        "zip_generated_structures": "crystal_generator.zip_structures",
        "zip_generated_structures_single": "crystal_generator.zip_structures_single",
        "zip_generated_structures_filename": "crystal_generator.zip_structures_filname",
        "z_prime": "crystal_generator.z_prime",
        "old_cell_bounds": "crystal_generator.old_cell_bounds",
        "target_volume_parameter": "crystal_generator.target_volume_parameter",
        "max_volume_parameter": "crystal_generator.max_volume_parameter",
    },
}

DEFAULT_CONFIG_LOCATIONS = (
    os.path.join(os.path.dirname(__file__), "cspy_defaults.toml"),
    "cspy.toml",
)


class CspyConfiguration(object):
    _settings = None
    _runtime_settings = {}

    def __init__(self, args_dict=None):

        if not args_dict:
            args_dict = {}

        self.config_locations = tuple(x for x in DEFAULT_CONFIG_LOCATIONS if os.path.exists(x))
        self._defaults = DEFAULTS
        self._runtime_settings = recursive_dict_update(
            self._runtime_settings, args_dict
        )
        self.convert_to_new_setting_names()

        for loc in self.config_locations:
            with open(loc) as f:
                self._defaults = recursive_dict_update(self._defaults, toml.load(f))

        self._runtime_settings.update(self._defaults)
        self._settings = ChainMap(self._runtime_settings, os.environ)

    def save(self, filename="cspy.toml", settings="runtime"):
        """Save the current configuration to a file
        """
        if settings == "runtime":
            to_write = self._runtime_settings
        else:  # write all settings
            to_write = self._settings
        with open(filename, "w") as f:
            toml.dump(to_write, f)

    def __getitem__(self, key):
        return self._settings.__getitem__(key)

    def __setitem__(self, key, value):
        self._settings.__getitem__(key, value)

    def __delitem__(self, key):
        self._settings.__delitem__(key)

    def check_for_updates(self):
        defaults = DEFAULTS
        for loc in self.config_locations:
            with open(loc) as f:
                defaults = recursive_dict_update(defaults, toml.load(f))
        return defaults

    def set(self, setting_name, value):
        """ Set `setting_name` to be `value` in the underlying
        settings, where nesting levels are separated by `sep`,
        creating required levels as we go.

        >>> config = CspyConfiguration({})
        >>> config.set('tmp.level1.level2.val_bool', True)
        >>> config['tmp']['level1']['level2']['val_bool']
        True
        >>> config.set('tmp.level1.level2.val_float', 1.35)
        >>> config['tmp']['level1']['level2']['val_float']
        1.35
        """
        tmp = {}
        ref = tmp
        levels = setting_name.split(".")
        for x in levels[:-1]:
            ref[x] = {}
            ref = ref[x]
        ref[levels[-1]] = value
        self._runtime_settings = recursive_dict_update(self._runtime_settings, tmp)

    def get(self, setting_name, default_val=None, sep="."):
        """ Get the value for `setting_name` from the underlying
        settings, where levels are separated by `sep`. If at any
        level of nesting it is not defined, return `default_value`

        >>> config = CspyConfiguration({})
        >>> config.get('level.that.does.not.exist', 3.5)
        3.5
        >>> config.set('level.that.does.exist', 'sure does')
        >>> config.get('level.that.does.exist', 3.5)
        'sure does'
        >>> config['level']['that']['does']
        {'exist': 'sure does'}
        """
        tmp = self._settings
        levels = setting_name.split(sep)
        for level in levels:
            if level not in tmp:
                break
            tmp = tmp[level]
        else:
            return tmp
        return default_val

    def delete(self, setting_name, sep="."):
        """ Delete the value for `setting_name` from the underlying
        settings, where levels are separated by `sep`.

        >>> config = CspyConfiguration({})
        >>> config.set('level.that.does.exist', 'sure does')
        >>> config.get('level.that.does.exist')
        'sure does'
        >>> config.delete('level.that.does.exist')
        """
        nested_dict_delete(self._settings, setting_name, sep=sep)

    def convert_to_new_setting_names(self):
        for key in list(self._runtime_settings.keys()):
            key_done = False
            val = self._runtime_settings.pop(key)
            for section in OLD_ARGS_TO_NEW.keys():
                if key in OLD_ARGS_TO_NEW[section] and val is not None:
                    self.set(OLD_ARGS_TO_NEW[section][key], val)
                    key_done = True
            if not key_done:
                self.set(key, val)

if CONFIG is None:
    CONFIG = CspyConfiguration({})

def configure(args):
    global CONFIG
    if CONFIG is not None:
        args = recursive_dict_update(CONFIG._runtime_settings, args)
    else:
        CONFIG = CspyConfiguration(args)
    for k in ("SERVER_IP", "TIMEOUT", "PROCID"):
        os.environ[k] = str(CONFIG.get(k))
    return CONFIG
