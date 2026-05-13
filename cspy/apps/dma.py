import argparse
import numpy as np
from cspy.util.path import Path, TemporaryDirectory
from cspy.chem.multipole import DistributedMultipoles
from cspy.templates import get_template
from cspy.executable import Gaussian, Mulfit, ReturnCodeError
from cspy.crystal import Crystal
from cspy.util.logging_config import FORMATS, DATEFMT
from cspy.chem.multipole import Multipole
from collections import Counter, defaultdict
import logging

G09_SCF_TEMPLATE = get_template("gaussian_scf")
G09_SCF_TEMPLATE_FLEX = get_template("gaussian_scf_flex")
LOG = logging.getLogger(__name__)

def run_gdma(fchk, **kwargs):
    fchk = Path(fchk).absolute()
    mult_file = fchk.with_suffix(".dma")
    from cspy.chem.multipole.dma import gdma

    LOG.info("%s: Running GDMA -> %s", fchk.stem, mult_file.name)
    if mult_file.exists():
        LOG.info("%s: Skipping dma, found %s ", fchk.stem, mult_file.name)
        return mult_file
    mults = gdma(fchk.read_text(), **kwargs)
    mults.to_dma_file(fchk.with_suffix(".dma"))
    return mult_file


def run_psi4(mol, 
             name, 
             charge=0, 
             multiplicity=1, 
             basis_set="6-311G**", 
             method="B3LYP",
             **kwargs
             ):
    try:
        import psi4
    except ImportError:
        LOG.info('ImportError: Does you python environment have psi4 installed?')
        exit()
    LOG.info("%s: Generating wavefunction with Psi4", name)
    psi4.core.clean_timers()
    psi4.set_output_file(name + ".log", 
                         append=False)
    log_file = Path(name + ".log")
    psi4.set_memory(kwargs.get('mem', '2048 MB'))
    psi4.set_num_threads(kwargs.get('nprocshared', 1))
    fchk_file = Path(f"{name}.fchk")
    if fchk_file.exists():
        LOG.info(
            "%s: Skipping psi4, found (%s)",
            name,
            fchk_file.name,
        )
        return log_file, fchk_file
    molec = psi4.core.Molecule.from_string(mol.to_xyz_string(),
                                           fix_com=True, 
                                           fix_orientation=True
    )
    molec.set_molecular_charge(charge)
    molec.set_multiplicity(multiplicity)
    options = {
        "reference":"uks" if multiplicity !=1 or charge !=0 else "rks",
        "dft_spherical_points":590,
        "dft_radial_points":99,
        "dft_pruning_scheme":"robust",
        "basis":basis_set,
        "MAXITER":100,
        "D_CONVERGENCE":6,
        "E_CONVERGENCE":6
    }
    pcm_value = kwargs.pop("pcm")
    if pcm_value:
        LOG.info(f"Including PCM string with PCM value of eps = {pcm_value}")
        pcm_string = f"""
            Units = Angstrom
            Medium {{
                SolverType = IEFPCM
                Solvent = Explicit
                ProbeRadius = 1.385 # water
                Green<Inside> {{
                    Type = Vacuum            
                }}
                Green<outside> {{
                    Type = UniformDielectric
                    eps = {pcm_value}
                    epsdyn = 1.777849 
                }}
            }}

            Cavity {{
                RadiiSet = UFF
                Type = GePol
                Scaling = False
                Area = 0.3
                Mode = Implicit
            }}
        """
        pcm_dict_settings={
          "pcm":True,
          "pcm_scf_type":"total",
          "pcm__input":pcm_string
        }
        for k,v in pcm_dict_settings.items():
            options[k] = v
    disp_method = kwargs.get('psi4_dispersion_method', None)
    method_string = '-'.join([method.lower(), disp_method.lower()]) if disp_method else method.lower()
    psi4.set_options(options)
    _, wfn = psi4.energy(method_string, 
                molecule=molec,
                return_wfn=True)
    psi4.fchk(wfn, f'{name}.fchk')
    psi4.core.clean()
    psi4.core.clean_variables()
    return log_file, fchk_file


def run_g09(
    mol, name, charge=0, multiplicity=1, basis_set="6-311G**", method="B3LYP", redundant="", constraints=[], flex_scf=False, **kwargs
):

    LOG.info("%s: Generating wavefunction with Gaussian09", name)
    route_commands = "NoSymm " + "EmpiricalDispersion=GD3BJ " + kwargs.pop("route_commands", "")
    additional_sections = [x for x in kwargs.pop("additional_sections", "").split(";") if x]
    pcm = kwargs.pop("pcm")
    if pcm:
        route_commands+=" SCRF=(PCM,ExternalIteration,Read)"
        additional_sections += ["Eps="+str(pcm)]
    else:
        LOG.debug("Not using pcm")
    link0 = {"chk": "{}.chk".format(name)}
    link0.update(kwargs)
    LOG.info("G09       route commands: %s", route_commands)
    LOG.info("G09            basis set: %s", basis_set)
    LOG.info("G09 exchange-correlation: %s", method)
    LOG.info("Additional sections:\n%s", "\n".join(additional_sections))
    # LOG.info(additional_sections)
    if method.lower() == "mp2":
        method = "mp2=semidirect"
        route_commands += " density=current"

    if flex_scf:
        if len(constraints) != 0:
            coordinate_types = {
                1:'X', 
                2:'B', 
                3:'A', 
                4:'D'
            }
            ct1 = []
            for item in constraints:
                atoms = tuple(
                    int(x) for x in item.split()
                )
                ct1 += [
                    f"{coordinate_types[len(atoms)]} {' '.join([str(atom) for atom in atoms])} F"
                ]
        else:
            ct1=constraints
        input_file = G09_SCF_TEMPLATE_FLEX.render(
            link0=link0,
            title=name + " for dma",
            method=method,
            basis=basis_set,
            charge=charge,
            multiplicity=multiplicity,
            redundant=redundant,
            constraints=ct1,
            geometry=mol.to_xyz_string(header=False),
            route_commands=route_commands,
            additional_sections=additional_sections,
        )
    else:
        input_file = G09_SCF_TEMPLATE.render(
            link0=link0,
            title=name + " for dma",
            method=method,
            basis=basis_set,
            charge=charge,
            multiplicity=multiplicity,
            geometry=mol.to_xyz_string(header=False),
            route_commands=route_commands,
            additional_sections=additional_sections,
        )

    LOG.debug("G09 input for %s:\n%s", name, input_file)
    log_file = Path(name + ".log")
    fchk_file = log_file.with_suffix(".fchk")
    if log_file.exists() and fchk_file.exists():
        LOG.info(
            "%s: Skipping Gaussian09, found (%s, %s)",
            name,
            log_file.name,
            fchk_file.name,
        )
        return log_file, fchk_file
    with TemporaryDirectory() as tmpdirname:
        LOG.debug("created temp directory: %s", tmpdirname)
        g09 = Gaussian(
            input_file,
            output_file=name + ".log",
            run_formchk=name + ".chk",
            working_directory=tmpdirname,
        )
        try:
            g09.run()
        except ReturnCodeError as e:
            LOG.error("Error running gaussian input:\n%s", input_file)
            raise RuntimeError("Error running g09") from e
        with log_file.open("w") as f:
            f.write(g09.log_contents)
        with fchk_file.open("w") as f:
            f.write(g09.fchk_contents)
    return log_file, fchk_file


def idx_to_char(i):
    if i < 26:
        start_character = "A"
    else:
        start_character = "a"
        i -= 26
    return chr(ord(start_character) + i)


def calculate_equivalent_molecules(oriented_mols, tolerance=1e-5):
    nmols = len(oriented_mols)
    equivalent_to = {}
    for i in range(nmols):
        if i in equivalent_to:
            continue
        equivalent_to[i] = None
        m1 = oriented_mols[i]
        for j in range(i + 1, nmols):
            m2 = oriented_mols[j]
            if len(m1) != len(m2):
                continue
            rmsd, plane = m1.rmsd_with_reflection(m2)
            if rmsd < tolerance:
                equivalent_to[j] = i

    if any(x is not None for x in equivalent_to.values()):
        d = defaultdict(list)
        for k, v in equivalent_to.items():
            if v is not None:
                d[idx_to_char(v)].append(idx_to_char(k))
        string_rep = "\n".join(f"{k} = {','.join(v)}" for k, v in d.items())
        LOG.info("Identical molecules:\n%s", string_rep)
    return equivalent_to

#    limit=4,

def generate_multipoles(
    crystal,
    ranks,
    mf_format,
    axis_file=None,
    potential_type="F",
    basis_set="6-311G**",
    method="B3LYP",
    dma_switch=4,
    xyz_filenames=None,
    foreshorten_hydrogens=None,
    hydrogen_radius=None,
    charges=None,
    multiplicities=None,
    constraints=None,
    redundant=None,
    flex_scf=False,
    limit=4,
    timeout_gdma=3600.0,
    **kwargs
):
    file_content = None
    if axis_file is not None:
        p = Path(axis_file)
        assert p.exists(), "Specified axis file does not exist"
        file_content = p.read_text()

    crystal.neighcrys_setup(potential_type=potential_type, file_content=file_content)
    axis_file = Path("{}.mols".format(crystal.titl))
    axis_file.write_text(crystal.neighcrys_axis.to_string())

    mp_filename = mf_format.format("")
    mulfit_method = kwargs.pop("mulfit_method", "cumulative")

    cleanup_files = []
    oriented_mols = [mol.oriented() for mol in crystal.symmetry_unique_molecules()]
    if xyz_filenames:
        for n, mol in zip(xyz_filenames, oriented_mols):
            LOG.info("Saving oriented %s -> %s", mol, n)
            mol.save(n)
    density = "MP2" if "mp2" in method.lower() else "SCF"
    LOG.info("Using %s density", density)
    equivalent_to = calculate_equivalent_molecules(oriented_mols)
    for i, j in equivalent_to.items():
        name = "{}_{}".format(crystal.titl, idx_to_char(i))
        mol = oriented_mols[i]
        LOG.debug("Oriented molecule %d:\n%s", i, mol.to_xyz_string())
        xyz_oriented = Path(name + ".xyz")
        xyz_oriented.write_text(mol.to_xyz_string())
        if j is not None:
            dma_j = "{}_{}.dma".format(crystal.titl, idx_to_char(j))
            dma_file = name + ".dma"
            LOG.info("Copy %s -> %s", dma_j, dma_file)
            Path(dma_j).copy_to(dma_file)
            cleanup_files += [dma_file, xyz_oriented]
        elif len(mol) == 1:
            LOG.info('skipping gdma for single atom, setting it as a point charge')
            charge = 0 if charges is None else int(charges.split()[i])
            dma_file = name + '.dma'
            DistributedMultipoles([[(
                mol.elements[0].symbol, mol.elements[0].symbol,
                np.array([0.0, 0.0, 0.0]), Multipole(rank=0, moments=[charge])
            )]]).to_dma_file(dma_file)
            cleanup_files += [dma_file, xyz_oriented]
        else:
            charge = 0 if charges is None else int(charges.split()[i])
            multiplicity = 1 if multiplicities is None else int(multiplicities.split()[i])
            if 'use_psi4' in kwargs:
                LOG.debug(f'Starting Psi4 setup of {name}')
                log_level = LOG.level
                log_file, fchk_file = run_psi4(mol, name, charge=charge, multiplicity=multiplicity, basis_set=basis_set,
                                               method=method, **kwargs
                ) # Psi4 has its own log settings which overwrite to error level during calculation
                logging.getLogger(__name__).setLevel(level=log_level)
                LOG.debug(f"Psi4 calculation of {name} completed")
            else:
                log_file, fchk_file = run_g09(
                    mol, name, basis_set=basis_set, method=method,
                    charge=charge, multiplicity=multiplicity, redundant=redundant, constraints=constraints, flex_scf=flex_scf, **kwargs
                )
            dma_file = run_gdma(
                fchk_file.absolute(),
                density=density,
                dma_switch=dma_switch,
                foreshorten_hydrogens=foreshorten_hydrogens,
                hydrogen_radius=hydrogen_radius,
                limit=limit,
                timeout=timeout_gdma,
            )

            cleanup_files += [dma_file, log_file, fchk_file, xyz_oriented]

        mults = DistributedMultipoles.from_dma_file(dma_file)
        LOG.debug("Re-labelling atom sites")
        labels = mol.properties["neighcrys_label"]
        mults.molecules[0].labels = labels
        with open(mp_filename, "a") as f:
            f.write(mults.to_dma_string())
            f.write("\n#ENDMOL\n")

        fit_multipoles_with_mulfit(
            dma_file, ranks, labels, mf_format, mulfit_method=mulfit_method
        )
    return cleanup_files


def fit_multipoles_with_mulfit(dma_file, ranks, labels, mf_format, **kwargs):
    mulfit_method = kwargs.get("mulfit_method", "cumulative")
    mf_suffix = "_rank{{}}{}".format("o" if mulfit_method == "overall" else "")
    for rank in ranks:
        LOG.info("%s: Fitting multipoles, LMAX = %d", Path(dma_file).name, rank)
        mults = run_mulfit(dma_file, rank, **kwargs).distributed_multipoles
        mults.molecules[0].labels = labels
        mp_filename = mf_format.format(mf_suffix.format(rank))
        LOG.debug("Appending to %s", mp_filename)
        with open(mp_filename, "a") as f:
            f.write(mults.to_dma_string())
            f.write("\n#ENDMOL\n")


def generate_combined_name(filenames):
    """
    >>> generate_combined_name(['ftriben_1.xyz'])
    'ftriben_1'
    >>> generate_combined_name(['ftriben_1.xyz', 'ftriben_2.xyz'])
    'ftriben_1.ftriben_2'
    >>> generate_combined_name(['ftriben_1.xyz', 'ftriben_1.xyz', 'ftriben_2.xyz'])
    'ftriben_1x2.ftriben_2'
    """
    stems = [Path(x).stem for x in filenames]
    counts = dict(Counter(stems))
    return ".".join(
        "{n}{s}".format(n=n, s="x{}".format(s) if s > 1 else "")
        for n, s in counts.items()
    )


def generate_combined_res(xyz_files, title, filename, sort):
#    LOG.info("Generating combined res file from: %s", xyz_files)
#    LOG.info("WILL %sgroup and sort atoms into expected order", "" if sort else "NOT ")
    crystal = Crystal.from_xyz_files(xyz_files, titl=title, sort_atoms=sort)
    crystal.save(filename)
    crystal.properties["radiation_source"] = "CSP"
    return crystal


def run_mulfit(mult_file, rank, **kwargs):
    mult_file = Path(mult_file)
    LOG.info("%s: Running MULFIT with LMAX = %d", mult_file.name, rank)
    with TemporaryDirectory() as tmpdirname:
        working_directory = Path(tmpdirname)
        mults = DistributedMultipoles.from_dma_file(mult_file)
        tmp_dma = working_directory / "tmp_mulfit_{}.dma".format(mult_file.stem)
        exe = Mulfit(
            mults,
            working_directory=working_directory,
            default_rank=rank,
            title=mult_file.stem,
            dma_file=tmp_dma,
            **kwargs,
        )
        mulfit_output = exe.run()
    return mulfit_output


def main_mulfit():
    parser = argparse.ArgumentParser()
    parser.add_argument("multipole_file", type=str, help="Smart multipole file")
    parser.add_argument(
        "--ranks", "-r", type=int, nargs="+", default=[0], help="Rank(s) to fit"
    )
    parser.add_argument(
        "--cleanup",
        "-c",
        default=False,
        action="store_true",
        help="Cleanup temporary files from mulfit",
    )
    parser.add_argument(
        "--log-level", default="INFO", help="Control level of logging output"
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level, format=FORMATS[args.log_level], datefmt=DATEFMT
    )

    mult_file = Path(args.multipole_file)
    name = mult_file.stem
    dest_dir = mult_file.parent
    kwargs = {}
    for rank in args.ranks:
        mults = run_mulfit(args.multipole_file, rank, **kwargs).distributed_multipoles
        mults.to_dma_file(dest_dir / "{}_rank{}.dma".format(name, rank))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("xyz_files", type=str, nargs="+", help="Molecule xyz files")
    parser.add_argument(
        "-l",
        "--multipole-limit",
        default=4,
        type=int,
        help="Maximum rank for DMA calculation"
    )
    parser.add_argument(
        "-p",
        "--potential-type",
        default="F",
        choices=("F", "W"),
        help="Potential type labels for neighcrys"
    )
    parser.add_argument(
        "-b",
        "--basis-set",
        default="6-311G**",
        help="Basis set for Gaussian09 calculation "
        "(a string passed verbatim into Gaussian input file, e.g. 6-311G**)"
    )
    parser.add_argument(
        "--method",
        default="B3LYP",
        help="Electronic structure method (usually DFT functional) for "
        "Gaussian09 calculation (a string passed verbatim into "
        "Gaussian input file, e.g. B3LYP or PBEPBE)"
    )
    parser.add_argument(
        "--charges",
        help="The charges of the molecules, leaving the option "
        "empty will set all molecules to have a charge of 0"
    )
    parser.add_argument(
        "--multiplicities",
        help="The multiplicities to use in the Gaussian09 "
        "calculations, leaving this option empty will set all "
        "molecules to have a multiplicity of 1"
    )
    parser.add_argument(
        "--ranks",
        "-r",
        type=int,
        nargs="+",
        default=[0],
        help="Rank(s) to fit with MULFIT"
    )
    parser.add_argument("-d", "--dma-switch", default=4, type=int, help="dma switch")
    parser.add_argument(
        "--axis-file",
        "-a",
        default=None,
        help="Provide an axis file to specify the " "molecular axes (EXPERIMENTAL)"
    )
    parser.add_argument(
        "--log-level", default="INFO", help="Control level of logging output"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        default=True,
        dest="cleanup",
        help="Cleanup additional unnnecessary files"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_false",
        dest="cleanup",
        help="Don't clean up additional unnnecessary files"
    )
    parser.add_argument(
        "--gaussian-cpus", "-j", default=1, type=int, help="Number of cpus for G09 or Psi4"
    )
    parser.add_argument(
        "--gaussian-mem", "-m", default="2GB", type=str, help="Memory setting for G09 or Psi4"
    )
    parser.add_argument(
        "--sort",
        default=True,
        action="store_true",
        help="Sort atoms within the molecules"
    )
    parser.add_argument(
        "--no-sort",
        action="store_false",
        dest="sort",
        help="Don't sort atoms within the molecules"
    )
    parser.add_argument(
        "--overwrite-xyz-files",
        action="store_true",
        default=True,
        help="Overwrite the xyz files"
    )
    parser.add_argument(
        "--no-overwrite-xyz-files",
        action="store_false",
        dest="overwrite_xyz_files",
        help="Don't overwrite the xyz files"
    )
    parser.add_argument(
        "--foreshorten_hydrogens",
        type=float,
        help="The length to foreshortening hydrogens"
    )
    parser.add_argument(
        "--hydrogen_radius",
        type=float,
        help="Hydrogen radius setting for gdma integration"
    )
    parser.add_argument(
        "--additional-route-commands",
        default="",
        help="Additional route commands for running g09"
    )
    parser.add_argument(
        "--additional-sections",
        default="",
        help="Additional sections to append to the end of g09 input"
    )
    parser.add_argument(
        "--pcm",
        default=None,
        help="pcm value"
    )
    parser.add_argument(
        "--psi4",
        action='store_true',
        help="Run DMA calculation with psi4 + GDMA"
    )
    parser.add_argument(
        "--psi4-dispersion-method",
        default='D3',
        help="Specify the dispersion method used in the Psi4 calculation. Defaults to Grimmes D3"
    )

    parser.add_argument(
        "--timeout-gdma",
        default=3600.0,
        type=float,
        help="Timeout for the GDMA calculation",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level, format=FORMATS[args.log_level], datefmt=DATEFMT
    )
    logging.getLogger(__name__).setLevel(args.log_level)
    kwargs = {
        "nprocshared": args.gaussian_cpus,
        "mem": args.gaussian_mem,
        "pcm": args.pcm,        
    }
    if args.psi4:
        kwargs["use_psi4"] = True
        kwargs["psi4_dispersion_method"] = args.psi4_dispersion_method
    if args.potential_type == "W" and args.foreshorten_hydrogens is None:
        args.foreshorten_hydrogens = 0.1
        LOG.info(
            "Reducing hydrogen bond lengths by: %.2f", args.foreshorten_hydrogens
        )
    title = generate_combined_name(args.xyz_files)
    combined_res_filename = "{title}.res".format(title=title)
    mf_format = title + "{}.dma"
    crystal = generate_combined_res(
        args.xyz_files, title, combined_res_filename, args.sort
    )

    cleanup_files = generate_multipoles(
        crystal,
        args.ranks,
        mf_format,
        potential_type=args.potential_type,
        axis_file=args.axis_file,
        basis_set=args.basis_set,
        method=args.method,
        dma_switch=args.dma_switch,
        foreshorten_hydrogens=args.foreshorten_hydrogens,
        hydrogen_radius=args.hydrogen_radius,
        xyz_filenames=args.xyz_files if args.overwrite_xyz_files else None,
        route_commands=args.additional_route_commands,
        additional_sections=args.additional_sections,
        charges=args.charges,
        multiplicities=args.multiplicities,
        limit=args.multipole_limit,
        timeout_gdma=args.timeout_gdma,
        **kwargs
    )
    cleanup_files.append(combined_res_filename)
    if args.cleanup:
        for fname in cleanup_files:
            f = Path(fname)
            if f.exists():
                f.unlink()


if __name__ == "__main__":
    main()
