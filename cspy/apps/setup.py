#!/usr/bin/env python
import logging
from cspy.util.path import Path
from cspy.minimize import GaussianMinimizer
from collections import namedtuple
from itertools import groupby, count
from operator import attrgetter
from cspy.chem import Molecule
from cspy.templates import get_template
import os

LOG = logging.getLogger("csp-setup")

StructureFile = namedtuple("StructureFile", "prefix id")


def split_components(filenames, suffix=".xyz"):
    structure_files = []
    for comps in (f.split("_") for f in filenames):
        if len(comps) > 1:
            head = "_".join(comps[:-1])
            tail = comps[-1][: -len(suffix)]
            if tail.isdigit():
                tail = int(tail)
            sf = StructureFile(head, tail)
        else:
            sf = StructureFile(comps, "")
        structure_files.append(sf)
    return structure_files


def as_range(iterable):
    l = list(iterable)
    if len(l) > 1:
        return "{0}-{1}".format(l[0], l[-1])
    return "{0}".format(l[0])


def merged_int_ranges(xyz_list):
    ids = sorted((x.id for x in xyz_list))
    grouped = groupby(ids, key=lambda n, c=count(): n - next(c))
    return ",".join(as_range(g) for _, g in grouped)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "structure_files",
        nargs="+",
        type=str,
        help="List of structure files to process",
    )
    parser.add_argument(
        "-f",
        type=str,
        choices=("fit", "w99", "w99rev_6311"),
        default="fit",
        help="The name of the force field to use",
    )
    parser.add_argument(
        "--workdir", default=os.getcwd(), help="working directory for the job"
    )
    parser.add_argument(
        "--suffix", "-s", default=".xyz", help="File suffix to strip for naming"
    )
    parser.add_argument(
        "--method",
        default="B3LYP",
        help="Gaussian calculation method e.g. B3LYP, HF, MP2",
    )
    parser.add_argument(
        "--basis-set", default="6-311G**", help="Gaussian basis set e.g. 3-21G etc."
    )
    parser.add_argument(
        "--job-type",
        "-j",
        choices=("dma", "opt", "sp"),
        default="opt",
        help="What kind of job array to set up e.g. dma, opt, sp",
    )
    parser.add_argument(
        "--queue-system",
        "-q",
        choices=("slurm", "pbs"),
        default="slurm",
        help="Which queue system for our script",
    )
    parser.add_argument("--memory", default="2GB", help="memory per job")
    parser.add_argument("--walltime", default="1:59:00", help="walltime per job")
    parser.add_argument(
        "--procs", default=1, help="number of processors per node to use"
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--route-commands", default="", help="Additional route commands")
    parser.add_argument("--scrf", default="", help="Continuum solvation model")
    parser.add_argument("--scrf-solvent", default="water", help="Specify solvent continuum model solvent")
    parser.add_argument(
        "--charge",
        default=0,
        help="The charge of the molecule",
    )
    parser.add_argument(
        "--multiplicity",
        default=1,
        help="The multiplicity to use in the Gaussian09 calculation",
    )

    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    comps = split_components(args.structure_files, suffix=args.suffix)

    # TODO avoid hardcoding potential file locations
    ff_settings = {
        "fit": {"force_field_type": "F"},
        "w99": {"force_field_type": "W"},
        "w99rev_6311": {"force_field_type": "W"},
    }
    force_field = ff_settings[args.f]
    additional_sections = []
    if args.scrf:
        LOG.info("Using solvation model '%s' with solvent '%s'", args.scrf, args.scrf_solvent)
        solvent = args.scrf_solvent
        if "=" in solvent:
            for line in solvent.split(";"):
                additional_sections.append(line)
            args.route_commands += f" scrf=({args.scrf},read)"
        else:
            args.route_commands += f" scrf=({args.scrf},solvent={args.scrf_solvent})"

    template = get_template("{}.{}.template".format(args.job_type, args.queue_system))
    suffix = args.suffix
    if args.job_type == "opt":
        suffix = ".com"
        LOG.info("Generating optimisation inputs")
        route_commands = (
            "opt=(modredundant,maxcycle=500,maxstep=29) "
            "geom=(printinputorient) empiricaldispersion=gd3bj "
            "scf=(maxcycle=65,tight)"
        ) + " " + args.route_commands
        
        gm = GaussianMinimizer(
            method=args.method, basis_set=args.basis_set, route_commands=route_commands,
            additional_sections=additional_sections
        )
        mols = [Molecule.load(i) for i in args.structure_files]
        Molecule.group_atoms_by_element(mols)
        for i, fname in enumerate(args.structure_files):
            mol = mols[i]
            p = Path(fname)
            name = p.stem
            LOG.debug("Mol: %s", mol)
            input_file = gm.generate_input_file(
                mol, name=name, charge=args.charge,
                multiplicity=args.multiplicity
            )
            p.with_suffix(".com").write_text(input_file)

    if args.job_type == "sp":
        suffix = ".com"
        LOG.info("Generating single-point inputs")
        route_commands = (
            "geom=(printinputorient) empiricaldispersion=gd3bj "
            "scf=(maxcycle=65,tight)"
        ) + " " + args.route_commands
        
        gm = GaussianMinimizer(
            method=args.method, basis_set=args.basis_set, route_commands=route_commands,
            additional_sections=additional_sections
        )
        mols = [Molecule.load(i) for i in args.structure_files]
        Molecule.group_atoms_by_element(mols)
        for i, fname in enumerate(args.structure_files):
            mol = mols[i]
            p = Path(fname)
            name = p.stem
            LOG.debug("Mol: %s", mol)
            input_file = gm.generate_input_file(
                mol, name=name, charge=args.charge,
                multiplicity=args.multiplicity
            )
            p.with_suffix(".com").write_text(input_file)

    if args.job_type == "dma":
        LOG.info("Setting up dma directories")
        for fname in args.structure_files:
            xyz_file = Path(fname)
            d = xyz_file.with_suffix("")
            d.mkdir()
            xyz_file.rename(d / xyz_file.name)

    for group, files in groupby(comps, key=attrgetter("prefix")):
        with open("{}.{}.sh".format(group, args.job_type), "w") as f:
            f.write(
                template.render(
                    walltime=args.walltime,
                    prefix=group,
                    procs=args.procs,
                    array_ids=merged_int_ranges(files),
                    workdir=args.workdir,
                    suffix=suffix,
                    mem=args.memory,
                    **force_field
                )
            )


if __name__ == "__main__":
    main()
