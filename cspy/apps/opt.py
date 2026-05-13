import argparse
import logging
from pathlib import Path

from cspy import Crystal
from cspy.potentials import available_potentials
from cspy.util.logging_config import FORMATS, DATEFMT
from cspy.chem.multipole import DistributedMultipoles

LOG = logging.getLogger("cspy.opt")

def main():
    from cspy.configuration import CONFIG

    parser = argparse.ArgumentParser(
        description="Optimize crystal structures using various methods.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "crystal_structures", 
        nargs="+", 
        type=str, 
        help="Crystal structures to minimize"
    )
    parser.add_argument(
        "--basis-set",
        "-b",
        default="cc-pVTZ",
        help="Basis set for Gaussian09 calculation "
        "(a string passed verbatim into Gaussian "
        "input file, e.g. 6-311G**)",
    )
    parser.add_argument(
        "--method",
        "-m",
        default="PBEPBE",
        help="Method for Gaussian09 calculation "
        "(a string passed verbatim into Gaussian "
        "input file, e.g. B3LYP",
    )
    parser.add_argument(
        "-p",
        "--potential",
        default=CONFIG.get("csp.potential"),
        choices=available_potentials.keys(),
        help="intermolecular potential name",
    )
    parser.add_argument(
        "--charges",
        default=None,
        help="Charges of each molecule for G09 calculation",
    )   
    parser.add_argument(
        "--cutoff",
        default="calculate",
        help="dmacrys real space/repulsion-dispersion cutoff",
    )
    parser.add_argument(
        "--multipole-rank",
        "-r",
        default=4,
        type=int,
        help="Maximum angular momenta for DMA",
    )
    parser.add_argument(
        "--mulfit-rank",
        default=None,
        type=int,
        help="Maximum angular momenta for MULFIT",
    )
    parser.add_argument(
        "--mulfit-method",
        default="cumulative",
        choices=("cumulative", "overall"),
        help="Method for fitting in MULFIT",
    )
    parser.add_argument(
        "--MAXI",
        default=CONFIG.get("neighcrys.max_iterations"),
        help="Max number of iterations to use in DMACRYS",
    )
    parser.add_argument(
        "--single-point",
        action="store_true",
        default=False,
        help="Calculate a single point energy (don't optimize)",
    )
    parser.add_argument(
        "--multipole",
        default=None,
        help="Multipole file to be used in the calculations",
    )
    parser.add_argument(
        "--axis",
        type=str,
        default=None,
        help="Axis file to be used in the calculations",
    )
    parser.add_argument(
        "--reorder-atoms-if-high-rmsd",
        action="store_true",
        default=False,
        help="Try to reorder atoms to match multipoles if the multipole matching returns high RMSD (> 1e-3)",
    )
    parser.add_argument(
        "--reorder-method",
        type=str,
        default='molecular_axis',
        choices=("molecular_axis", "rdkit"),
        help="Method to reorder atoms if high RMSD is detected",
    )
    parser.add_argument(
        "--dmacrys-timeout",
        default=CONFIG.get("dmacrys.timeout"),
        type=float,
        help="Timeout for dmacrys calculations",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("INFO", "DEBUG", "WARN", "ERROR"),
        help="Logging verbosity",
    )
    parser.add_argument(
        "--outputs",
        default=False,
        action="store_true",
        help="Write the dmacrys output files to the working directory",
    )
    parser.add_argument(
        "--cleanup",
        default=True,
        action="store_true",
        help="Remove intermediate files (fchk, individual dma files)",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_false",
        dest="cleanup",
        help="Don't remove intermediate files (fchk, individual dma files)",
    )
    parser.add_argument(
        "--gaussian-cpus", "-j", default=2, type=int, help="Number of cpus for G09"
    )
    parser.add_argument(
        "--gaussian-mem", default="2GB", type=str, help="Memory setting for G09"
    )
    parser.add_argument(
        "--calculation-type",
        type=str,
        default="dmacrys",
        choices=["dmacrys", "dftb", "mace", "vasp"],
        help='''Specify the type of calculation to perform. Options include:    
        dmacrys: Perform a dmacrys calculation,
        dftb: Perform a dftb calculation,
        mace: Perform a mace calculation,
        vasp: Perform a vasp calculation.
        ''',
    )
    parser.add_argument(
        "--lattice-opt",
        action="store_true",
        default=False,
        help="Perform a full lattice optimisation with dftb",
    )
    parser.add_argument(
        "--vasp-calc",
        action="store_true",
        default=False,
        help="Perform calculation on the first molecule of the crystal structure. Vacuum=20",
    )
    parser.add_argument(
        "--mace-calc",
        action="store_true",
        default=False,
        help="Use this flag when running a dftb calculation to perform a full lattice optimisation",
    )
    parser.add_argument(
        "--mace_model",
        default='mace_mp',
        choices=("mace_mp", "mace-off", "mace_anicc", "/path/to/custom/mace.model"),
        help="Mace pre-trained model to use.",
    )

    parser.add_argument(
        "--dma-switch",
        default=4,
        type=str,
        help="dma switch",
    )
    parser.add_argument(
        "--pcm",
        default="",
        type=str,
        help="Use PCM model",
    )

    parser.add_argument(
        "--conformation",
        action="store_true",
        default=False,
        help="Perform calculation on the first molecule of the crystal structure. Vacuum=20",
    )

    args = parser.parse_args()
    logging.basicConfig(
        format=FORMATS[args.log_level], datefmt=DATEFMT, level=args.log_level
    )
    kwargs = vars(args)
    bondlength_cutoffs = {}
    if args.multipole is not None:
        mols = DistributedMultipoles.from_dma_file(args.multipole).molecules
        for mol in mols:
            for k, v in mol.neighcrys_bond_cutoffs().items():
                if k not in bondlength_cutoffs or v > bondlength_cutoffs[k]:
                    bondlength_cutoffs[k] = v      
    ase_kwargs={}
    if args.calculation_type == "mace":
        ase_kwargs = {
            "calculator_type": "mace",
            "mace": {
                "model": args.mace_model
            }
        }
    mapper = {
        'dmacrys':['minimize', kwargs],
        'dftb': ['minimize_with_dftb', kwargs],
        'mace': ['minimize_with_mace', ase_kwargs],
        'vasp': ['minimize_with_vasp', kwargs]
    }
    for crystal_file in args.crystal_structures:
        crystal = Crystal.load(crystal_file)
        if args.conformation:
            crystal = crystal.to_vasp_conformation_input()
        if args.calculation_type == "dmacrys":
            if bondlength_cutoffs:
                crystal.properties["bondlength_cutoffs"] = bondlength_cutoffs
            if args.multipole is None:
                suffix = ".".join(
                     (args.method, args.basis_set, ("dma" + str(args.multipole_rank)))
                 )
            else:
                suffix = ".".join((args.method, args.multipole))
                crystal.titl = Path(crystal_file).stem + "." + suffix

        LOG.info(f"Calculation type set to {args.calculation_type}")
        result = getattr(crystal, mapper[args.calculation_type][0])(**mapper[args.calculation_type][1])
        if not args.single_point:
            try:
                result.save(crystal.titl + ".opt.cif")
            except AttributeError as exc:
                LOG.error(f"{exc}: Cannot save structure from failed minimization.")
    

if __name__ == "__main__":
    main()

