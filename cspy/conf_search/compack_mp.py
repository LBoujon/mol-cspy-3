# Change log - 28/10/2021:
# #######################################################################
#
# 1. added restart capabilities - need to know how many calculations have
#    already run which can usually be obtained by knowing the length of
#    the rmsd_uniquestr.txt file plus the length of rmsd_matches.txt.
#
# 2. added density threshold arguement
#
# 3. added match_from_list arguement - can give a second input file which will
#    compare each listed structure against the structures in the initial input
# #######################################################################


import argparse
from functools import partial
from collections import defaultdict
from multiprocessing import Pool
from itertools import combinations as itc
import logging
from numpy import absolute

logging.basicConfig(filename="debug_log.txt",
                    filemode="w+",
                    level=logging.DEBUG)
LOG = logging.getLogger(__name__)

try:
    from ccdc.crystal import PackingSimilarity
    from ccdc.io import CrystalReader
    from ccdc.descriptors import CrystalDescriptors
except ImportError:
    import sys
    LOG.error("csd-pyhon-api package is not installed. "
              "Follow installation instructions here: https://downloads.ccdc.cam.ac.uk/documentation/API/installation_notes.html")
    sys.exit(1)

def read_structures(input_file):

    structures = defaultdict(list)
    with open(input_file, 'r') as f:
        structures = {line.split()[0]:[line.split()[1], line.split()[2]] 
                      for line in f}

    return structures
    
def compack_calculation(inputs, nmolecules=30, distance_tol=0.2, angle_tol=20,
                        powder_check=False, exp_match=False,
                        include_hydrogens=False, match_from_list=False):
    (reference, comparison, reference_energy, comparison_energy) = inputs

    if reference == comparison:
        return 0
    
    try:
        crys1 = CrystalReader(reference)[0]
    except:
        csd = CrystalReader('csd')
        crys1 = csd.crystal(reference)

    crys2 = CrystalReader(comparison)[0]

    if powder_check:
        powder_crys1 = CrystalDescriptors.PowderPattern.from_crystal(crys1)
        powder_crys2 = CrystalDescriptors.PowderPattern.from_crystal(crys2)
        pxrd_sim = powder_crys1.similarity(powder_crys2)
        LOG.debug("Powder similarity for {} and {} = {}".format(reference,
                                                                comparison,
                                                                pxrd_sim))
        if pxrd_sim <= 0.8:
            return 0

    sim = PackingSimilarity()
    sim.settings.distance_tolerance = distance_tol
    sim.settings.angle_tolerance = angle_tol
    sim.settings.ignore_hydrogen_positions = False if include_hydrogens else True

    n = 4
    while n < nmolecules:
        n = min(round(n * 1.618), nmolecules)
        sim.settings.packing_shell_size = n
        com = sim.compare(crys1, crys2)
        if com is None:
            with open("rmsd_uniquestr.txt", "a") as f:
                f.writelines("No matches - {} {} {} {} {}/{}\n".format(
                                                             reference,
                                                             reference_energy,
                                                             comparison,
                                                             comparison_energy,
                                                             0, n))
            return 0

        elif com.nmatched_molecules != n:
            nmatch_mol = com.nmatched_molecules
            rmsd = com.rmsd
            with open("rmsd_uniquestr.txt", "a") as f:
                f.writelines("{} {} {} {} {}/{} {}\n".format(reference,
                                                    reference_energy,
                                                    comparison,
                                                    comparison_energy,
                                                    nmatch_mol, n, rmsd))
            return 0

        nmatch_mol = com.nmatched_molecules
        rmsd = com.rmsd

    with open("rmsd_matches.txt", "a") as f:
        f.writelines("{} {} {} {} {}/{} {}\n".format(reference,
                                                      reference_energy,
                                                      comparison,
                                                      comparison_energy,
                                                      nmatch_mol, n, rmsd))

        if exp_match:
            return comparison
        
        elif match_from_list:
            return reference
        
        else:
            test = float(reference_energy) > float(comparison_energy)
            return reference if test else comparison


def write_to_file(input_list, filename):
    with open(filename, "w") as f:
        f.writelines("".join(item + "\n") for item in sorted(input_list))

    return 0


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "str_list",
        type=str,
        help="A file containing the name of each structure, its lattice energy " 
             "and its density."
        )
    
    parser.add_argument(
        "--exp_match",
        action='store_true',
        help="Compare inputs against an experimental crystal structure",
        default=False
        )
    
    parser.add_argument(
        "--exp_str",
        type=str,
        help="File name containing the experimental crystal structure."
             "Alternatively, specify the CCDC reference code."
             "e.g. name.cif, name.res, ACETAC01"
        )
    
    parser.add_argument(
        "-j",
        "--num_threads",
        type=int,
        help="Number of parallel threads to use when doing the comparisons",
        default=1
        )
    
    parser.add_argument(
        "--e_threshold",
        type=float,
        help="Energy threshold used in clustering within which structrues are "
             "considered the same",
        default=1.00
        )
    
    parser.add_argument(
        "--dens_threshold",
        type=float,
        help="Density threshold used in clustering within which structrues are "
             "considered the same",
        default=0.05
        )
    
    parser.add_argument(
        "--nmolecules",
        type=int,
        help="Number of molecules in the compack cluster",
        default=30
        )
    
    parser.add_argument(
        "--distance_tol",
        type=float,
        help="Distance tolerance in the compack calculation as a decimal",
        default=0.2
        )
    
    parser.add_argument(
        "--angle_tol",
        type=int,
        help="Angle tolerance in the compack calculation in degrees",
        default=20
        )
    
    parser.add_argument(
        "--include_hydrogens",
        action='store_true',
        help="This will include hydrogens in compack calculation",
        default=False
        )
    
    parser.add_argument(
        "--powder_check",
        action='store_true',
        help="Calculate PXRD patterns for reference and comparison structures. "
             "If similarity < 0.8, the structures are assumed to be duplicates.",
        default=False
        )
    
    parser.add_argument(
        "--match_from_list",
        type=str,
        help="Specify the name of a second input file which is used to compare"
             " against all structures in the first input file. (e.g. comparing"
             " a list of Z'=2 structures against Z'=1 structures) This can be"
             " thought of as a looped exp_match scenario. Returns a list of"
             " structures in the second input file that match structures listed"
             " in the first input file.",
        default=None
        )    
    parser.add_argument(
        "--show_progress",
        action='store_true',
        help="This will output a progress bar (Requires tqdm package)",
        default=False
        )
    
    parser.add_argument(
        "--restart",
        type=int,
        help="Specify position to restart calculations from",
        default=False
        )

    args = parser.parse_args()
    LOG.debug("Arguement list")
    for item in vars(args).items():
        LOG.debug("{}\n".format(item))
    
    structures = read_structures(args.str_list)
    if args.exp_match:
        LOG.debug("Running exp_match with {} cores\n".format(args.num_threads))
        
        calculations = [(args.exp_str, x, structures[x][0], 
                         structures[x][0])
                    for x in structures.keys()]
    
    elif args.match_from_list:
        LOG.debug("Running match_from_list with {} cores\n".format(args.num_threads))
        str_to_match = read_structures(args.match_from_list)
        calculations = [[(x, y, str_to_match[x][0], structures[y][0])
                         for y in structures.keys()] 
                        for x in str_to_match.keys()]
        
        calculations = [x for item in calculations for x in item] #flattern
        
    
    else:
        calculations = [(x[0], x[1], structures[x[0]][0],
                        structures[x[1]][0]) for x in
                        itc(structures.keys(), r=2) if 
                        absolute(float(structures[x[0]][0]) - 
                                 float(structures[x[1]][0])) <= args.e_threshold
                        and absolute(float(structures[x[0]][1]) - 
                                 float(structures[x[1]][1])) <= args.dens_threshold]
    
    if args.restart:
            calculations = calculations[args.restart:]
        
    
    total = len(calculations)  
    LOG.debug("Length of calculations = {}".format(total))
    LOG.debug("First 10 items in calculations = {}".format(calculations[:10]))
    
    with Pool(int(args.num_threads)) as pool:
        compack = partial(compack_calculation,
                        nmolecules=args.nmolecules,
                        distance_tol=args.distance_tol,
                        angle_tol=args.angle_tol,
                        include_hydrogens=args.include_hydrogens,
                        powder_check=args.powder_check,
                        exp_match=args.exp_match,
                        match_from_list=args.match_from_list
                        )
        if args.show_progress:
            from tqdm import tqdm
            duplicates = list(tqdm(pool.imap(compack, calculations),
                               ascii=True, total=total))
        else:
            duplicates = pool.map(compack, calculations)
            
    duplicates = set(x for x in duplicates if x != 0)
    
    if args.exp_match:
        write_to_file(duplicates, "experimental_matches.txt")
    
    else:
        write_to_file(duplicates, "duplicates.txt")
    

if __name__ == "__main__":
    main()





