import argparse

parser = argparse.ArgumentParser(description='Clusters conformers based on their torsional angles.')
# parser.add_argument('-db','--database_file', type=str, metavar='', help='Name of database file for clustering. holds a list of all conformers and their associated energies.')
parser.add_argument('-t','--torsions', type=str, default=False, metavar='', help='List of torsion angles for the molecule equal to the first conformer of the series.')
parser.add_argument('-np', '--processors', type=int, metavar='', default=1, help='Number of processors used.')
# parser.add_argument('-ewin', '--energy_window', type=float, metavar='', default=5.0, help='Energy window in which conformers should be compared.')
parser.add_argument('-o', '--overlay', action='store_true', help='Perform an overlay on molecules before clustering, recommended for some molecules.')
parser.add_argument('-r', '--reference', type=str, metavar='', default=False, help='Compare all structures in energies list to a reference structure.')
parser.add_argument('-e','--energies_file', type=str, metavar='', help='Name of energies file, holds a list of all conformers and their associated energies.')
args=parser.parse_args()

from torsions_func import *

def compare(ref_all_torsions, comp_all_torsions, to_compare):

    #non-mirrored
    highest_non_mirrored = 0
    for torsion in to_compare:
        xyz1_tor = get_torsion_angle(ref_all_torsions, torsion)
        xyz2_tor = get_torsion_angle(comp_all_torsions, torsion)
        equiv_tor = torsion[-1]
        dif = get_difference(xyz1_tor, xyz2_tor, equiv_tor)
        if dif > highest_non_mirrored:
            highest_non_mirrored = dif

    #mirrored
    highest_mirrored=0
    for torsion in to_compare:
        xyz1_tor = get_torsion_angle(inv_ref_torsions, torsion)
        xyz2_tor = get_torsion_angle(comp_all_torsions, torsion)
        equiv_tor = torsion[-1]
        dif = get_difference(xyz1_tor, xyz2_tor, equiv_tor)
        if dif > highest_mirrored:
            highest_mirrored = dif

    highest = min(highest_non_mirrored,highest_mirrored)
    return highest

#takes in reference
if __name__ == "__main__":
    #start time
    # print("Initialising...")
    
    #loading in arguments from argparse
    reference_xyz = args.reference
    # processors = args.processors
    torsions_file_str = args.torsions
    energies_file_str = args.energies_file
    
    ref_geom = get_geom(reference_xyz)
    to_compare = read_torsions(torsions_file_str)
    ref_torsions = get_torsions_f(ref_geom, to_compare)
    inv_ref_geom = get_inverted_geom(reference_xyz)
    inv_ref_torsions = get_torsions_f(inv_ref_geom, to_compare)
    file_energies, n_files = get_energies(energies_file_str)
    file_list = file_energies[0]
    all_conformer_torsions = get_conformer_torsions(file_list, to_compare)
    current_lowest = 180
    for xyz in file_list:
        # print(reference_xyz, xyz)
        comp_torsions=find_all_torsions_for_conformer(xyz, all_conformer_torsions)
        smallest_angle_where_all_torsions_within = compare(ref_torsions, comp_torsions, to_compare)
        if smallest_angle_where_all_torsions_within < current_lowest:
            current_lowest = smallest_angle_where_all_torsions_within
            current_lowest_xyz = xyz
    print(reference_xyz, current_lowest_xyz, current_lowest)
