import argparse
import multiprocessing

parser = argparse.ArgumentParser(description='Clusters conformers based on their torsional angles.')
parser.add_argument('-e','--energies', type=str, metavar='', help='Name of energies file, holds a list of all conformers and their associated energies.')
parser.add_argument('-t','--torsions', type=str, default=False, metavar='', help='List of torsion angles for the molecule equal to the first conformer of the series atoms should be indexed from 1. eg. 1 2 3 4 2. The final integer gives the symmetry of a torsion.')
parser.add_argument('-np', '--processors', type=int, metavar='', default=1, help='Number of processors used.')
parser.add_argument('-ewin', '--energy_window', type=float, metavar='', default=5, help='Energy window in which conformers should be compared.')
parser.add_argument('-rms', '--max_rms_angle', type=float, metavar='', default=5.0, help='Maximum rms angle between selected torsions.')
parser.add_argument('-smax', '--single_angle_max', type=float, metavar='', default=10.0, help='Maximum single torsion angle between selected torsions.')
parser.add_argument('-o', '--overlay', action='store_true', help='First perform an overlay between conformations.')
# parser.add_argument('-test', '--test', type=str, metavar='', default=False, help='Test script on two xyzs.')
# parser.add_argument('-r', '--reference', type=str, metavar='', default=False, help='Compare all structures in energies list to a reference structure.')
parser.add_argument('-c', '--compare', type=str, metavar='', default=False, help='Compare two lists together.')

args=parser.parse_args()

#Import libaries
import sys, math
from webbrowser import get
from operator import itemgetter
from multiprocessing import Pool, Process
import datetime
from os import path,remove
from itertools import islice
import numpy as np
from cspy.conf_search import torsions_func
import cspy

begin_time = datetime.datetime.now()

#inputs name of torsions file
def read_torsions(torsion_file):
    with open(torsion_file) as tor:
        lines = tor.readlines()
        comparing_torsions = []
        for line in lines:
            int_list=[]
            for atom_no in line.split():
                int_list.append(int(atom_no))
            comparing_torsions.append(int_list)
    return comparing_torsions

# read file data into a 2-d array
def get_file_string_array(file_name):
    try:
        file = open(file_name, "r")
    except IOError:
        print('Error: file (%s) not found!\n' % (file_name))
        sys.exit()
    lines = file.readlines()
    file.close()
    array = []
    for line in lines:
        array.append(line.split())
    return array

#take in a list of files and energies
# def get_energies(energies_file_name):
#     energies_array = get_file_string_array(energies_file_name)
#     energies_array = sorted(energies_array, key=itemgetter(1), reverse=True)
#     n_files=len(energies_array)
#     file_names=['' for i in range(n_files)]
#     energies = [0.0 for i in range(n_files)]
#     for i in range(n_files):
#         file_names[i] = energies_array[i][0].split(".xyz")[0]+".xyz" #can change this to if statement
#         energies[i] = float(energies_array[i][1])
#     file_energies=[file_names, energies]
#     return file_energies, n_files

def get_torsion(xyz, atom_list, invert=False):
    if invert == True:
        geom = torsions_func.get_inverted_geom(xyz)
    else:
        geom = torsions_func.get_geom(xyz)

    bond_graph = torsions_func.get_bond_graph(geom)
    at_types, coords = geom[0:2]
    i = atom_list[0] - 1
    j = atom_list[1] - 1
    k = atom_list[2] - 1
    l = atom_list[3] - 1
    t1234 = torsions_func.get_t1234(coords[i], coords[j], coords[k], coords[l])
    return t1234

#calculated the difference between two torsion angles takes into account periodicity of multiplicity of certain torsions angles.
def get_difference(torsion_angle1, torsion_angle2, equiv_torsions):
    dif = min(abs((360/equiv_torsions) - abs(torsion_angle1 - torsion_angle2)), abs(torsion_angle1 - torsion_angle2), abs((360) - abs(torsion_angle1 - torsion_angle2)), abs(360/equiv_torsions - abs(torsion_angle1 + torsion_angle2)), abs(360 - abs(torsion_angle1 + torsion_angle2)))
    return dif

#complutes the rms angle between a list of angular differences
def rms_angle(differences):
    dif_squared=[]
    for dif in differences:
        dif_squared.append(dif**2)
    mean_squared=sum(dif_squared)/len(dif_squared)
    root_mean_squared=math.sqrt(mean_squared)
    return root_mean_squared

def get_new_torsions_from_overlay(xyz_file_name1, xyz_file_name2, torsions):
    m1 = cspy.Molecule.load(xyz_file_name1)
    m2 = cspy.Molecule.load(xyz_file_name2)
    m3, order, rmsd = m1.overlay(m2)
    # print(order)
    new_torsions=[]
    for torsion in torsions:
        atoms=torsion[:-1]
        new_atoms=[]
        new_torsion=[]
        for atom in atoms:
            new_atom_index = order[atom-1]
            new_atom = new_atom_index+1
            new_atoms.append(new_atom)
            # print(atom, new_atom)
            new_torsion = new_atoms
        new_torsion.append(torsion[-1])
        new_torsions.append(new_torsion)
    return new_torsions

#compares two xyz files
def compare(xyz_file_name1, xyz_file_name2, torsions, single_angle_max, max_rms_angle, invert_ref=False):

    if args.overlay:
        new_torsions=get_new_torsions_from_overlay(xyz_file_name1, xyz_file_name2, torsions)
        # if torsions == new_torsions:
            # print("Atom ordering did not change after over overlay.")
        # else:
            # print("OLD:", torsions)
            # print("NEW:", new_torsions)
    else:
        new_torsions=torsions

    #load in molecules and set up test array
    unique_array=[1 for i in range(len(torsions)+1)] # if unique = 1, conformers are unique
    test_no=-1
    # xyz1 = cspy.Molecule.load(xyz_file_name1)
    # xyz2 = cspy.Molecule.load(xyz_file_name2)
    differences=[]


    if args.overlay:
        number_of_torsions = len(torsions)
        for k in range(number_of_torsions):
            #compare torsion1 with new_torsion1
            atoms1=torsions[k][:-1]
            atoms2=new_torsions[k][:-1]
            test_no+=1
            dihedral1 = get_torsion(xyz_file_name1, atoms1, invert=invert_ref)
            dihedral2 = get_torsion(xyz_file_name2, atoms2)
            # dihedral1 = xyz1.dihedral_angle(atoms)
            # dihedral2 = xyz2.dihedral_angle(atoms)
            equiv_tor = torsions[k][-1]
            dif = get_difference(dihedral1, dihedral2, equiv_tor)
            # print(xyz_file_name1, xyz_file_name2)
            # print(torsion, dihedral1, dihedral2, equiv_tor, dif)
            differences.append(dif)
            if abs(dif) < single_angle_max:
                unique_array[test_no]=0
    else:
        #perform comparisons for each torsion angle in torsions
        for torsion in torsions:
            atoms=torsion[:-1]
            test_no+=1
            dihedral1 = get_torsion(xyz_file_name1, atoms, invert=invert_ref)
            dihedral2 = get_torsion(xyz_file_name2, atoms)
            # dihedral1 = xyz1.dihedral_angle(atoms)
            # dihedral2 = xyz2.dihedral_angle(atoms)
            equiv_tor = torsion[-1]
            dif = get_difference(dihedral1, dihedral2, equiv_tor)
            # print(xyz_file_name1, xyz_file_name2)
            # print(torsion, dihedral1, dihedral2, equiv_tor, dif)
            differences.append(dif)
            if abs(dif) < single_angle_max:
                unique_array[test_no]=0
    
    #calculate rms angle across all torsion angles in torions
    rms = rms_angle(differences)
    if rms < max_rms_angle:
        unique_array[-1] = 0

    #test if all test have been passed.
    if unique_array == [0 for i in range(len(torsions)+1)]:
        return 0
        
    else:
        #test failed, conformers are unique.
        return 1

#determines the rank of the xyz within the list of file energies.
def get_rank(xyz_file_name, file_energies):
    n=0
    for file in file_energies[0]:
        n+=1
        if file == xyz_file_name:
            return n

#returns a list of ranks which have the same name of file.
def get_ranks(xyz_file_name, file_energies):
    n=0
    ranks=[]
    for file in file_energies[0]:
        n+=1
        if file == xyz_file_name:
            ranks.append(n)
            return ranks

def get_comparison_list(xyz_file_name, file_energies, energy_threshold):
    rank=get_rank(xyz_file_name, file_energies)
    xyz_energy=file_energies[1][rank-1]
    all_files=file_energies[0]
    all_energies=file_energies[1]
    all_files_higher_energy=all_files[rank:]
    all_energies_higher_energy=all_energies[rank:]
    comparison_list=[]
    limit=xyz_energy+energy_threshold
    #add energy threshold
    for i in range(len(all_files_higher_energy)):
        energy=all_energies_higher_energy[i]
        if energy <= limit:
            comparison_list.append(all_files_higher_energy[i])
        else:
            break
    return comparison_list

#compare xyz file to conformers within energy threshold.
def compare_against_list(ref_xyz, xyz_comparison_list, torsions, matched_list, single_angle_max=args.single_angle_max, max_rms_angle=args.max_rms_angle):
    for xyz in xyz_comparison_list: #this list should be split between processors
        result = compare(ref_xyz, xyz, torsions, single_angle_max, max_rms_angle)
        n=0
        if result == 0:
            matched_list[xyz] = xyz
        elif compare(ref_xyz, xyz, torsions, single_angle_max, max_rms_angle, invert_ref=True) == 0:
            matched_list[xyz] = xyz
        # print(matched_list)
    return matched_list

def compare_against_list_mp_Process(xyz_file_name, comparison_list, torsions, processors=args.processors):
    manager = multiprocessing.Manager()
    matched_list = manager.dict()
    processes=[]
    for n in range(processors):
        to_do_list=comparison_list[n::processors]
        p = multiprocessing.Process(target=compare_against_list, args=(xyz_file_name, to_do_list, torsions, matched_list,))
        p.start()
        processes.append(p)
    for process in processes:
        process.join()
    return matched_list.values()

def eng_compare(list_of_xyzs, list_of_energies): #outputs the file which is lowest in energy
    lowest = min(list_of_energies)
    lowest_index = list_of_energies.index(lowest)
    return list_of_xyzs[lowest_index]

def get_energy(xyz, file_energies):
    for file in file_energies[0]:
        print(xyz, file)
        if file == xyz:
            index = file_energies[0].index(xyz)
            return file_energies[1][index]

#combines the a list of files.
def combine_files(list_of_filenames, combined_name):
    with open(str(combined_name), 'w') as outfile:
        for fname in list_of_filenames:
            with open(fname) as infile:
                outfile.write("{}\n".format(infile.read()))

#take in a list of files and energies
def get_energies(energies_input):
    list_of_energies_files = energies_input.split(' ')
    all_file_names = []
    all_energies = []
    all_file = []
    for energies_file_name in list_of_energies_files:
        print("Getting energies for:", energies_file_name)
        energies_array = get_file_string_array(energies_file_name)
        n_files=len(energies_array)
        file_names = ['' for i in range(n_files)]
        energies = [0.0 for i in range(n_files)]
        file = ['' for i in range(n_files)]

        for i in range(n_files):
            file_names[i] = energies_array[i][0].split(".xyz")[0]+".xyz" #can change this to if statement
            energies[i] = float(energies_array[i][1])
            file[i] = energies_file_name
        
        all_file_names += file_names
        all_energies += energies
        all_file += file

    all_file_energies=[all_file_names, all_energies, all_file]
    all_n_files=len(all_file_energies[0])
    all_file_energies = np.array(all_file_energies)
    all_file_energies = all_file_energies[:, all_file_energies[1, :].argsort()[::-1]]
    all_file_energies = all_file_energies.tolist()
    for i in range(all_n_files):
        all_file_energies[1][i] = float(all_file_energies[1][i])
    all_n_files=len(all_file_energies[0])
    return all_file_energies, all_n_files

# def list_compare2(lists, torsions):
#     list_array = lists.split(' ')
#     list1 = list_array[0]
#     list2 = list_array[1]
#     combine_lists
#     list1_files, n1_files = get_energies(list1)
#     list2_files, n2_files = get_energies(list2)
#     list_1_only=[]
#     list_1_matched = []
#     list_2_matched = []


# def list_compare(lists, torsions):
#     #compare every member of list 1 with every member of list 2
#     list_array = lists.split(' ')
#     list1 = list_array[0]
#     list2 = list_array[1]
#     list1_files, n1_files = get_energies(list1)
#     list2_files, n2_files = get_energies(list2)
#     list_1_only=[]
#     list_1_matched = []
#     list_2_matched = []
#     comparison_list = []
#     for item in list1_files[0]:
#         reference_matches = compare_against_list_mp_Process(item, list2_files[0], torsions)
#         print(f"{item} matched to {reference_matches}.")
#         if len(reference_matches) == 0:
#             list_1_only.append(item)
#             comparison_list += [item, list1]
#         else:
#             list_1_matched += item
#             list_2_matched += reference_matches
#             item_energy=get_energy(item, list1_files)
#             all_files=[item]
#             all_energies=[item_energy]
#             for xyz in reference_matches:
#                 all_files.append(xyz)
#                 energy=get_energy(xyz, list2_files)
#                 all_energies.append(energy)
            
#             lowest_eng = eng_compare(all_files, all_energies)
#             comparison_list += [lowest, both]
#             all_files = reference_matches.append(item)


    # print("list_1_only is:", list_1_only)
    # list_1_matched = list(set(list1_files[0]) - set(list_1_only))
    # list_2_matched = [I for n, I in enumerate(list_2_matched) if I not in list_2_matched[:n]] #remove duplicate items of list

    # print("list_2_matched is:", both_lists)
    # list_2_only = list(set(list2_files[0]) - set(list_2_matched))
    # print("list_2_only is:", list_2_only)
 
    # with open('comparison_results.txt', 'w') as comp:
    #     for xyz in list_1_only:
    #         comp.write(f"{xyz} {get_energy(xyz, list1_files)}, {list1}")
        # for xyz in list_2_only:

#if reference matches file is not an empty list, the file has matched with an item from the other list (both)
#if reference matches file is an empty list, no file has matched, the file in the first list is unique (list1)
#where an item from list2 hasn't been matched, full_list2 - matched_list = blongs to list2 only

def main():
    torsions = read_torsions(args.torsions)
    files, n_files = get_energies(args.energies)
    uniques = [1 for i in range(len(files[0]))]
    unique_list=[files[0], uniques, files[2]]
    n=0
    for xyz_file_name in files[0]:
        exists_in = [files[2][n]]
        n+=1
        print(f"working on molecule {n} out of {n_files}.\r")
        #check if conformer has already been matched
        rank=get_rank(xyz_file_name, files)

        if unique_list[1][rank-1] == 0:
            continue
        else:
            comparison_list = get_comparison_list(xyz_file_name, files, args.energy_window)
            matched_list = compare_against_list_mp_Process(xyz_file_name, comparison_list, torsions)
            for xyz in matched_list:
                comparison_ranks=get_ranks(xyz, files)
                for comparison_rank in comparison_ranks:
                    unique_list[1][comparison_rank-1] = 0
                    exists_in.append(unique_list[2][comparison_rank-1])
            exists_in = list(dict.fromkeys(exists_in))

            if len(exists_in) > 1:
                print(f"Changed {xyz_file_name} to BOTH.")
                unique_list[2][rank-1] = "BOTH"

            print(f"{len(comparison_list)} comparisons were made.")
            print(f"{xyz_file_name} matched to {matched_list}")
            print(f"{xyz_file_name} has matched to files in {exists_in}:", unique_list[2][rank-1])


    print(f"FINAL UNIQUES LIST: {unique_list[1]}")

    if path.exists("uniques.txt"):
        remove("uniques.txt")
    if path.exists("uniques.xyz"):
        remove("uniques.xyz")

    unique_name_list=[]

    for i in range(n_files):
        with open('uniques.txt', 'a') as uniques:
            if unique_list[1][i] == 1:
                if args.compare:
                    uniques.write("{} {} {}\n".format(unique_list[0][i], unique_list[1][i], unique_list[2][i]))
                    unique_name_list.append(unique_list[0][i])
                else:
                    uniques.write("{}\n".format(unique_list[0][i]))
                    unique_name_list.append(unique_list[0][i])

    print(unique_name_list)
    #create combined xyz file
    with open("uniques.xyz", 'w') as outfile:
        for fname in unique_name_list:
            with open(fname) as infile:
                outfile.write("{}\n".format(infile.read()))
                
    print(f"{unique_list[1].count(1)} uniques.")
    execution_time = datetime.datetime.now() - begin_time
    print(f"Script took {execution_time} to complete.")

if __name__ == "__main__":
    torsions=read_torsions(args.torsions)
    if args.compare:
        list_compare(args.compare, torsions)
    else:
        main()

