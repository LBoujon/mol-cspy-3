import argparse
import multiprocessing

parser = argparse.ArgumentParser(description='Clusters conformers based on their torsional angles.')
parser.add_argument('-db','--database_file', type=str, metavar='', help='Name of database file for clustering. holds a list of all conformers and their associated energies.')
parser.add_argument('-t','--torsions', type=str, default=False, metavar='', help='List of torsion angles for the molecule equal to the first conformer of the series.')
parser.add_argument('-np', '--processors', type=int, metavar='', default=1, help='Number of processors used.')
parser.add_argument('-ewin', '--energy_window', type=float, metavar='', default=5.0, help='Energy window in which conformers should be compared.')
parser.add_argument('-o', '--overlay', action='store_true', help='Perform an overlay on molecules before clustering, recommended for some molecules.')
args=parser.parse_args()

#Import libaries
import multiprocessing
import sys, math
from webbrowser import get
from operator import itemgetter
from multiprocessing import Pool, Process
import datetime
from os import path,remove
from itertools import islice
import sqlite3
import cspy
from torsions_func import *
from operator import itemgetter

# read in geometry from xyz file
def get_geom(array):
    xyz_array = array
    n_atoms = int(xyz_array[0][0])
    at_types = ['' for i in range(n_atoms)]
    coords = [[0.0 for j in range(3)] for i in range(n_atoms)]
    for i in range(n_atoms):
        at_types[i] = xyz_array[i+2][0]
        for j in range(3):
            coords[i][j] = float(xyz_array[i+2][j+1])
    geom = [at_types, coords]
    return geom

#reads the database as a tuple
def get_conformers(database):
    conn = sqlite3.connect(database)
    cur = conn.cursor()
    cur.execute("select id,xyz_coordinates,energy from distorted_molecules order by energy")
    conformers = cur.fetchall()  
    conn.close()
    return conformers

#should an overlay be done for every comparison?
def overlay(conformers):
    xyzs = {}
    for id, xyz, energy in conformers:
        xyzs[id] = xyz

        # print(xyz)
        m2 = cspy.Molecule.from_xyz_string(xyz)
        m3 = cspy.Molecule.to_xyz_string(m2)
        print(m3)
        

# returns conformer infomation as a dictionary
def get_conformer_details(conformers):
    ids=[]
    energies=[]
    torsions=[]
    inv_torsions=[]
    energies_array = []
    for id, xyz, energy in conformers:
        array=[]
        list_of_lines = xyz.split('\n')
        for line in list_of_lines:
            stripped_line=" ".join(line.split())
            line_split=stripped_line.split(' ')
            array.append(line_split)
        geom = get_geom(array)
        bond_graph = get_bond_graph(geom)
        torsions = get_torsions(geom, bond_graph)
        inv_geom = get_inverted_geom(array)
        inv_bond_graph = get_bond_graph(inv_geom)
        inv_torsions = get_torsions(inv_geom, inv_bond_graph)
        info = [id, energy, torsions, inv_torsions]
        energies_array.append(info)
    energies_array = sorted(energies_array, key=itemgetter(1), reverse=False)
    n_files=len(energies_array)
    file_names = ['' for i in range(n_files)]
    energies = [0.0 for i in range(n_files)]
    torsions_list = [0.0 for i in range(n_files)]
    inv_torsions_list = [0.0 for i in range(n_files)]
    for i in range(n_files):
        file_names[i] = energies_array[i][0] #can change this to if statement
        energies[i] = float(energies_array[i][1])
        torsions_list[i] = energies_array[i][2]
        inv_torsions_list[i] = energies_array[i][3]
    file_energies=[file_names, energies]
    torsions_array = [file_names, torsions_list, inv_torsions_list]
    energies_array=[ids, energies]

    return file_energies, torsions_array, n_files

def append_equiv_to_table():
    conn = sqlite3.connect(database)
    cur = conn.cursor()
    cur.execute("ALTER TABLE distorted_molecules ADD COLUMN equivalent_to")

    conformers = cur.fetchall()
    conn.close()

def get_rank(id, file_energies):
    n=0
    for file in file_energies[0]:
        n+=1
        if file == id:
            return n

def get_conformer_lists(file_energies):
    all_files=file_energies[0]
    all_energies=file_energies[1]
    to_compare_all = []
    for n in range(len(all_files)):
        energy = all_energies[n]
        # print(all_files[n], all_energies[n])
        to_compare = []
        rank=n+1
        all_files_higher_energy=all_files[rank:]
        all_energies_higher_energy=all_energies[rank:]
        limit=energy+energy_threshold
        for i in range(len(all_files_higher_energy)):
            energy=all_energies_higher_energy[i]
            if energy <= limit:
                to_compare.append(all_files_higher_energy[i])
            else:
                break
        to_compare_all.append(to_compare)
    all_compare = [all_files, to_compare_all]

    return all_compare

def get_comparison_list(xyz_file_name, all_comparison_lists):
    n=0
    for file in all_comparison_lists[0]:
        if file == xyz_file_name:
            return all_comparison_lists[1][n]
        n+=1

if __name__ == "__main__":
    
    #start time
    print("Initialising...")
    begin_time = datetime.datetime.now()

    
    #loading in arguments from argparse
    database = args.database_file
    energy_threshold = args.energy_window #kJ/mol
    processors = args.processors
    torsions_file_str = args.torsions
    

    #Begining pre-comparison calculations
    print("Calculating torsion angles")
    conformers = get_conformers(database)
    file_energies, torsions_array, n_files = get_conformer_details(conformers)
    all_comparison_lists=get_conformer_lists(file_energies) # list in the form [id, [list of xyz]]
    # print(len(all_comparison_lists), len(all_comparison_lists[0]), len(all_comparison_lists[1]))
    file_list=file_energies[0]
    uniques=[1 for i in range(len(file_list))]
    unique_list=[file_list, uniques]
    all_conformer_torsions = torsions_array
    # print(len(all_conformer_torsions), len(all_conformer_torsions[0]), len(all_conformer_torsions[1]))
    to_compare = read_torsions(torsions_file_str)

    #starting comparison between conformers.
    print("Starting comparison between conformers")
    n=0
    for xyz_file_name in file_list:
        begin_trial_time = datetime.datetime.now()
        n+=1
        print(f"working on molecule {n} out of {n_files}.\r")
        rank=n+1
        # rank=get_rank(xyz_file_name, file_energies)
        if unique_list[1][rank-1] == 0:
            print("conformer matched, skipping.")
            continue
        else:
            print("conformer has not been matched.")
            ref_all_torsions = find_all_torsions_for_conformer(xyz_file_name, all_conformer_torsions)
            # print(ref_all_torsions)
            inv_ref_all_torsions = find_all_torsions_for_conformer(xyz_file_name, all_conformer_torsions, inverted = True)
            comparison_list=get_comparison_list(xyz_file_name, all_comparison_lists)
            # print(comparison_list)
            if args.overlay:
                matched_list = compare_against_list_mp_Process(ref_all_torsions, inv_ref_all_torsions, comparison_list, to_compare, processors, all_conformer_torsions, overlay=True)
            else:
                matched_list = compare_against_list_mp_Process(ref_all_torsions, inv_ref_all_torsions, comparison_list, to_compare, processors, all_conformer_torsions)
            for xyz in matched_list:
                comparison_rank=get_rank(xyz, file_energies)
                unique_list[1][comparison_rank-1] = 0
            
        trial_execution_time = datetime.datetime.now() - begin_trial_time
        print(f"Calculation_time for {xyz_file_name}: {trial_execution_time}")
        print(f"{len(comparison_list)} comparisons were made.")

    # print(f"FINAL UNIQUES LIST: {unique_list[1]}")

    if path.exists("uniques.txt"):
        remove("uniques.txt")
    if path.exists("uniques.xyz"):
        remove("uniques.xyz")

    unique_name_list=[]

    for i in range(n_files):
        with open('uniques.txt', 'a') as uniques:
            if unique_list[1][i]==1:
                uniques.write("{}\n".format(unique_list[0][i]))
                unique_name_list.append(unique_list[0][i])

    print(unique_name_list)
    #create combined xyz file
    with open("uniques.xyz", 'w') as outfile:
        for fname in unique_name_list:
            with open(fname) as infile:
                outfile.write("{}\n".format(infile.read()))
                
    execution_time = datetime.datetime.now() - begin_time

    print(f"Total execution time: {execution_time}")