#torsion functions

#Import libaries
import sys, math
from webbrowser import get
from operator import itemgetter
from multiprocessing import Pool, Process
import datetime
from os import path,remove
from itertools import islice
import multiprocessing

## CONSTANTS ##

# threshold beyond average of covalent radiii to determine bond cutoff
bond_thresh = 1.2
single_angle_max = 10.0
max_rms_angle = 5.0

# conversion from radians to degrees and vice versa
rad2deg = 180.0 / math.pi
deg2rad = 1.0 / rad2deg

# covalent (or ionic) radii by atomic element (Angstroms) from
# "Inorganic Chemistry" 3rd ed, Housecroft, Appendix 6, pgs 1013-1014
cov_rads = {  'H' : 0.37, 'C' : 0.77, 'O' : 0.73, 'N' : 0.75, 'F' : 0.71,
  'P' : 1.10, 'S' : 1.03, 'Cl': 0.99, 'Br': 1.14, 'I' : 1.33, 'He': 0.30,
  'Ne': 0.84, 'Ar': 1.00, 'Li': 1.02, 'Be': 0.27, 'B' : 0.88, 'Na': 1.02,
  'Mg': 0.72, 'Al': 1.30, 'Si': 1.18, 'K' : 1.38, 'Ca': 1.00, 'Sc': 0.75,
  'Ti': 0.86, 'V' : 0.79, 'Cr': 0.73, 'Mn': 0.67, 'Fe': 0.61, 'Co': 0.64,
  'Ni': 0.55, 'Cu': 0.46, 'Zn': 0.60, 'Ga': 1.22, 'Ge': 1.22, 'As': 1.22,
  'Se': 1.17, 'Kr': 1.03, 'X' : 0.00, '1' : 0.37, '6' : 0.77, '8' : 0.73,
  '7' : 0.75, '9' : 0.71, '15': 1.10, '16': 1.03, '17': 0.99, '35': 1.14,
  '53': 1.33, '2' : 0.30, '10': 0.84, '18': 1.00, '3' : 1.02, '4' : 0.27,
  '5' : 0.88, '11': 1.02, '12': 0.72, '13': 1.30, '14': 1.18, '19': 1.38,
  '20': 1.00, '21': 0.75, '22': 0.86, '23': 0.79, '24': 0.73, '25': 0.67,
  '26': 0.61, '27': 0.64, '28': 0.55, '29': 0.46, '30': 0.60, '31': 1.22,
  '32': 1.22, '33': 1.22, '34': 1.17, '36': 1.03}

cov_rads = {}

## IO FUNCTIONS ##

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

# read in geometry from xyz file
def get_geom(xyz_file_name):
    xyz_array = get_file_string_array(xyz_file_name)
    n_atoms = int(xyz_array[0][0])
    at_types = ['' for i in range(n_atoms)]
    coords = [[0.0 for j in range(3)] for i in range(n_atoms)]
    for i in range(n_atoms):
        at_types[i] = xyz_array[i+2][0]
        for j in range(3):
            coords[i][j] = float(xyz_array[i+2][j+1])
    geom = [at_types, coords]
    return geom

#finds an xyz file with the same name within a larger xyz.
def get_string_from_large_xyz(xyz_file_name):
    large_xyz_file_name = args.use_long
    large_xyz_array = get_file_string_array(large_xyz_file_name)
    n_lines = len(large_xyz_array)
    lines_per_xyz = n_lines/n_files
    # print(range(n_lines))
    # print(lines_per_xyz)
    # print(f"searching large xyz for file matching {xyz_file_name}")
    for line_number in islice(range(n_lines), 1, None, int(lines_per_xyz)):
        # print(line_number)
        # print(xyz_file_name.split('.xyz')[0])
        # print(large_xyz_array[line_number])
        # print(f"Testing against: {large_xyz_array[line_number][0]}")
        if xyz_file_name.split('.xyz')[0] == large_xyz_array[line_number][0]:
            # print(f"Match made for {xyz_file_name} and {large_xyz_array[line_number][0]}")
            xyz_array = large_xyz_array[line_number-1:line_number+int(lines_per_xyz)-1]
            # print(f"PRINTING RESULTING ARRAY: {xyz_array}.")
            return xyz_array

# input syntax and usage warnings
def get_inputs():
    if (not len(sys.argv) == 2):
        print('Usage: torsions.py XYZ_FILE\n')
        print('  XYZ_FILE: coordinates of target molecule\n')
        sys.exit()
    else:
        xyz_file_name = sys.argv[1]
        return xyz_file_name

# print geometry to screen
def print_geom(geom, comment):
    at_types, coords = geom[0:2]
    n_atoms = len(at_types)
    print('%i\n%s\n' % (n_atoms, comment), end='')
    for i in range(n_atoms):
        print('%-2s' % (at_types[i]), end='')
        for j in range(3):
            print(' %12.6f' % (coords[i][j]), end='')
        print('\n', end='')
    print('\n', end='')

# print bond graph to screen
def print_bond_graph(geom, bond_graph, comment):
    at_types = geom[0]
    n_atoms = len(at_types)
    print('%s\n' % (comment), end='')
    for i in range(n_atoms):
        print(' %4i %-2s -' % (i+1, at_types[i]), end='')
        for j in range(len(bond_graph[i])):
            print(' %i' % (bond_graph[i][j] + 1), end='')
        print('\n', end='')
    print('\n', end='')
    
# print list of bond lengths to screen
def print_bonds(geom, bonds):
    at_types = geom[0]
    n_bonds = len(bonds)
    print('%i bond(s) found (Angstrom)' % (n_bonds))
    for q in range(n_bonds):
        n1, n2  = bonds[q][0:2]
        r12 = bonds[q][2]
        nstr = '%i-%i' % (n1+1, n2+1)
        tstr = '(%s-%s) ' % (at_types[n1], at_types[n2])
        print(' %-15s  %-13s    %6.4f\n' % (nstr, tstr, r12), end='')
    print('\n', end='')
    
# print list of bond angles to screen
def print_angles(geom, angles):
    at_types = geom[0]
    n_angles = len(angles)
    print('%i angle(s) found (degrees)' % (n_angles))
    for q in range(n_angles):
        n1, n2, n3 = angles[q][0:3]
        a123 = angles[q][3]
        nstr = '%i-%i-%i' % (n1+1, n2+1, n3+1)
        tstr = '(%s-%s-%s) ' % (at_types[n1], at_types[n2], at_types[n3])
        print(' %-15s  %-13s   %7.3f\n' % (nstr, tstr, a123), end='')
    print('\n', end='')

# print list of torsion angles to screen
def print_torsions(geom, torsions):
    at_types = geom[0]
    n_torsions = len(torsions)
    print('%i torsion(s) found (degrees)' % (n_torsions))
    for q in range(n_torsions):
        n1, n2, n3, n4 = torsions[q][0:4]
        t1234 = torsions[q][4]
        nstr = '%i-%i-%i-%i' % (n1+1, n2+1, n3+1, n4+1)
        tstr = '(%s-%s-%s-%s) ' % (at_types[n1], at_types[n2], at_types[n3], at_types[n4])
        print(' %-15s  %-13s  %8.3f\n' % (nstr, tstr, t1234), end='')
    print('\n', end='')


## MATH FUNCTIONS ##

# calculate distance between two 3-d cartesian coordinates
def get_r12(coords1, coords2):
    r2 = 0.0
    for p in range(3):
        r2 += (coords2[p] - coords1[p])**2
    r = math.sqrt(r2)
    return r

# calculate unit vector between to 3-d cartesian coordinates
def get_u12(coords1, coords2):
    r12 = get_r12(coords1, coords2)
    u12 = [0.0 for p in range(3)]
    for p in range(3):
        u12[p] = (coords2[p] - coords1[p]) / r12
    return u12

# calculate dot product between two unit vectors
def get_udp(uvec1, uvec2):
    udp = 0.0
    for p in range(3):
        udp += uvec1[p] * uvec2[p]
    udp = max(min(udp, 1.0), -1.0)
    return udp

# calculate unit cross product between two unit vectors
def get_ucp(uvec1, uvec2):
    ucp = [0.0 for p in range(3)]
    cos_12 = get_udp(uvec1, uvec2)
    sin_12 = math.sqrt(1 - cos_12**2)
    ucp[0] = (uvec1[1]*uvec2[2] - uvec1[2]*uvec2[1]) / sin_12
    ucp[1] = (uvec1[2]*uvec2[0] - uvec1[0]*uvec2[2]) / sin_12
    ucp[2] = (uvec1[0]*uvec2[1] - uvec1[1]*uvec2[0]) / sin_12
    return ucp

# calculate angle between three 3-d cartesian coordinates
def get_a123(coords1, coords2, coords3):
    u21 = get_u12(coords2, coords1)
    u23 = get_u12(coords2, coords3)
    dp2123 = get_udp(u21, u23)
    a123 = rad2deg * math.acos(dp2123)
    return a123

# calculate torsion angle between four 3-d cartesian coordinates
def get_t1234(coords1, coords2, coords3, coords4):
    u21 = get_u12(coords2, coords1)
    u23 = get_u12(coords2, coords3)
    u32 = get_u12(coords3, coords2)
    u34 = get_u12(coords3, coords4)
    u21c23 = get_ucp(u21, u23)
    u32c34 = get_ucp(u32, u34)
    dp = get_udp(u21c23, u32c34)
    sign = 2 * float(get_udp(u21c23, u34) < 0) - 1
    t1234 = rad2deg * sign * math.acos(dp)
    return t1234

## TOPOLOGY FUNCTIONS ##

# build graph of which atoms are covalently bonded
def get_bond_graph(geom):
    at_types, coords = geom[0:2]
    n_atoms = len(at_types)
    bond_graph = [[] for i in range(n_atoms)]
    for i in range(n_atoms):
        try:
            covrad1 = cov_rads[at_types[i]]
        except:
            print("Atom type not found.")  
        for j in range(i+1, n_atoms):
            covrad2 = cov_rads[at_types[j]]
            thresh = bond_thresh * (covrad1 + covrad2)
            r12 = get_r12(coords[i], coords[j])
            if (r12 < thresh):
                bond_graph[i].append(j)
                bond_graph[j].append(i)
    return bond_graph

# determine atoms which are covalently bonded from bond graph
def get_bonds(geom, bond_graph):
    at_types, coords = geom[0:2]
    n_atoms = len(at_types)
    bonds = []
    for i in range(n_atoms):      
        for a in range(len(bond_graph[i])):
            j = bond_graph[i][a]
            if (i < j):
                r12 = get_r12(coords[i], coords[j])
                bonds.append([i, j, r12])
    return bonds

# determine atoms which form a bond angle from bond graph
def get_angles(geom, bond_graph):
    at_types, coords = geom[0:2]
    n_atoms = len(at_types)
    angles = []
    for j in range(n_atoms):
        n_jbonds = len(bond_graph[j])
        for a in range(n_jbonds):
            i = bond_graph[j][a]
            for b in range(a+1, n_jbonds):
                k = bond_graph[j][b]
                a123 = get_a123(coords[i], coords[j], coords[k])
                angles.append([i, j, k, a123])
    return angles

# determine atoms which form torsion angles from bond graph
def get_torsions(geom, bond_graph):
    at_types, coords = geom[0:2]
    n_atoms = len(at_types)
    torsions = []
    for j in range(n_atoms):
        n_jbonds = len(bond_graph[j])
        for a in range(n_jbonds):
            k = bond_graph[j][a]
            if (k < j):
                continue
            n_kbonds = len(bond_graph[k])
            for b in range(n_jbonds):
                i = bond_graph[j][b]
                if (i == k):
                    continue
                for c in range(n_kbonds):
                    l = bond_graph[k][c]
                    if (l == j or l == i):
                        continue
                    t1234 = get_t1234(coords[i], coords[j], coords[k], coords[l])
                    torsions.append([i, j, k, l, t1234])
    return torsions

# calulate torsions from torsions file
def get_torsions_f(geom, to_compare):
    at_types, coords = geom[0:2]
    torsions = []
    for torsion in to_compare:
        i = torsion[0] - 1
        j = torsion[1] - 1
        k = torsion[2] - 1
        l = torsion[3] - 1
        t1234 = get_t1234(coords[i], coords[j], coords[k], coords[l])
        torsions.append([torsion[0], torsion[1], torsion[2], torsion[3], t1234])
    return torsions

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

# def get_torsion_angle(all_torsions, torsion):
#     atoms=torsion[0:4]
#     for tor in all_torsions:
#         atoms2=[x+1 for x in tor[:-1]]
#         atoms.sort()
#         atoms2.sort()
#         if atoms == atoms2:
#             return tor[-1]

def get_torsion_angle(all_torsions, torsion):
    for calc_torsion in all_torsions:
        if calc_torsion[0:4] == torsion[0:4]:
            return calc_torsion[4]

#take in a list of files and energies
def get_energies(energies_file_name):
    energies_array = get_file_string_array(energies_file_name)
    energies_array = sorted(energies_array, key=itemgetter(1), reverse=True)
    n_files=len(energies_array)
    file_names=['' for i in range(n_files)]
    energies = [0.0 for i in range(n_files)]
    for i in range(n_files):
        file_names[i] = energies_array[i][0].split(".xyz")[0]+".xyz" #can change this to if statement
        energies[i] = float(energies_array[i][1])
    file_energies=[file_names, energies]
    return file_energies, n_files

#computes the rms angle between two xyz files.
def rms_angle(ref_all_torsions, comp_all_torsions, to_compare):
    dif_squared=[]
    for torsion in to_compare:
        equiv_tor = torsion[-1]
        angle1 = get_torsion_angle(ref_all_torsions, torsion)
        angle2 = get_torsion_angle(comp_all_torsions, torsion)
        dif = get_difference(angle1, angle2, equiv_tor)
        dif_squared.append(dif**2)
    mean_squared=sum(dif_squared)/len(dif_squared)
    root_mean_squared=math.sqrt(mean_squared)
    return root_mean_squared

#calculated the difference between two torsion angles takes into account periodicity of multiplicity of certain torsions angles.
def get_difference(torsion_angle1, torsion_angle2, equiv_torsions):
    dif = min(abs((360/equiv_torsions) - abs(torsion_angle1 - torsion_angle2)), abs(torsion_angle1 - torsion_angle2), abs((360) - abs(torsion_angle1 - torsion_angle2)), abs(360/equiv_torsions - abs(torsion_angle1 + torsion_angle2)), abs(360 - abs(torsion_angle1 + torsion_angle2)))
    return dif

#takes in two xyz files and compares their torsion angles. Returns 1 if unique or 0 if matched.
def compare(ref_all_torsions, inv_ref_all_torsions, xyz_file_name2, to_compare, all_conformer_torsions):
    unique_array=[1 for i in range(len(to_compare)+1)] # if unique = 1, conformers are unique
    test_no=-1
    comp_all_torsions=find_all_torsions_for_conformer(xyz_file_name2, all_conformer_torsions)
    for torsion in to_compare:
        test_no+=1
        xyz1_tor = get_torsion_angle(ref_all_torsions, torsion)
        xyz2_tor = get_torsion_angle(comp_all_torsions, torsion)
        equiv_tor = torsion[-1]
        dif = get_difference(xyz1_tor, xyz2_tor, equiv_tor)
        # print(dif, single_angle_max)
        if abs(dif) < single_angle_max:
            unique_array[test_no]=0
        elif unique_array[test_no]==1:
            xyz1_tor = get_torsion_angle(inv_ref_all_torsions, torsion)
            dif = get_difference(xyz1_tor, xyz2_tor, equiv_tor)
            if abs(dif) < single_angle_max:
                unique_array[test_no]=0

    rms = rms_angle(ref_all_torsions, comp_all_torsions, to_compare)
    if rms < max_rms_angle:
        # print("rms is within maximum rms angle.")
        unique_array[-1] = 0
    else:
        rms = rms_angle(inv_ref_all_torsions, comp_all_torsions, to_compare)
        if rms < max_rms_angle:
            unique_array[-1] = 0

    #test if all test have been passed.
    if unique_array == [0 for i in range(len(to_compare)+1)]:
        #test has passed, conformers are matches.
        return 0
        # print("{} and {}, are not unique.".format(xyz_file_name1, xyz_file_name2))
    else:
        #test failed, conformers are unique.
        return 1
        # print("{} and {}, are unique.".format(xyz_file_name1, xyz_file_name2))

    # print(unique_array)

def compare2(ref_all_torsions, inv_ref_all_torsions, xyz_file_name2, to_compare, all_conformer_torsions):
    comp_all_torsions=find_all_torsions_for_conformer(xyz_file_name2, all_conformer_torsions)
    #non_mirrored
    finished_non_mirrored = False
    while finished_non_mirrored == False:
        for torsion in to_compare:
            xyz1_tor = get_torsion_angle(ref_all_torsions, torsion)
            xyz2_tor = get_torsion_angle(comp_all_torsions, torsion)
            equiv_tor = torsion[-1]
            dif = get_difference(xyz1_tor, xyz2_tor, equiv_tor)
            if abs(dif) > single_angle_max:
                finished_non_mirrored = True
        rms = rms_angle(ref_all_torsions, comp_all_torsions, to_compare)
        if rms < max_rms_angle:
            return 1
    finished_mirrored = False
    while finished_mirrored == False:
        #could not match non-mirrored image, trying mirrored
        for torsion in to_compare:
                xyz1_tor = get_torsion_angle(inv_ref_all_torsions, torsion)
                xyz2_tor = get_torsion_angle(comp_all_torsions, torsion)
                equiv_tor = torsion[-1]
                dif = get_difference(xyz1_tor, xyz2_tor, equiv_tor)
                if abs(dif) > single_angle_max:
                    finished_mirrored = True
        rms = rms_angle(inv_ref_all_torsions, comp_all_torsions, to_compare)
        if rms < max_rms_angle:
            return 1
    return 0

def get_inverted_geom(xyz_file_name):
    geom = get_geom(xyz_file_name)
    coords = geom[1]
    new_coords=[]
    for coordinate in coords:
        x_coordinate=coordinate[0]
        new_x_coordinate=x_coordinate*-1
        new_coordinate=coordinate
        new_coordinate[0]=new_x_coordinate
        new_coords.append(new_coordinate)
    return geom

#determines the rank of the xyz within the list of file energies.
def get_rank(xyz_file_name, file_energies):
    n=0
    for file in file_energies[0]:
        n+=1
        if file == xyz_file_name:
            return n

#compare single xyz against all xyz of higher
#create a l
# ist of xyz files in which the reference should be compared to.
def get_comparison_list(xyz_file_name):
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
def compare_against_list(ref_all_torsions, inv_ref_all_torsions, comparison_list, matched_list, to_compare, all_conformer_torsions, overlay):
    # print(f"comparing {xyz_file_name} to {comparison_list}")
    # if overlay==True:
    #     continue
        #overlay molecules
    for xyz in comparison_list: #this list should be split between processors
        #compare
        if compare(ref_all_torsions, inv_ref_all_torsions, xyz, to_compare, all_conformer_torsions) == 0:
            #add matched item to list
            #return list
            matched_list[xyz] = xyz
                # with open(xyz_file_name.split('.xyz')[0]+"_matches", 'a') as matches:
                #     matches.write("{} ".format(xyz))

def compare_against_list_mp_Process(ref_all_torsions, inv_ref_all_torsions, comparison_list, to_compare, processors, all_conformer_torsions, overlay=False):
    manager = multiprocessing.Manager()
    matched_list = manager.dict()
    processes=[]
    for n in range(processors):
        to_do_list=comparison_list[n::processors]
        p = multiprocessing.Process(target=compare_against_list, args=(ref_all_torsions, inv_ref_all_torsions, to_do_list, matched_list, to_compare, all_conformer_torsions, overlay))
        p.start()
        processes.append(p)
    for process in processes:
        process.join()
    # print(matched_list.values(), "main")
    return matched_list.values()

def compare_against_list_mp(comparison_list):
    chunks = [comparison_list[i::processors] for i in range(processors)]
    pool = Pool(processes=processors)
    result = pool.map(compare_against_list, chunks)

def get_max_comparisons(file_list):
    comparisons=0
    for xyz in file_list:
        comparisons+=len(get_comparison_list(xyz))
    return comparisons

#calculates all torsion angles for each conformer and returns them as a list
def get_conformer_torsions(file_list, to_compare):
    conformer_torsions=[]
    inv_conformer_torsions=[]
    for conformer in file_list:
        geom = get_geom(conformer)
        torsions = get_torsions_f(geom, to_compare)
        conformer_torsions.append(torsions)
        inv_geom = get_inverted_geom(conformer)
        inv_torsions = get_torsions_f(inv_geom, to_compare)
        inv_conformer_torsions.append(inv_torsions)
    all_conformer_torsions=[file_list, conformer_torsions, inv_conformer_torsions]

    return all_conformer_torsions

def find_all_torsions_for_conformer(xyz_file_name, all_conformer_torsions, inverted=False):
    n=0
    for file in all_conformer_torsions[0]:
        if file == xyz_file_name:
            if inverted == True:
                return all_conformer_torsions[2][n]
            else:
                return all_conformer_torsions[1][n]
        n+=1