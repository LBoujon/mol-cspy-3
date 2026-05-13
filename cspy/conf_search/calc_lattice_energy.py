#count the number of folders
import os
from cspy import Molecule
from cspy import Crystal
import subprocess
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil 
from math import gcd
from functools import reduce

# setup arguments for argparse.
parser = argparse.ArgumentParser(description='Calculates lattice energies for co-crystal and solvate systems.')
parser.add_argument('structures', type=str, metavar='', help='Name of file containing the name of structures.')
parser.add_argument('-m', '--molecules', type=str, metavar='', default=1, help='molecules used')
parser.add_argument('-j', '--processors', type=int, metavar='', default=1, help='Number of processors used.')
parser.add_argument('-s', '--stoichiometry', type=str, metavar='', default="1", help='Stoichiometry of molecules for lattice energy calculations')
parser.add_argument('-r', '--report', action='store_true', help='Plots graphs with larger text')

group = parser.add_mutually_exclusive_group()
args=parser.parse_args()

def read_structures(filename):
    with open(filename, 'r') as f:
        return [line.split('\n')[0] for line in f]

def platon(structure):
    bashCommand = f"platon -o {structure} -N"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    output_file_name = structure.split('.cif')[0]+'_pl.res'
    return output_file_name

def isNaN(number):
    return number != number

def pcm_energy_from_log_file(log):
    with open(log, 'r') as fp:
        for line in reversed(fp.readlines()):
            if '<psi(f)|   H    |psi(f)>' in line:
                return float(line.split()[5])
        return float('NaN')

def lattice_energy_from_output(output_file):
    with open(output_file, 'r') as fp:
        for line in reversed(fp.readlines()):
            if 'Final energy:' in line:
                return float(line.split()[4][:-1])
        return float('NaN')

def lattice_density_from_output(output_file):
    with open(output_file, 'r') as fp:
        for line in reversed(fp.readlines()):
            if 'final density:' in line:
                return float(line.split()[7])
        return float('NaN')

def unique(list1):
    # insert the list to the set
    list_set = set(list1)
    # convert the set to the list
    unique_list = (list(list_set))
    return unique_list

def average(lst):
    return sum(lst) / len(lst)

def find_equivalents(output_file, letter):
    with open(output_file, 'r') as fp:
        for line in reversed(fp.readlines()):
            if str(letter)+" =" in line:
                return float(line.split()[7])
        return float('NaN')

def run_dma(xyz):
    bashCommand = f"cspy-opt {xyz} -p fit --method PBE1PBE --basis-set 6-311G** --cutoff 30.0 -j 8 --single-point --no-cleanup --pcm 3.0"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

def get_matches_df(list):
    matched_df = pd.DataFrame({'structure' : [], 'lattice_energy' : [], 'density' : []})
    for structure in list:
        structure_data = df.loc[df['structure'] == structure]
        matched_df = matched_df.append(structure_data, ignore_index=True)
    return matched_df

def stoichiometry(structure_file_name):
    
    crystal = Crystal.load(structure_file_name)
    uniques =  crystal.unique_components()

    mols = []
    numbers = []
    stoichiometries = {}

    for n in range(len(uniques)):
        mol = uniques[0][n]
        formula = mol.molecular_formula
        mols.append(formula)
        n_formula = uniques[1][n]
        numbers.append(len(n_formula))

    solved_numbers = solve(numbers)

    for n in range(len(solved_numbers)):
        formula = mols[n]
        n_formula = int(solved_numbers[n])
        stoichiometries[formula] = n_formula
    return stoichiometries

def solve(numbers):
    denominater = reduce(gcd,numbers)
    solved = [i/denominater for i in numbers]
    return solved

structures = read_structures(args.structures)
# structures = ["idelalisib_A_9.dmac-QR-19-10858-3.cif"]
dir = os.getcwd()

#dictionary of various global minimum energies for various molecules.
global_min_energy = {'C22H18N7OF': -3692387.6252760002, 'C4H9NO': -755024.4943463901, 'C5H5N': -651262.8268, 'C9H10N2O3': -1794576.6784} #idelalisib, dimethylacetamide, pyridine, MNIAAN_mol

if not os.path.isdir('optimised'):
    os.mkdir("optimised")

def get_energies(structures):
    completed = []
    failed = []
    nodir = []
    with open("table.csv", 'w') as t:
        t.write("structure,lattice_energy,density,num_of_mols\n")
    if os.path.isfile("failed.txt"):
        os.remove("failed.txt")
    if os.path.isfile("nodir.txt"):
        os.remove("nodir.txt")    
    if os.path.isfile("optimised/energies.txt"):
            os.remove("optimised/energies.txt")
    for structure in tqdm(structures):
        os.chdir(dir)
        # print(structure)
        folder=structure.split(".cif")[0]
        opt_str = folder + "_p1.opt.cif"
        try:
            os.chdir(folder)
        except:
            print(f"{folder} not found.")
            with open("nodir.txt", 'a') as f:
                f.write(f"{structure}\n")
            print(f"Current directory is {os.getcwd()}")
            os.chdir(dir)
            nodir.append(structure)
            continue
        
        # platon_str = platon(opt_str)
        # crystal = Crystal.load(platon_str)
        # uniques =  crystal.symmetry_unique_molecules()
        # Z_prime = len(uniques)
        
        try:
            # print(opt_str)
            platon_str = platon(opt_str)
            crystal = Crystal.load(platon_str)
            uniques =  crystal.symmetry_unique_molecules()
            Z_prime = len(uniques)
        except:
            print("Could not calculate Z_prime.")
            os.chdir(dir)
            continue

        if os.path.isfile("errors.txt"):
            intermolecular_energy = lattice_energy_from_output("errors.txt")
        else:
            print(f"Could not find minimization output for {structure}. Skipping.")
            with open("../failed.txt", 'a') as f:
                f.write(f"{structure}\n")
            os.chdir(dir)
            continue

        if isNaN(intermolecular_energy):
            print(f"Minimisation failed for {structure}. Skipping.")
            failed.append(structure)
            with open("../failed.txt", 'a') as f:
                f.write(f"{structure}\n")
            os.chdir(dir)
            continue
        try:
            stoichs = {}
            min_Z_prime = 0
            mols = args.molecules.split(' ')
            for mol, stoich in zip(mols[::2], mols[1::2]):
                min_Z_prime += int(stoich)
                stoichs[mol] = int(stoich)
        except:
            print("Something went wrong with stoichiometry calculation. Skipping.")
            continue
        
        density = lattice_density_from_output("errors.txt")
        xyzs = [f for f in os.listdir() if f.endswith('.xyz') and os.path.isfile(os.path.join(f))]
        species_list = {}

        logs = [f for f in os.listdir() if f.endswith('.log') and os.path.isfile(os.path.join(f))]

        for log in logs:
            xyz = log.split(".log")[0] + ".xyz"
            dma = xyz.split(".xyz")[0] + ".dma"
            # print(xyz)
            try:
                mol = Molecule.load(xyz)
            except:
                print(f"Could not find xyz file. Skipping {log}.")
                continue
            formula = mol.molecular_formula
            
            
            conformer_energy = pcm_energy_from_log_file(log)
            if isNaN(conformer_energy):
                print(f"No energy found for {xyz}, performing single-point calculation.")
                run_dma(xyz)
                os.rename(formula+"_A.log", log)
                conformer_energy = pcm_energy_from_log_file(log)

                remove_list = [formula+".mols", formula+"_A.xyz", formula+"_A.fchk", formula+"_A.dma", formula+".dma"]
                for item in remove_list:
                    if os.path.isfile(item):
                        os.remove(item)

            conformer_energy_kjmol = conformer_energy * 4.359744E-21 * 6.02214076E23
            
            if formula not in species_list:
                species_list[formula] = []
            species_list[formula].append(conformer_energy_kjmol)
            total_intramolecular_contribution = 0

        for formula in species_list:
            number_of_molecules = species_list[formula]
            average_energy = average(species_list[formula])
            delta_E_conf = average_energy - global_min_energy[formula]
            try:
                formula_contribution = delta_E_conf * stoichs[formula]
                total_intramolecular_contribution += formula_contribution
            except:
                print(f"{formula} not found in optimised crystal structure. Skipping.")
                break
        
        normalised_intermolecular_energy = intermolecular_energy * min_Z_prime / Z_prime
        # lattice_energy = normalised_intermolecular_energy + total_intramolecular_contribution
        lattice_energy = intermolecular_energy + total_intramolecular_contribution

        # if min_Z_prime != Z_prime:
        print(f"total_intramolecular_contribution: {total_intramolecular_contribution}, intermolecular energy: {intermolecular_energy} Z_prime: {Z_prime}, lattice energy: {lattice_energy}")

        os.chdir(dir)
        with open("table.csv", 'w') as t:
            t.write("structure,lattice_energy,density,num_of_mols,total_intramolecular_contribution,normalised_intermolecular_energy,intermolecular_energy, \n")
        with open("debug_table.csv", 'a') as dt:
            dt.write(f"{opt_str},{lattice_energy},{density},{Z_prime},{total_intramolecular_contribution}, {normalised_intermolecular_energy}, {intermolecular_energy}\n")

        with open("table.csv", 'a') as t:
            t.write(f"{opt_str},{lattice_energy},{density},{Z_prime}\n")
            
        with open("optimised/energies.txt", 'a') as e:
            e.write(f"{opt_str} {lattice_energy} {density}\n")
        try:
            shutil.copyfile(f"{folder}/{opt_str}", f"optimised/{opt_str}")
        except:
            print(f"file missing: {folder}/{opt_str}. Skipping.")
            continue

    
    print(nodir)
        
    print(f"{len(failed)}/{len(structures)} failed.")

if not os.path.isfile("table.csv"):
    get_energies(structures)

plt.rcParams['font.family']='serif'

df = pd.read_csv("table.csv")

df = df.sort_values(by='lattice_energy')
df.to_csv('optimised/energies.txt', index=None, header=None, sep=' ')
df.to_csv('table.csv', index=None)

print(df)

if os.path.isfile("optimised/unique_structures.txt"):
    uniques=read_structures("optimised/unique_structures.txt")
    df = get_matches_df(uniques)

if os.path.isfile("optimised/experimental_matches.txt"):
    matches=read_structures("optimised/experimental_matches.txt")
else:
    matches=[]

matched_df=get_matches_df(matches)

# plt.ylim(-280,-200)
cleaned_df = df[df['lattice_energy'] < 0]
cleaned_df = cleaned_df[cleaned_df['lattice_energy'] > -500]
high_df = df[df['lattice_energy'] < -500]
low_df = df[df['lattice_energy'] > 0]
print(high_df)
print(low_df)

cleaned_df['relative_lattice_energy'] = cleaned_df['lattice_energy'] - cleaned_df['lattice_energy'].min()

print(cleaned_df)

energy_threshold=15

low_energy_df = cleaned_df[cleaned_df['relative_lattice_energy'] < energy_threshold]

print(low_energy_df)

if os.path.isfile("low_energy_structures.txt"):    
    os.remove("low_energy_structures.txt")

print(low_energy_df['structure'].values)

with open('low_energy_structures.txt', 'a') as les:
    for struct in low_energy_df['structure'].values:
        les.write(struct+"\n")

if args.report:
    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots()
    plt.scatter(cleaned_df.density, cleaned_df.lattice_energy, s=10)
    plt.scatter(matched_df.density, matched_df.lattice_energy, marker="D", s=20,)
    plt.ylabel('Lattice Energy / ${kJmol^{-1}}$',fontsize=18)
    plt.xlabel('Density / ${gcm^{-3}}$',fontsize=18)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    plt.tight_layout()
    plt.savefig('crystal_landscape_report.png', dpi=600)
else:
    # plt.scatter(cleaned_df.density, cleaned_df.lattice_energy, s=10)
    # plt.scatter(matched_df.density, matched_df.lattice_energy, marker="D", s=20, color='orange')
    plt.scatter(cleaned_df.density, cleaned_df.lattice_energy, s=3, c=cleaned_df.num_of_mols)
    plt.colorbar()
    plt.ylabel('Lattice Energy / ${kJmol^{-1}}$')
    plt.xlabel('Density / ${gcm^{-3}}$')
    plt.savefig('crystal_landscape.png', dpi=600)

exit()

