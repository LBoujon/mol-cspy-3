import argparse

parser = argparse.ArgumentParser(description='Clusters conformers based on their torsional angles.')
parser.add_argument('-e','--energies_file', type=str, metavar='', help='Name of energies file, holds a list of all conformers and their associated energies.')
parser.add_argument('-m', '--use_master', type=str, metavar='', help='Overlay conformers to a master file, if not stated, will use first item of list.')
parser.add_argument('-np', '--processors', type=int, metavar='', default=1, help='Number of processors used.')

import cspy
import sys
from multiprocessing import Process
import os
from cspy.conf_search.torsions_func import get_energies

def overlay(m1, file_list):
    for file in file_list:
        m2 = cspy.Molecule.load(file)
        print("{} loaded, performing overlay".format(file))
        m3, order, rmsd = m1.overlay(m2)
        print("overlay complete, saving to overlayed")
        m3.save("overlayed/"+str(file))
        print(order)

def overlay_mp(m1, file_list, processors):
    processes=[]
    for n in range(processors):
        to_do_list=file_list[n::processors]
        p = Process(target=overlay, args=(m1, to_do_list,))
        p.start()
        processes.append(p)
    for process in processes:
        process.join()

    # chunks = [file_list[i::core_count] for i in range(core_count)]
    # pool = Pool(processes=core_count)
    # result = pool.map(overlay, chunks)

def main():
    args=parser.parse_args()

    core_count = args.processors
    energies_file_str=args.energies_file
    master = args.use_master

    file_energies, n_files = get_energies(energies_file_str)


    #overlay all atoms
    if not os.path.isdir('overlayed'):
        os.mkdir('overlayed')

    if args.use_master:
        master = args.use_master
    else:
        master = file_energies[0][0]

    file_list = file_energies[0]

    m1 = cspy.Molecule.load(master)
    
    overlay_mp(m1, file_list, core_count)
    for conformer in file_list:
        os.rename("overlayed/"+str(conformer), str(conformer))
    os.rmdir("overlayed")

if __name__ == "__main__":
    main()
