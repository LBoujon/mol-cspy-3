from rdkit.Chem.rdFreeSASA import SASAAlgorithm
from rdkit.Chem import rdFreeSASA
from rdkit import Chem
from scipy.sparse import csr_matrix
import networkx
from networkx.algorithms import isomorphism
import cspy
import numpy as np
from rdkit.Geometry import Point3D
from rdkit.Chem import AllChem
import sys

def surface_area(file_list):
    for file in file_list:
        hmol1 = Chem.AddHs(m1)
        ptable = Chem.GetPeriodicTable()
        radii1 = [ptable.GetRvdw(atom.GetAtomicNum()) for atom in hmol1.GetAtoms()]
        opts = rdFreeSASA.SASAOpts()
        opts.algorithm = SASAAlgorithm.ShrakeRupley
        opts.probeRadius = 1.8
        area = rdFreeSASA.CalcSASA(hmol1, radii1, opts=opts)

def overlay_mp(file_list):
    chunks = [file_list[i::core_count] for i in range(core_count)]
    pool = Pool(processes=core_count)
    result = pool.map(overlay, chunks)

def main():
    file_list = sys.argv[1]

    for arg in sys.argv[1:]:
        input = arg
        file_extension = input.split('.')[-1]
        name=input.split('.')[0]

    if file_extension == 'mol':
        m1 = Chem.MolFromMolFile(input)

    else:
        print('ERROR: Please provide .xyz or .mol file.')
        quit()

    from rdkit.Chem.rdFreeSASA import SASAAlgorithm
    from rdkit.Chem import rdFreeSASA


    hmol1 = Chem.AddHs(m1)
    #AllChem.EmbedMolecule(m1)

    ptable = Chem.GetPeriodicTable()
    radii1 = [ptable.GetRvdw(atom.GetAtomicNum()) for atom in hmol1.GetAtoms()]


    opts = rdFreeSASA.SASAOpts()

    opts.algorithm = SASAAlgorithm.ShrakeRupley
    opts.probeRadius = 1.8



    area = rdFreeSASA.CalcSASA(hmol1, radii1, opts=opts)
    #print("Probe radius: {}".format(opts.probeRadius))
    print(area)

if __name__ == "__main__":
    main()