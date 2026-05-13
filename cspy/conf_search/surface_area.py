from rdkit import Chem
from scipy.sparse import csr_matrix
import networkx
from networkx.algorithms import isomorphism
import cspy
import numpy as np
from rdkit.Geometry import Point3D
from rdkit.Chem import AllChem
import sys



input = sys.argv[1]
file_extension = input.split('.')[-1]
name=input.split('.')[0]

if file_extension == 'xyz':
    mol_out=name+'.mol'
    
    
    mol_a = Chem.MolFromSmiles('C1=C2C(=C(C=C1)F)C(=O)N(C(=N2)C(N(C3=C4C(=NC=N3)N=C[N]4[H])[H])CC)C5=CC=CC=C5')
    mol_b = Chem.MolFromSmiles('C1=C2C(=C(C=C1)F)C(=O)N(C(=N2)C(N(C3=C4C(=NC=N3)[N](C=N4)[H])[H])CC)C5=CC=CC=C5')
    mol_c = Chem.MolFromSmiles('')
    mol_d = Chem.MolFromSmiles('')


    for mol_i in [mol_a, mol_b]:
        mol = Chem.AddHs(mol_i)
        n_atoms = len(mol.GetAtoms())
    
        mat = np.zeros((n_atoms, n_atoms))
        for atom in mol.GetAtoms():
            for neigh in atom.GetNeighbors():
                mat[atom.GetIdx(), neigh.GetIdx()] = 1
        graph_1 = networkx.from_scipy_sparse_matrix(csr_matrix(mat))
    
        mol2 = cspy.Molecule.load(input)
        mol2.guess_bonds()
        graph_2 = networkx.from_scipy_sparse_matrix(csr_matrix(mol2.bonds))
    
        GM = isomorphism.GraphMatcher(graph_1, graph_2)
        if not GM.is_isomorphic():
            continue
    
        mapping = GM.mapping
        sorted_map = {i: mapping[i] for i in range(n_atoms)}
        order = list(sorted_map.values())
    
        AllChem.EmbedMolecule(mol)
        pos = mol2.positions[order]
        conf = mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            x, y, z = pos[i]
            conf.SetAtomPosition(i, Point3D(x, y, z))
    
        Chem.MolToMolFile(mol, mol_out)
        #m1 = mol
        m1 = Chem.MolFromMolFile(mol_out)
        break

elif file_extension == 'mol':
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

