from rdkit import Chem
from rdkit.Chem import rdMolTransforms
import sys
import subprocess
from cspy import Molecule
from cspy.molbuilder.subprocess.common import coordinates_from_rdkit_molecule
import os

def convert_to_mol(input):
    ident=input.split(".")[0]
    extension=input.split(".")[-1]
    bashCommand = "obabel {} -O {}.mol".format(input, ident)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return "{}.mol".format(ident)

def get_dihedral_atoms_list(molecule: Chem.Mol):

    # Define a rotatable bond
    RotatableBond = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")

    # Find atom pairs that have a rotatable bond between them
    rot_bonds = molecule.GetSubstructMatches(RotatableBond)

    all_dihedrals = []

    # loop through rotatable bond pairs!
    print("Number of rotatable bonds:", len(rot_bonds), rot_bonds)
    for ij in rot_bonds:

        dihedrals = [ij[0], ij[1]]

        # Get the first atom in the rot bond and its neighbours
        atom1 = molecule.GetAtomWithIdx(ij[0])
        nn_atom1 = [x.GetIdx() for x in atom1.GetNeighbors()]
        nn_atom1_types = [x.GetSymbol() for x in atom1.GetNeighbors()]
    
        # Get the second atom in the rot bond and its neighbours
        atom2 = molecule.GetAtomWithIdx(ij[1])
        nn_atom2 = [x.GetIdx() for x in atom2.GetNeighbors()]

        print(atom1.GetIdx())
        print("nn_atom1", nn_atom1)
        print(atom2.GetIdx())
        print("nn_atom2", nn_atom2)

        # find a unique and allowed definition of the dihedral atoms (i*, j, k, l*)
        for i in nn_atom1:
            for l in nn_atom2:
                if i != l and i not in dihedrals and l not in dihedrals:
                    # Dihedral Atoms: (i, j, k, l)
                    dihedrals = [i] + dihedrals + [l]
                    break
            if len(dihedrals) == 4:
                
                break

        if len(dihedrals) != 4:
            raise RuntimeError(
                'ERROR. "get_dihedral_atoms_list": wrong number of atoms to form dihedral for atom pair (%d, %d)'
                % (ij[0], ij[1])
            )

        all_dihedrals = all_dihedrals + [dihedrals]

        final_dihedrals = []

        for d in all_dihedrals:
            final_dihedrals = final_dihedrals + [d]

    return final_dihedrals

def index_from_one(dihedrals):
    new_dihedrals = []
    for dihedral in dihedrals:
        new_dihedral = []
        for atom in dihedral:
            new_atom = atom+1
            new_dihedral.append(new_atom)
        new_dihedrals.append(new_dihedral)
    return new_dihedrals

def find_torsions(xyz_file):
    xyz_input = xyz_file
    mol_input = convert_to_mol(xyz_input)
    mol = Chem.MolFromMolFile(mol_input, removeHs=False)
    # mol = Chem.MolFromMolFile(mol_input, removeHs=True)
    cspy_mol_original = Molecule.load(xyz_input)
    dihedrals = get_dihedral_atoms_list(mol)
    print(f"old dihedrals:", dihedrals)
    # dihedrals = index_from_one(dihedrals)
    find_symetry_of_torsions(cspy_mol_original, dihedrals)
    # find rotational symetry
    # for dihedral in dihedrals:
    #     rotational_symetry_number = find_symetry_of_torsion(cspy_mol_original, dihedral)
    #     print(rotational_symetry_number)
    return dihedrals

def coordinates_to_xyz_file(name, coordinates, xyz_file_name):
    with open(xyz_file_name, 'w') as xyz:
        xyz.write('{}\n'.format(len(coordinates)))
        xyz.write('{}\n'.format(name))
        for coordinate in coordinates:
            xyz.write('{} {:.6f} {:.6f} {:.6f}\n'.format(*coordinate))

# inputs cspy_mol outputs distorted cspy_mol
def distort_molecule(cspy_mol, torsion, degrees, constrained_torsion=None):
    cspy_mol.save('cspy_mol.xyz')
    mol_input = convert_to_mol('cspy_mol.xyz')
    rdkit_mol = Chem.MolFromMolFile(mol_input, removeHs=False)
    a=torsion[0]
    b=torsion[1]
    c=torsion[2]
    d=torsion[3]
    conformer = rdkit_mol.GetConformer()
    dihedral_angle = rdMolTransforms.GetDihedralDeg(conformer, a, b, c, d)
    new_dihedral_angle = dihedral_angle + degrees
    rdMolTransforms.SetDihedralDeg(conformer, a, b, c, d, new_dihedral_angle)
    coord = coordinates_from_rdkit_molecule(rdkit_mol, conformer)
    coordinates_to_xyz_file("Distorted conformation", coord, "distorted_conformation.xyz")
    distorted_molecule = Molecule.load("distorted_conformation.xyz")
    # os.remove("distorted_conformation.xyz")
    # os.remove("cspy_mol.xyz")
    return distorted_molecule


def find_symetry_of_torsion(cspy_mol, torsion):
    print("Attempting to find rotational symetry of torsion:", torsion)
    i = 3
    while 0 < i < 4:
        i -= 1
        degrees = 360.0/i
        print(f"Distorting {torsion} by {degrees}.")
        distorted = distort_molecule(cspy_mol, torsion, degrees)
        try:
            reordered_mol, order, rmsd = cspy_mol.overlay(distorted)
            print("Overlay yielded rmsd:", rmsd)
            if rmsd < 0.01:
                return i
        except:
            print("Overlay produces a new molecule, ignoring.")

#returns the rmsd for a distorsion around a given torsion by degrees
def rmsd_from_overlay_and_distorsion(cspy_mol, torsion, degrees):
    print(f"Distorting {torsion} by {degrees} degrees. {i}")
    distorted = distort_molecule(cspy_mol, torsion, degrees)
    distorted.save("distorted_conformation_"+str(torsion)+"_"+str(degrees)+".xyz")
    try:
        rmsd = cspy_mol.rmsd(distorted)
        if rmsd < max_rmsd:
            if 0.0 < rmsd < best_rmsd:
                print(f"The new best rmsd is {rmsd}, rotational symetry {i}.")
                # except:
                # print("Overlay produces a new molecule, ignoring.")
            i -= 1
    except:
        print("Overlay produces a new molecule, ignoring.")
# outputs the list of symetries for each torsion.
def find_symetry_of_torsions2(cspy_mol, torsions):
    print("Attempting to find rotational symetry of all torsions:", torsions)
    results=[]
    for index in range(len(torsions)):
        torsions_copy = torsions.copy()
        torsion = torsions_copy.pop(index)
        other_torsions = torsions_copy
        print("Attempting to find rotational symetry of torsion:", torsion)
        n = 5 # maximum rotational symmetry value
        i = 4
        max_rmsd = 0.3
        best_rmsd = -1.0
        rmsds = []

        for i in range(5): # from values 2 to 6
            i += 2
            degrees = 360.0/i
            

            try:
                rmsd = cspy_mol.rmsd(distorted)
                print("Overlay yielded rmsd:", rmsd)
                rmsds.append(rmsd)
                
                if rmsd < max_rmsd:
                    if 0.0 < rmsd < best_rmsd:
                        print(f"The new best rmsd is {rmsd}, rotational symetry {i}.")


                else:
                    for other_torsion in other_torsions:
                        print(f"Distoring {torsion} with {other_torsion} by {degrees} degrees.")
                        more_distorted = distort_molecule(distorted, other_torsion, degrees)
                        more_distorted.save("distorted_conformation_"+str(torsion)+"_"+str(other_torsion)+"_"+str(degrees)+".xyz")
                        try:
                            rmsd = cspy_mol.rmsd(more_distorted)
                            print("Overlay yielded rmsd:", rmsd)
                            if rmsd < max_rmsd:
                                print(f"Rotational symetry of {torsion} is {i}.")
                                torsion.append(i)
                                break
                                # return i
                        except:
                            print("Overlay produces a new molecule, ignoring.")
            except:
                print("Overlay produces a new molecule, ignoring.")
            i -= 1

        if best_rmsd < max_rmsd:
            
            results.append(torsion)
    print(results)

def find_symetry_of_torsions(cspy_mol, torsions):
    print("Attempting to find rotational symetry of all torsions:", torsions)
    results=[]
    for index in range(len(torsions)):
        torsions_copy = torsions.copy()
        torsion = torsions_copy.pop(index)
        other_torsions = torsions_copy
        print("Attempting to find rotational symetry of torsion:", torsion)
        n = 5 # maximum rotational symmetry value
        i = 4
        max_rmsd = 0.3
        while 0 < i <= n:
            if i == 1:
                print(f"Rotational symetry of {torsion} is {i}.")
                torsion.append(i)
                break
            degrees = 360.0/i
            print(f"Distorting {torsion} by {degrees} degrees.")
            distorted = distort_molecule(cspy_mol, torsion, degrees)
            distorted.save("distorted_conformation_"+str(torsion)+"_"+str(degrees)+".xyz")
            try:
                rmsd = cspy_mol.rmsd(distorted)
                print("Overlay yielded rmsd:", rmsd)
                if rmsd < max_rmsd:
                    print(f"Rotational symetry of {torsion} is {i}.")
                    torsion.append(i)
                    break
                    # return i
                else:
                    for other_torsion in other_torsions:
                        print(f"Distoring {torsion} with {other_torsion} by {degrees} degrees.")
                        more_distorted = distort_molecule(distorted, other_torsion, degrees)
                        more_distorted.save("distorted_conformation_"+str(torsion)+"_"+str(other_torsion)+"_"+str(degrees)+".xyz")
                        try:
                            rmsd = cspy_mol.rmsd(more_distorted)
                            print("Overlay yielded rmsd:", rmsd)
                            if rmsd < max_rmsd:
                                print(f"Rotational symetry of {torsion} is {i}.")
                                torsion.append(i)
                                break
                                # return i
                        except:
                            print("Overlay produces a new molecule, ignoring.")
            except:
                print("Overlay produces a new molecule, ignoring.")
            i -= 1
        results.append(torsion)
    print(results)

def write_torsions(torsions, outfile):
    with open(outfile, "w") as f:
        for torsion in torsions:
            f.write("{} {} {} {} {}\n".format(torsion[0]+1, torsion[1]+1, torsion[2]+1, torsion[3]+1, torsion[4]))

def coordinates_from_rdkit_molecule(molecule, conformer):
    coordinates = []
    for atom in molecule.GetAtoms():
        position = conformer.GetAtomPosition(atom.GetIdx())
        coordinates.append(
            [atom.GetAtomicNum(), position.x, position.y, position.z]
        )
    return coordinates

def main():
    xyz_file = str(sys.argv[1])
    torsions = find_torsions(xyz_file)
    write_torsions(torsions, "generated_torsions")
    print(torsions)

if __name__ == "__main__":
    main()
