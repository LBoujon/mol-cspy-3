from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import TorsionFingerprints
import itertools
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sys
import os
import argparse


#argparse setup
parser = argparse.ArgumentParser(description='Generate starting positions for multiple CREST searches.')
parser.add_argument('-xyz','--xyz_file', type=str, metavar='', help='Name of the molecule for clustering without extension')
parser.add_argument('-np', '--processors', type=int, metavar='', default=1, help='Number of processors used.')
parser.add_argument('-nc', '--conformers', type=int, metavar='', default=1000, help='Number of conformers generated to sample conformational space.')
# parser.add
# parser.add_argument('-r', '--reset', action='store_true', help='Deletes directories and files for a hard reset.')

# group = parser.add_mutually_exclusive_group()
args=parser.parse_args()



#converts between file types.
def obabel(conformer_file, new_extension):
    ident=conformer_file.split('.')[:-1]
    bashCommand = "obabel {} -O {}.{}".format(conformer_file, ident, new_extension)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

def coordinates_from_rdkit_molecule(molecule, conformer):
    coordinates = []
    for atom in molecule.GetAtoms():
        position = conformer.GetAtomPosition(atom.GetIdx())
        coordinates.append(
            [atom.GetAtomicNum(), position.x, position.y, position.z]
        )
    return coordinates

def append_coordinates_to_xyz_file(name, coordinates, xyz_file_name):
    with open(xyz_file_name, 'a') as xyz:
        xyz.write('{}\n'.format(len(coordinates)))
        xyz.write('{}\n'.format(name))
        for coordinate in coordinates:
            xyz.write('{} {:.6f} {:.6f} {:.6f}\n'.format(*coordinate))

def get_torsions(molecule):
    torsions = TorsionFingerprints.CalculateTorsionLists(molecule)
    return torsions

def align_conformers(molecule):
    rmslist = []
    AllChem.AlignMolConformers(molecule, RMSlist=rmslist)
    return rmslist

#global variable
def generate_conformers(input):
    molecule = Chem.MolFromMolFile(input)
    molecule = Chem.AddHs(molecule)
    cids = AllChem.EmbedMultipleConfs(molecule, numConfs=args.conformers, numThreads=0)
    res = AllChem.MMFFOptimizeMoleculeConfs(molecule, numThreads=0)
    engs = AllChem.UFFOptimizeMolecule(molecule)
    return cids

#optimise generated conformer using xtb
def optimise(conformer):
    bashCommand = "xtb {} -opt".format(conformer)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

def calculate_torsions(cids):
    conf_torsions = pd.DataFrame({'Conformer' : []})
    index = 0
    for conf in cids:
        conf_num = int(conf) + 1
        conf_name = 'conformer_' + str(conf_num)
        torsion_lists = TorsionFingerprints.CalculateTorsionLists(molecule)
        torsions = TorsionFingerprints.CalculateTorsionAngles(molecule, torsion_lists[0], torsion_lists[1], confId=conf)
        list = []
        for tort in range(len(torsions)):
            list.append(torsions[tort][0][0])
        series_list = pd.Series(list)
        conf_torsions = conf_torsions.append(series_list, ignore_index=True)
        conf_torsions['Conformer'].iloc[[conf]] = conf_name

def main(input):
    extension=input.split('.')[-1]
    if extension != 'mol':
        obabel(input_xyz, mol)
    generate_conformers(input)
    calculate_torsions(input)
    
torsion_atoms = []

print(torsion_lists[0])
print(torsion_lists[1])

for torsion_list in torsion_lists:
    for torsion in torsion_list:
        correct_label_atom_list = []
        for atom in torsion[0][0]:
            correct_label_atom_list.append(int(atom)+1)
        torsion_atoms.append(str(correct_label_atom_list))

#print("Torsion list:\n{}".format(torsion_atoms))

conf_torsions = conf_torsions.set_index('Conformer')
#conf_torsions.columns=torsion_atoms
print(conf_torsions)
conf_torsions.to_csv('torsions.csv')



###########################
#export input for geoPCA
##########################

conf_torsions.to_csv('GeoPCA_input.csv', header=False, index=False)


#write out all conformers to conformers.xyz
n = 0

if os.path.isfile('conformers.xyz'):
    os.remove('conformers.xyz')

for conformer in molecule.GetConformers():
    n+=1
    conf_name = 'conformer' + '_' + str(n)
    coord = coordinates_from_rdkit_molecule(molecule, conformer)
    append_coordinates_to_xyz_file(conf_name, coord, 'conformers.xyz')

#################
#perform principle component analysis
################

pca = PCA(n_components=0.60)
pca.fit(conf_torsions)
x_pca = pca.transform(conf_torsions)

#################
#perform UMAP
#################

#umap_model = umap.UMAP(n_neighbors=14, min_dist=0.4, random_state=42, n_components = 3)
#x_umap = umap_model.fit_transform(conf_torsions)

##############
#explained variance
############

#convert to percentage
y = []
for i in pca.explained_variance_ratio_:
    y.append(100*i)

#print(len(pca.explained_variance_ratio_))
number_of_components=len(pca.explained_variance_ratio_) + 1
x = range(1, number_of_components)
plt.bar(x, y)
plt.xlabel('Principle Component')
plt.ylabel('Explained Variance / %')

#save figure
plt.savefig('explained_variance.png')
plt.close()

################
#loadings and scree plot
################

loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=torsion_atoms)
print(loadings)

###################
#k-means
##################

Sum_of_squared_distances = []
K = range(2,30)

for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(x_pca)
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum of squared distances')
#plt.title('Elbow Method For Optimal k')
plt.savefig('elbow_plot.png')
plt.close()

from sklearn.metrics import silhouette_score

sil = []
kmax = 10

# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2

for k in K:
    kmeans_sil = KMeans(n_clusters = k).fit(x_pca)
    labels = kmeans_sil.labels_
    sil.append(silhouette_score(x_pca, labels, metric = 'euclidean'))

k_opt=0
n=1
prev=0
print(sil)
for i in sil:
    n+=1 #value of k for i
    print('working on k = {}'.format(n))
    if i > prev:
        prev = i
    else:
        k_opt = n-1 
        break


print('The optimal value of k is {}'.format(k_opt))

#plot silhouette graph

plt.plot(K, sil, 'bx-')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
#plt.title('Silhouette Plot For Optimal k')
plt.savefig('Silhouette_plot.png')
plt.close()



kmeans = KMeans(n_clusters=k_opt)
kmeans.fit(x_pca)
x_pca_kmeans = kmeans.predict(x_pca)

kmeans_df = conf_torsions
kmeans_df['cluster'] = x_pca_kmeans
kmeans_df.to_csv('groups.csv')

#######
#plot pca graph
######

print(x_pca)

plt.scatter(x_pca[:,0], x_pca[:,1], c=kmeans_df['cluster'])
plt.xlabel('PC1')
plt.ylabel('PC2')



plt.savefig('pca_distribution.png')
plt.close()
