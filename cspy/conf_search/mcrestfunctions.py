#list of functions used for mCREST
from cspy.conf_search.clustangles import *
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem
# from rdkit.Chem import rdMolTransforms
from rdkit.Chem import TorsionFingerprints
import os
import logging

LOG = logging.getLogger(__name__)

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

def convert_to_mol(input):
    ident=input.split(".")[0]
    extension=input.split(".")[-1]
    # if extension != "mol":
    bashCommand = "obabel {} -O {}.mol".format(input, ident)
    process = subprocess.run(bashCommand.split(), capture_output=True)
    # output, error = process.communicate()
    return "{}.mol".format(ident)

def generate_RDKit_conformers(input, number_of_conformers):
    mol = convert_to_mol(input)
    molecule = Chem.MolFromMolFile(mol)
    molecule = Chem.AddHs(molecule)
    cids = AllChem.EmbedMultipleConfs(molecule, numConfs=number_of_conformers, numThreads=0, randomSeed=42)
    res = AllChem.MMFFOptimizeMoleculeConfs(molecule, numThreads=0)
    energies=[]
    for re in res:
        energies.append(re[1]) 

    torsion_data = {}
    index = 0
    for conf in cids:
        conf_num = int(conf) + 1
        conf_name = f'conformer_{conf_num}'
        torsion_lists = TorsionFingerprints.CalculateTorsionLists(molecule)
        torsions = TorsionFingerprints.CalculateTorsionAngles(molecule, torsion_lists[0], torsion_lists[1], confId=conf)

        row_data = [conf_name]
        for tort in range(len(torsions)):
            row_data.append(torsions[tort][0][0])
        torsion_data[conf_name] = [t[0][0] for t in torsions]

    conf_torsions = pd.DataFrame.from_dict(torsion_data,orient="index")
    conf_torsions.to_csv('torsions.csv', header=None, index=None)

    n = 0
    if os.path.isfile('conformers.xyz'):
        os.remove('conformers.xyz')
    for conformer in molecule.GetConformers():
        n+=1
        conf_name = 'conformer' + '_' + str(n)
        coord = coordinates_from_rdkit_molecule(molecule, conformer)
        append_coordinates_to_xyz_file(conf_name, coord, 'conformers.xyz')
    return conf_torsions, energies

#write out all conformers to conformers.xyz

def perform_PCA(conf_torsions, method='dpca'):
    a = Angles()
    minimum_torsions=2
    
    a.read_csv("torsions.csv", units="degrees") #change this to read from array
    if method == 'dpca':
        try:
           dp = dpca(a) #run pca directly from array
        # dp.advice(70)
           c = dp.project(5, a)
        # PCA_df = pd.DataFrame(c)
           return c
        except:
            LOG.error("Minimum torsion angles for PCA not reached.")
            quit()
    if method == 'geopca':
        geo = geopca(a.unitsphere())
        g = geo.project(a.unitsphere())
        return g.dataset

def cluster_data(data, method, plot, k_opt=0):
    if k_opt == 0:
        if method == 'kmeans':
            sil = []
            kmax = 15
            K = range(2,kmax+1)
            for k in K:
                kmeans_sil = KMeans(n_clusters = k, random_state=42).fit(data)
                labels = kmeans_sil.labels_
                sil.append(silhouette_score(data, labels, metric = 'euclidean',random_state=42))

            n=1
            prev=0
            for i in sil:
                n+=1 #value of k for i
                LOG.info('working on k = {}'.format(n))
                if i > prev:
                    prev = i
                else:
                    k_opt = n-1 
                    break
            
            if k_opt == 0:
                LOG.warning("Could not determine optimum value for k, setting k=4")
                k_opt = 4

        if plot == 'True':
            #elbow_plot_calc
            Sum_of_squared_distances = []

            for k in K:
                km = KMeans(n_clusters=k)
                km = km.fit(data)
                Sum_of_squared_distances.append(km.inertia_)
                
            #plot elbow plot
            plt.plot(K, Sum_of_squared_distances, 'bx-')
            plt.xlabel('k')
            plt.ylabel('Sum of squared distances')
            plt.savefig('elbow_plot.png')
            plt.close()

            #plot silhouette graph
            plt.plot(K, sil, 'bx-')
            plt.xlabel('k')
            plt.ylabel('Silhouette Score')
            #plt.title('Silhouette Plot For Optimal k')
            plt.savefig('Silhouette_plot.png')
            plt.close()

    #final k_means
    LOG.info('The optimal value of k is {}'.format(k_opt))
    kmeans = KMeans(n_clusters=k_opt)
    kmeans.fit(data)
    x_pca_kmeans = kmeans.predict(data)
    kmeans_df = pd.DataFrame(data)
    # print(kmeans_df)
    PCs=5
    columns = []
    for n in range(1, PCs+1):
        columns.append("PC"+str(n))
    kmeans_df.columns=columns
    kmeans_df['cluster'] = x_pca_kmeans
    kmeans_df.to_csv('groups.csv', index=None)

    #plot pca_distribution
    plt.scatter(data[:,0], data[:,1], c=kmeans_df['cluster'], s=2)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig('pca_distribution.png')
    plt.close()

    return k_opt

def pick_from_cluster(energies_list, k_opt, by='Energy'):
    df = pd.read_csv('groups.csv')
    df['energy']=energies_list
    conformer_list=[]
    list_size = df.shape[0]
    for n in range(1, list_size+1):
        conformer_list.append("conformer_"+str(n))
    df['conformer']=conformer_list
    starting_positions=[]
    for cluster in range(k_opt):
        cluster_list = df.loc[df['cluster'] == cluster]
        if by == 'Energy':
            best=cluster_list['energy'].argmin()
            starting_positions.append(best+1)

    return starting_positions

    # if by == 'Energy': #find the lowest energy conformer from each cluster
    #     for eng in engs:
    #         n+=1
    #         name="conformer_"+n
    #         print(eng[1])

    # if by == 'Surface Area': #find the conformer with the highest surface area from each cluster
    # for cluster in range(k_opt):
    #     cluster_list = df.loc[df['cluster'] == cluster]
    #     highest_area=cluster_list.iloc[cluster_list['conf_area'].argmax()]['conformer']
    #     print(highest_area,cluster)
    #     with open('starting_positions.txt', 'a') as positions:
    #         positions.write("{}\n".format(highest_area))

def crest(molecule, processes):
    bashCommand = "crest {} -T {} -v4 -ewin 9".format(molecule, processes)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
