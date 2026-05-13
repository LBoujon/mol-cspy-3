import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('pca.csv')

#df = df.drop([0])

df.columns=['PC1','PC2']

x_pca = df

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
#print(sil)
for i in sil:
    n+=1 #value of k for i
#    print('working on k = {}'.format(n))
    if i > prev:
        prev = i
    else:
        k_opt = n-1
        break


#print('The optimal value of k is {}'.format(k_opt))

#plot silhouette graph

plt.plot(K, sil, 'bx-')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
#plt.title('Silhouette Plot For Optimal k')
plt.savefig('Silhouette_plot.png')
plt.close()

#run k-means for optimimum k.
kmeans = KMeans(n_clusters=k_opt)
kmeans.fit(x_pca)
x_pca_kmeans = kmeans.predict(x_pca)

df['cluster'] = x_pca_kmeans

#plot pca distribution

#print(df)
#plt.scatter(df['PC1'], df['PC2'], c=df['cluster'])
plt.scatter(df['PC1'], df['PC2'])
plt.xlabel('PC1')
plt.ylabel('PC2')


plt.savefig('pca_distribution.png')
plt.close()

#obtain conformer from each cluster
#what metric should be used?

#by area
areas = pd.read_csv('../conformers/areas.txt', sep=' ', header=None)
areas.columns=['conformer','area']

conf_list=[]

#for every conformers in cluster 
for conf in range(1000):  #0-999
    conf_num = int(conf) + 1
    conf_name = 'conformer_' + str(conf_num) + '.mol'
    conf_list.append(conf_name)
    
df['conformer'] = conf_list

df['conf_area'] = df['conformer'].map(areas.set_index('conformer')['area'])

df.to_csv('conformer_data.csv')

if os.path.isfile('starting_positions.txt'):
    os.remove('starting_positions.txt')

for cluster in range(k_opt):
    cluster_list = df.loc[df['cluster'] == cluster]
    highest_area=cluster_list.iloc[cluster_list['conf_area'].argmax()]['conformer']
    print(highest_area,cluster)
    with open('starting_positions.txt', 'a') as positions:
        positions.write("{}\n".format(highest_area))

