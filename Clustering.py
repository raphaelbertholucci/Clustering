
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore') 

df = pd.read_csv('taiwan_data.csv')
df.head()

del df['Bankrupt']

####    K-means method    ####

import sklearn.cluster as cluster
import seaborn as sns

kmeans = cluster.KMeans(n_clusters=30 ,init="k-means++", max_iter=1000)
kmeans = kmeans.fit(df)

df['Clusters'] = kmeans.labels_

print('\n\n K-means method:\n\n')
print(kmeans.labels_)

df.to_csv('clusters.csv', index = False)

####    PCA method   ####

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X_std = StandardScaler().fit_transform(df)
pca = PCA()
principalComponents = pca.fit_transform(X_std)
features = range(pca.n_components_)
PCA_components = pd.DataFrame(principalComponents)

plt.scatter(PCA_components[0], PCA_components[1], edgecolor='none', alpha=0.7, s=40, cmap=plt.cm.get_cmap('nipy_spectral', 10))
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.colorbar()
plt.show()

####    Elbow Method to Indetify Clusters    ####
####    WSS -> Within-Cluster-Sum of Squared    ####

K = range(1,31)
wss = []
for k in K:
    kmeans=cluster.KMeans(n_clusters=k,init="k-means++")
    kmeans=kmeans.fit(df)
    wss_iter = round(kmeans.inertia_, 4)
    wss.append(wss_iter)

mycenters = pd.DataFrame({'Clusters' : K, 'WSS' : wss})
print('\n\n Using Elbow Method:\n\n')
print(mycenters)

plt.plot(K, wss, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(K)
plt.show()