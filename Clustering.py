
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

# We will use 2 Variables for this example
kmeans = cluster.KMeans(n_clusters=30 ,init="k-means++")
kmeans = kmeans.fit(df)
print('\n\n K-means method:\n\n')
print(kmeans.cluster_centers_)


####    Elbow Method to Indetify Clusters    ####
####    WSS -> Within-Cluster-Sum of Squared    ####

K = range(1,30)
wss = []
for k in K:
    kmeans=cluster.KMeans(n_clusters=k,init="k-means++")
    kmeans=kmeans.fit(df)
    wss_iter = round(kmeans.inertia_, 15)
    wss.append(wss_iter)

mycenters = pd.DataFrame({'Clusters' : K, 'WSS' : wss})
print('\n\n Using Elbow Method:\n\n')
print(mycenters)