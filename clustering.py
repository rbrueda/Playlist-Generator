#goal: perform multiple clustering algorithms for different categories

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# create a blob of 200 data points
#we only want 2-dimensional data in this case

#to do: make these variables
dataset = make_blobs(n_samples=200,
                     n_features=2,
                     centers=4,
                     cluster_std = 1.6,
                     random_state = 50)

points = dataset[0]

#create a dendrogram 
dendrogram = sch.dendrogram(sch.linkage(points, method='ward'))

#perform the clustering here
#to get n clusters, we can try to find how many unique elements there are -- for examle genre 
hc = AgglomerativeClustering(n_clusters=4,linkage='ward')

y_hc = hc.fit_predict(points)

plt.scatter(points[y_hc ==0,0], points[y_hc == 0,1], s=100, c='cyan')
plt.scatter(points[y_hc ==1,0], points[y_hc == 1,1], s=100, c='yellow')
plt.scatter(points[y_hc ==2,0], points[y_hc == 2,1], s=100, c='red')
plt.scatter(points[y_hc ==3,0], points[y_hc == 3,1], s=100, c='green')

#add the centroids
plt.scatter()

#step 1: vectorize all my data
#step 2: find optimal k value
#step 3: to perform k-mean clustering algorithm
#step 4: label data
#step 5: put data in separate datasets
#step 6: add it to UI
#step 7: if even possible -- to sync to spotify

