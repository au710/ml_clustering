import os
import re
from pathlib import Path
from shutil import copy
import random

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from skimage import data, io, color
from skimage.filters import threshold_otsu
#import cv2
import pandas as pd

from wand.image import Image


# clustering analysis here
pca_df = pd.read_csv("pca.csv")

# build feature array of shape (n_samples, n_features)
X = np.array(pca_df[["0", "1"]])

#elbow analysis here
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k,init='k-means++').fit(X)
    kmeanModel.fit(X)
    distortions.append(kmeanModel.inertia_)
#    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1))/ X.shape[0])

wcss = distortions
l_dist = []
for i in range(1,10):
    p1 = np.array([1,wcss[0]])
    p2 = np.array([9,wcss[8]])
    p = np.array([i,wcss[i-1]])
    print(p)
    l_dist.append(np.linalg.norm(np.cross(p2-p1,p1-p))/np.linalg.norm(p2-p1))

    
# Plot the elbow
fig, ax1 = plt.subplots()
color = 'tab:blue'
ax1.plot(K, distortions, color=color,marker='x')
ax1.set_xlabel('k')
ax1.set_ylabel('WCSS',color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2=ax1.twinx() #initiate second axes with same x axis
color ='tab:red'
ax2.plot(K,l_dist,color=color,marker='o')
ax2.set_ylabel('Distance',color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('The Elbow Method showing the optimal k')

plt.savefig("results/elbow.png")
plt.show()

