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
K = range(1,20)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.savefig("results/elbow.png")
plt.show()


