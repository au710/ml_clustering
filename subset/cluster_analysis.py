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

# number of clusters found by elbow method
#y_pred = KMeans(n_clusters=4).fit_predict(X)
kmeanModel = KMeans(n_clusters=4).fit(X)
y_pred=kmeanModel.predict(X)
pca_df["cluster"] = y_pred
pca_df.to_csv("clusters.csv")

clusters = pca_df.groupby("cluster")
plt.figure()
#plot clustering
for id, cluster in clusters:
    plt.scatter(cluster["0"], cluster["1"], label=id)
    for index, row in cluster.iterrows():
        plt.text(row["0"]+0.5, row["1"], row["cluster"], size=10)
plt.legend(loc=0)
plt.xlabel("PC1", size=15)
plt.ylabel("PC2", size=15)
plt.tight_layout()
plt.savefig("results/clustering.png")

# make montages of clusters
selections = []
for cluster_id, cluster in clusters:
    print(f"INFO: Cluster id {cluster_id}")
    
    cluster_dir = f"clusters/{cluster_id}"
    os.system(f"rm -r {cluster_dir}")
    
    Path(cluster_dir).mkdir(parents=True, exist_ok=True)
    span = len(cluster.label)
    plt.subplots(1, span, figsize=(span, 4))
    
    selection = []
    for j, label in enumerate(cluster.label):
        print(label)
        selection.append(label)
        os.system(f"cp assets/{label}* {cluster_dir}")
        column = j + 1
        plt.subplot(1, span, column)
        plt.axis("off")
        fpath = f"../coloured/{label}.png"
        plt.imshow(io.imread(fpath))
    selections.append(selection)        
    plt.savefig(f"results/cluster_{cluster_id}.png")


# Trim montages in place
run_paths = list(Path("results").glob("cluster_?.png"))
for source in run_paths:
    with Image(filename=source.as_posix()) as img:
        img.trim()
        img.save(filename=source.as_posix())

i=0        
# choose three random images from each cluster
for s in selections:
    a = random.sample(s, 3)
    print(a)
    span = 3
    plt.subplots(1, span)#, figsize=(span, 4))
    
    rnd = []
    for j, label in enumerate(a):
        print(label)
        rnd.append(label)
        column = j + 1
        plt.subplot(1, span, column)
        plt.axis("off")
        fpath = f"../coloured/{label}.png"
        plt.imshow(io.imread(fpath))
    plt.savefig(f"results/clusterRND_{i}.png")
    i+=1
    
# %% Trim montages in place
run_paths = list(Path("results").glob("clusterRND_?.png"))
for source in run_paths:
    with Image(filename=source.as_posix()) as img:
        img.trim()
        img.save(filename=source.as_posix())


