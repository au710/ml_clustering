import os
import re
from pathlib import Path
from shutil import copy
import random

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from skimage import data, io, color
from skimage.filters import threshold_otsu
#import cv2
import pandas as pd

from wand.image import Image

image_set = "thresholding"

# read in image names
name_list = []
path_list = Path(image_set).glob("*.png")
for image_path in path_list:
    name = image_path.stem
    name_list.append(name)
category = ["unknown"] * len(name_list)

# read in actual images
all_flat = []
for i, name in enumerate(name_list):
    img_flat = np.empty(0)
    img = io.imread(f"./{image_set}/{name}.png", as_gray=True)
    img_flat = np.append(img_flat, img.flatten())
    all_flat.append(img_flat)

df = pd.DataFrame(all_flat)
df.index = name_list

# pca analysis ran here
n_comp = 2
pca = PCA(n_components=n_comp)
im1_pca = pca.fit_transform(df)

pca_df = pd.DataFrame(im1_pca)
pca_df["label"] = name_list
pca_df["cat"] = category

# Visualise PCA results
groups = pca_df.groupby("cat")
for name, group in groups:
    plt.scatter(group[0], group[1], label=name)
plt.legend(loc="best")
plt.savefig("./results/pca_cluster.pdf")

pca_df.to_csv("pca.csv")


