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
from skimage.filters import threshold_otsu, rank
from skimage.morphology import disk
import pandas as pd

from wand.image import Image

my_dpi=96
image_set = "assets"

# read in image names
name_list = []
path_list = Path(image_set).glob("*.png")
for image_path in path_list:
    name = image_path.stem
    name_list.append(name)
category = ["unknown"] * len(name_list)

os.system(f"rm -r thresholding")
os.system(f"mkdir thresholding")
os.system(f"mkdir local_thresholding")

# read in actual images
all_flat = []
for i, name in enumerate(name_list):
    img_flat = np.empty(0)
    img = io.imread(f"./{image_set}/{name}.png", as_gray=True)
    thresh = threshold_otsu(img)
    bin_img = img >= thresh
    #ret,bin_img = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
    plt.figure(figsize=(100/my_dpi, 100/my_dpi), dpi=my_dpi)
    plt.imshow(bin_img, cmap=plt.cm.gray)
    plt.axis('off')
    plt.savefig(f"thresholding/{name}.png", dpi=my_dpi)



