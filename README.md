# ml_clustering

to run the following should be sufficient to create environment

```
conda create --name cracking python=3.7
source activate cracking
pip install -r requirements.txt
...

Run's pca analysis and then k-means clustering to group results together

Otsu threshholding carried out initially to make clustering analysis easier

Analysis is repeated in subset folder upon the largest cluster
