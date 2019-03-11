import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, metrics
from sklearn.model_selection import train_test_split

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

import seaborn as sns

from pyDOE import lhs

runs_df = pd.read_csv("data/run_list.csv") # conversion of csv file to dataframe
clusters_df = pd.read_csv("clusters.csv")

# create feature dataframe by joining on image label
feature_df = pd.merge(runs_df, clusters_df, on='label')


mask = ['gc', 'k'] #extract parameters 

features = np.array(feature_df[mask]) #  array of parameters
targets = feature_df["cluster"].values # array of predicted clusters

# classifier = svm.SVC(gamma=0.005)
# classifier = svm.SVC(gamma='auto')
classifier = GaussianProcessClassifier(1.0 * RBF(1.0))


X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.20, random_state=42)
classifier.fit(X_train, y_train)

expected = y_test # correct ans
predicted = classifier.predict(X_test)


print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s\n" % metrics.confusion_matrix(expected, predicted))

print("Comparing expected and predicted:", np.column_stack((expected, predicted)))


# %% Run prediction

# lhs_samples = lhs(2, samples=40)
# samples_df = pd.DataFrame(lhs_samples)
# samples_df.columns = ["a", "b"]

# a_lower = 0.03
# a_upper = 30

# b_lower = 0.03
# b_upper = 0.34

# samples_df.a = samples_df.a.apply(
#     lambda x, xmin=a_lower, xmax=a_upper: (xmax - xmin) * x + xmin
# )

# samples_df.b = samples_df.b.apply(
#     lambda x, xmin=b_lower, xmax=b_upper: (xmax - xmin) * x + xmin
# )

samples_df = feature_df[["k", "gc", "cluster"]]
samples_df.columns = ["a", "b", "cluster"]



h = .01
x_min, x_max = samples_df.a.min(), samples_df.a.max()
y_min, y_max = samples_df.b.min(), samples_df.b.max()
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.tab10,
           aspect='auto', origin='lower', alpha=0.2)
plt.xlim(a_lower, a_upper)
plt.ylim(b_lower, b_upper)


prediction = classifier.predict(samples_df[["a", "b"]])
samples_df["Cluster"] = prediction

from matplotlib.colors import ListedColormap
cmap = ListedColormap(sns.color_palette())
cluster_count = samples_df.Cluster.unique().size
# plt.figure()
palette = sns.color_palette("tab10", cluster_count)
sns.scatterplot(x='a', y='b', hue="Cluster", data=samples_df, palette=palette)
plt.legend(loc=1)
plt.xlabel("$a_0$")
plt.ylabel("$b_0$")
plt.savefig("results/prediction.pdf")
