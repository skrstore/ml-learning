import matplotlib.pyplot as plt
import pandas as pd

# %matplotlib inline
import numpy as np

X = np.array(
    [
        [5, 3],
        [10, 15],
        [15, 12],
        [24, 10],
        [30, 30],
        [85, 70],
        [71, 80],
        [60, 78],
        [70, 55],
        [80, 91],
    ]
)


from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(
    n_clusters=2, affinity="euclidean", linkage="ward"
)  # "euclidean" (distance between the datapoints).
cluster.fit_predict(X)


print(cluster.labels_)


plt.scatter(X[:, 0], X[:, 1], c=cluster.labels_, cmap="rainbow")
plt.show()
