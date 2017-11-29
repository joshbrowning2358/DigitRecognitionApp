import cPickle as pickle
from __future__ import division

import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans

with open('data/train_bounding_boxes.p', 'rb') as f:
    bounding_boxes = pickle.load(f)
with open('data/extra_bounding_boxes.p', 'rb') as f:
    bounding_boxes = pickle.load(f)

labels = []
widths = []
heights = []
asp_ratios = []

for k, v in bounding_boxes.items():
    for box in v:
        label, x, y, width, height = box
        labels += [label]
        widths += [width]
        heights += [height]
        asp_ratios += [width/height]

# Aspect ratios seem very centered around 0.5.  Using other anchor boxes may be difficult.
sns.distplot(asp_ratios)
sns.distplot(widths)
sns.distplot(heights)

# Clustering the height and width gives clusters with aspect ratios all very close to 0.5
clustering_model = KMeans(n_clusters=3)
clustering_model.fit(np.transpose(np.vstack([widths, heights])))
clustering_model.cluster_centers_

# Clustering aspect ratios directly gives different centers; maybe this is the best approach?
clustering_model = KMeans(n_clusters=3)
clustering_model.fit(np.array(asp_ratios).reshape((-1, 1)))
clustering_model.cluster_centers_
