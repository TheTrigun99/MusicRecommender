import faiss
import numpy as np
xq = np.load("embedding_last.npy")
xb= np.load("embeddings_rest.npy")
from faiss.contrib.evaluation import knn_intersection_measure
print(faiss.MatrixStats(xb).comments)
Dref, Iref = faiss.knn(xq, xb, k=2, metric=faiss.METRIC_INNER_PRODUCT)
print(Dref,Iref)