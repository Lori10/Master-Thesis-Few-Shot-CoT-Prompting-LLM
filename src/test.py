from sklearn.metrics import pairwise_distances

print(pairwise_distances([[1,2,3]], [[1,2,3]], metric='euclidean', force_all_finite=False).ravel().astype(float))