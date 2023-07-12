import pickle
import numpy as np
import pandas as pd
# from sklearn.metrics import pairwise_distances

# scores = pairwise_distances([[1,2,3]], [[1,2,3]], metric='euclidean', force_all_finite=True).ravel().astype(float)
# print(scores)

file = open("embeddings", "rb")
embeddings = np.load(file)

with open('uncertainties.pkl', 'rb') as f:
    uncertainty_list = pickle.load(f)

    uncertainties_series = pd.Series(data=uncertainty_list, index=questions_idxs)
    first_question_idx = list(uncertainties_series.sort_values(ascending=False).head(1).index)[0]
    selected_idxs = [first_question_idx]
    selected_data = [embeddings[first_question_idx]]