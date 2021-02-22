import os
import pandas as pd
from sklearn.mixture.gaussian_mixture import GaussianMixture as GMM
import numpy as np
import matplotlib.pyplot as plt


PATH_DATA = r"D:\Speciale\Data\Exploration\SimpleStats\data.csv"
df = pd.read_csv(PATH_DATA)

N = int(df.values.shape[0] / 3)

gmms = []
for i in range(3):
    gmm = GMM(n_components=1, covariance_type='full')
    gmm.fit(df.values[(i*N):((i+1)*N), 1:])
    gmms.append(gmm)
    scores = gmm.score_samples(df.values[(i*N):((i+1)*N), 1:])

all_scores = []
for i in range(3):
    scores = gmms[i].score_samples(df.values[:, 1:])
    all_scores.append(scores)

scores_c1 = np.array(all_scores[0])
scores_c2 = np.array(all_scores[1])
scores_c3 = np.array(all_scores[2])

A = np.transpose(np.stack((scores_c1, scores_c2, scores_c3)))
pred_class = np.argmax(A, axis=1)

diff = np.sum(np.abs(np.sign(pred_class - df.values[:,0])))

print("Training accuracy = %.1f%%" % (100 - 100*diff/df.values.shape[0]))
