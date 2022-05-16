from numpy import dtype
from scipy.stats import spearmanr
import numpy as np
import copy

def spearman(dataset, threshold):
    features = dataset.shape[1] - 1
    coef = np.zeros((features,features),dtype=np.float)
    p = np.zeros((features,features),dtype=np.float)

    equal_columns = np.all(dataset == dataset[0,:],axis=0)

    for i in range(features):
        for j in range(features):
            if i != j and (equal_columns[i] == False and equal_columns[j] == False):
                c, pv = spearmanr(dataset[:,i], dataset[:,j])
                coef[i][j] = np.abs(c)
                p[i][j] = pv

    deleted_features = []

    while True:
        max_coef = np.amax(coef)
        if max_coef >= threshold:
            index = np.where(coef == max_coef)
            # assert len(index) == 1
            # if len(index) != 1:
            #     print(coef)
            #     print(index)

            row = index[0][0]
            col = index[1][0]

            if p[row][col] > 0.05:
                coef[row][col] = -2
            else:
                deleted_features.append(col)
                coef[:,col] = -2
                coef[col,:] = -2
        else:
            break
    
    # print(dataset.shape)
    deleted_features.sort()
    return deleted_features