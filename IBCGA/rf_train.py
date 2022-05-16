from xml.sax.handler import feature_external_ges
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.inspection import permutation_importance
import pandas as pd
import matplotlib.pyplot as plt
from delete_columns_from_numpy import delete_columns_from_numpy_array
from imblearn.ensemble import BalancedRandomForestClassifier
import seaborn as sns
from scipy.stats import spearmanr

def compute_spearman(arr):
    matrix = np.zeros((arr.shape[1],arr.shape[1]))
    for i in range(arr.shape[1]):
        for j in range(arr.shape[1]):
            # print(np.reshape(arr[:,i],(arr.shape[0],)).shape)
            # print(np.reshape(arr[:,j],(arr.shape[0],)).shape)
            matrix[i][j] = spearmanr(np.reshape(arr[:,i],(arr.shape[0],)),np.reshape(arr[:,j],(arr.shape[0],)))[0]
    return matrix

def rf_train(train_dataset, val_dataset,num=5,all=False):
    X_train = train_dataset[:,:-1]
    y_train = train_dataset[:,-1]

    # matrix = np.corrcoef(X_train, rowvar=False)
    # matrix = compute_spearman(X_train)
    # names = [i for i in range(1,65)]
    # df = pd.DataFrame(matrix, index=names, columns=names)
    # mask = np.triu(np.ones_like(matrix, dtype=bool))
    # # print(matrix.shape)
    # fig, ax = plt.subplots()
    # cmap = sns.diverging_palette(250, 15, s=75, l=40,
    #                          n=9, center="light", as_cmap=True)
    # _ = sns.heatmap(df, center=0, annot=False, 
    #             fmt='.1f', square=True, cmap=cmap
    #             , mask = mask
    #             # , xticklaels = [i for i in range(1,65,3)]
    #             # , yticklabels = [i for i in range(1, 65,3)]
    #             )
    # plt.show()

    X_test = val_dataset[:,:-1]
    y_test = val_dataset[:,-1]

    # clf = RandomForestClassifier(n_estimators = num, random_state=1,
    # n_jobs=-1,bootstrap=False,min_samples_leaf=5,max_features="log2"
    # #, class_weight='balanced_subsample'#{0:1,1:4}
    # )
    clf = BalancedRandomForestClassifier(n_estimators = num, random_state=1,
    n_jobs=-1,bootstrap=False,min_samples_leaf=5,max_features="log2"
    )
    clf.fit(X_train, y_train)

    # result = permutation_importance(
    #     clf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    # )

    # print(result.importances_mean)
    # bad_indices = np.where(result.importances_mean < 0)
    # print(bad_indices)

    # bad_indices[0].sort()
    
    # X_train = delete_columns_from_numpy_array(X_train, [35])
    # X_test = delete_columns_from_numpy_array(X_test, [35])

    # clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # predictions = clf.predict_proba(X_test)[:,1]
    # predictions = predictions.reshape((predictions.shape[0],1))
    # prev = np.loadtxt('ensemble_ind_val_temp.txt')
    # prev = prev.reshape((prev.shape[0],-1))
    # # prev = prev[:,:-1]
    # if prev.shape[0] != 0:
    #     prev = np.concatenate((prev,predictions),axis = 1)
    # else:
    #     prev = predictions
    # np.savetxt('ensemble_ind_val_temp.txt',prev)

    # print(predictions.shape)

    cm = confusion_matrix(y_test, y_pred)

    tp = cm[1][1]
    fn = cm[1][0]
    tn = cm[0][0]
    fp = cm[0][1]

    sn = tp / (tp + fn)
    sp = tn / (fp + tn)
    acc = (tp + tn) / (tp + tn + fp + fn)
    mcc = (tp * tn - fp * fn) / (np.sqrt((tp+fn)*(tp+fp)*(tn+fn)*(tn+fp)))

    # forest_importances = pd.Series(result.importances_mean, index=[i for i in range(X_train.shape[1])])
    # names = [i for i in range(1,65)]
    # df = pd.DataFrame(forest_importances, index=names)
    # print(type(forest_importances))
    # print(forest_importances)

    # fig, ax = plt.subplots(figsize=(12,5))
    # df.plot.bar(yerr=result.importances_std, ax=ax)
    # ax.set_title("Feature importances using permutation on full model")
    # ax.set_ylabel("Mean accuracy decrease")
    # ax.get_legend().remove()
    # fig.tight_layout()
    # plt.show()

    if all:
        return sn, sp, acc, mcc
    else:
        return mcc