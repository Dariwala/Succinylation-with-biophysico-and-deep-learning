from xml.sax.handler import feature_external_ges
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.inspection import permutation_importance
import pandas as pd
import matplotlib.pyplot as plt
from delete_columns_from_numpy import delete_columns_from_numpy_array

def rf_train(train_dataset, val_dataset,num=5,all=False):
    X_train = train_dataset[:,:-1]
    y_train = train_dataset[:,-1]

    X_test = val_dataset[:,:-1]
    y_test = val_dataset[:,-1]

    clf = RandomForestClassifier(n_estimators = num, random_state=1,n_jobs=-1,bootstrap=False,min_samples_leaf=5,max_features="log2")
    clf.fit(X_train, y_train)

    # result = permutation_importance(
    #     clf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    # )

    # bad_indices = np.where(result.importances < 0)

    # bad_indices[0].sort()
    
    # X_train = delete_columns_from_numpy_array(X_train, bad_indices[0])
    # X_test = delete_columns_from_numpy_array(X_test, bad_indices[0])

    # clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    predictions = clf.predict_proba(X_test)[:,1]
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
    # print(type(forest_importances))
    # print(forest_importances)

    # fig, ax = plt.subplots()
    # forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    # ax.set_title("Feature importances using permutation on full model")
    # ax.set_ylabel("Mean accuracy decrease")
    # fig.tight_layout()
    # plt.show()

    if all:
        return predictions, sn, sp, acc, mcc
    else:
        return mcc