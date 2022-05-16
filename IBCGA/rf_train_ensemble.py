from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import numpy as np

def rf_train(train_dataset, val_dataset,num=5,all=False):
    X_train = train_dataset[:,:-1]
    y_train = train_dataset[:,-1]

    X_test = val_dataset[:,:-1]
    y_test = val_dataset[:,-1]

    clf = RandomForestClassifier(n_estimators = num, random_state=1,n_jobs=-1,bootstrap=True,oob_score=True)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # print(y_test.shape)

    count = np.zeros((y_test.shape[0],),dtype=np.int)

    count_one = np.zeros((int(np.sum(y_test))),dtype=np.int)
    count_zero = np.zeros((y_test.shape[0] - int(np.sum(y_test))),dtype=np.int)

    one_ind = 0
    zero_ind = 0

    for i in range(y_test.shape[0]):
        count[i] += (y_test[i] == y_pred[i])

        if y_test[i] == 0:
            count_zero[zero_ind] += count[i]
            zero_ind += 1
        else:
            count_one[one_ind] += count[i]
            one_ind += 1
    return count, count_zero, count_one