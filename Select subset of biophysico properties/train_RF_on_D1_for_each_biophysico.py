from random import random
import numpy as np
# from train_test_split import train_test
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle

for c in range(16,17):
    # with open('rf_566_biophysico','ab') as f:
    #     pickle.dump(566,f)
    # file = open(str(c)+'.txt','w')

    for i in range(566):
        train_dataset = np.loadtxt('..//Datasets//D1//Bio-physico//Train//'+str(c)+'_'+str(i)+'.txt',dtype=float)
        val_dataset = np.loadtxt('..//Datasets//D1//Bio-physico//Val//'+str(c)+'_'+str(i)+'.txt',dtype=float)


        X_train = train_dataset[:,:-1]
        y_train = train_dataset[:,-1]

        X_test = val_dataset[:,:-1]
        y_test = val_dataset[:,-1]

        # train_test(X,y,.2)

        # size = 0.2

        # X_train, X_test, y_train, y_test = train_test(X,y,size=size)

        # print(X_train.shape)
        # print(y_train.shape)
        # print(X_test.shape)
        # print(y_test.shape)

        # clf = make_pipeline(StandardScaler(), SGDClassifier(random_state=1, max_iter=10000))
        clf = RandomForestClassifier(n_estimators = 300, 
        random_state=1,n_jobs=-1,
        bootstrap=False,min_samples_leaf=5,max_features="log2")
        clf.fit(X_train, y_train)

        # with open('rf_566_biophysico','ab') as f:
        #     pickle.dump(clf,f)

        # print(clf.score(X_test,y_test))

        # y_pred = clf.predict(X_train)
        y_pred = clf.predict(X_test)

        # cm = confusion_matrix(y_train, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # print(cm)

        # probs = clf.predict_proba(X_test)[:,1]

        tp = cm[1][1]
        fn = cm[1][0]
        tn = cm[0][0]
        fp = cm[0][1]

        sn = tp / (tp + fn)
        sp = tn / (fp + tn)
        acc = (tp + tn) / (tp + tn + fp + fn)
        mcc = (tp * tn - fp * fn) / (np.sqrt((tp+fn)*(tp+fp)*(tn+fn)*(tn+fp)))

        print(i,sn,sp,acc,mcc)
        # file.write(str(i) + ' '+str(sn)+" " +str(sp) + " " + str(acc) + " " + str(acc) + '\n')

        # for i in range(X_test.shape[0]):
        #     label = y_pred[i]
        #     if label == y_test[i]:
        #         samples[i] += 1
        # break
    # file.close()
