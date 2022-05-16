import numpy as np
from compute_label_from_preds_with_threshold import compute_label_from_preds_with_threshold
from sklearn.metrics import accuracy_score, recall_score, roc_curve, roc_auc_score
import sys

preds = np.loadtxt('ensemble_ind_loss.txt')
# preds = np.delete(preds,[16,17,19,22,26,33],axis = 1)
# preds = preds[:,10:15]
# preds = np.concatenate((preds[:,:15],preds[:,31:32]), axis = 1)
# preds_loss = np.loadtxt('ensemble_ind_val_temp.txt')
# preds = np.concatenate((preds, preds_loss), axis=1)

# all_preds = np.loadtxt('all_r.txt')
# preds = preds[:,15:]
# preds_1 = np.loadtxt('ensemble_val.txt')
# preds = np.concatenate((preds_1,preds), axis = 1)
# np.savetxt('ensemble_ind_loss.txt',preds)

# for thresh in range(int(sys.argv[1]),int(sys.argv[2])):
actual, y_pred_float = compute_label_from_preds_with_threshold(float(sys.argv[1]), preds)

def create_label(pos,neg):
    y_true = np.ones((pos,))
    y_true = np.concatenate((y_true,np.zeros((neg,))))
    return y_true

# y_true = create_label(696,686)
# y_true = create_label(1479,16457)
y_true = create_label(254,2977)
# import pickle
# with open('val_label_ind','rb') as f:
#     y_true = pickle.load(f)

tp = tn = fp = fn = 0

for i in range(y_true.shape[0]):
    if y_true[i] == 0:
        if actual[i] == 0:
            tn += 1
        else:
            fp += 1
    else:
        if actual[i] == 1:
            tp += 1
        else:
            fn += 1

sn = tp / (tp + fn)
sp = tn / (fp + tn)
acc = (tp + tn) / (tp + tn + fp + fn)
mcc = (tp * tn - fp * fn) / (np.sqrt((tp+fn)*(tp+fp)*(tn+fn)*(tn+fp)))

print(sn,sp,acc,mcc)

# fprs = []
# tprs = []
# aucs = []

# for i in range(preds.shape[1]):
#     fpr, tpr, _ = roc_curve(y_true, preds[:,i])
#     auc = roc_auc_score(y_true, preds[:,i])
#     fprs.append(fpr)
#     tprs.append(tpr)
#     aucs.append(auc)

# # fpr, tpr, _ = roc_curve(y_true, all_preds)
# # auc = roc_auc_score(y_true, all_preds)

# from matplotlib import pyplot as plt

# for i in range(preds.shape[1]):
#     if i in [0,1,2,6,7]:
#         plt.plot(fprs[i], tprs[i], linestyle='--', label='r = '+str(i+2)+" (AUC = "+str(format(aucs[i],".4f"))+")")
# # plt.plot(fpr, tpr, linestyle='--', label='r = 29'+" (AUC = "+str(format(auc,".4f"))+")")
# plt.legend()
# plt.show()