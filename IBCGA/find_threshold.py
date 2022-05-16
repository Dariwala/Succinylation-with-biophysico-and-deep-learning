from scipy.optimize import differential_evolution
from compute_label_from_preds_with_threshold import compute_label_from_preds_with_threshold
import numpy as np
# import scipy

# print(scipy.__version__)

def create_label(pos,neg):
    y_true = np.ones((pos,))
    y_true = np.concatenate((y_true,np.zeros((neg,))))
    return y_true

def loss_function(threshold, preds):
    actual, preds_y_float = compute_label_from_preds_with_threshold(threshold,preds)
    # y_true = create_label(696,686)
    # y_true = create_label(1479,16457)
    # y_true = create_label(254,2977)
    import pickle
    with open('val_label_ind','rb') as f:
        y_true = pickle.load(f)
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

    return 1 - mcc


def find_threshold(preds):
    bound_w = [(0.4,0.6)]
    search_arg = (preds,)
    result = differential_evolution(loss_function, bound_w, args=search_arg, tol=1e-7)
    print(result['x'][0],1-loss_function(result['x'][0],preds))

if __name__ == '__main__':
    preds = np.loadtxt('ensemble_ind_val_loss.txt')
    # preds = preds[:,10:15]
    find_threshold(preds)