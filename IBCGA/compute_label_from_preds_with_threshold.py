import numpy as np

def compute_label_from_preds_with_threshold(threshold, preds):
    actual = np.zeros((preds.shape[0],))
    y_pred_float = np.zeros((preds.shape[0],))

    for i in range(preds.shape[0]):
        avg = 0
        for j in range(preds.shape[1]):
            avg += preds[i][j]
        avg /= preds.shape[1]
        y_pred_float[i] = avg
        actual[i] = 0 if avg < threshold else 1
    return actual, y_pred_float