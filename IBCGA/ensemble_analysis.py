import pickle
from fitness_for_ensemble import fitness
from load_data import load_essentials
from load_data import load_essentials_test
from load_data import load_essentials_test_only_train
import numpy as np

proteins, val_proteins, fastas, val_fastas, p_samples, n_samples, positives, negatives, val_positives, val_negatives = load_essentials_test_only_train()

effectives = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]
effectives = list(range(1,21))

with open('bests_optimzed(16)_50_8_6_1_20_5_0_mcc','rb') as f:
    result = []
    result_zero = []
    result_one = []
    ind = 0
    while True:
        try:
            r,best = pickle.load(f)
            # ind += 1
            # if ind <= 11: #for acc only, because this file contains some garbages
            #     continue
            if r not in effectives:
                continue
            # print(r)
            count, count_zero, count_one = fitness(best,proteins, val_proteins, fastas, val_fastas, p_samples, n_samples, positives, negatives, val_positives, val_negatives, True, False, 300, True)
            if isinstance(result, list):
                result = count
                result_zero = count_zero
                result_one = count_one
            else:
                result += count
                result_zero += count_zero
                result_one += count_one
        except EOFError:
            break

tn = 0.0
tp = 0.0
fn = 0.0
fp = 0.0
threshold = 0.5

# for i in range(result.shape[0]):
#     c += (result[i]/len(effectives) >= threshold)

for i in range(result_zero.shape[0]):
    tn += (result_zero[i]/len(effectives) > (1 - threshold))
    fp += (result_zero[i]/len(effectives) <= (1 - threshold))

for i in range(result_one.shape[0]):
    tp += (result_one[i]/len(effectives) >= threshold)
    fn += (result_one[i]/len(effectives) < threshold)

# print(tp,tn,fp,fn,(tp+fn)*(tp+fp)*(tn+fn)*(tn+fp))

sn = tp / (tp + fn)
sp = tn / (fp + tn)
acc = (tp + tn) / (tp + tn + fp + fn)
mcc = (tp * tn - fp * fn) / (np.sqrt((tp+fn)*(tp+fp)*(tn+fn)*(tn+fp)))

print(sn,sp,acc,mcc)