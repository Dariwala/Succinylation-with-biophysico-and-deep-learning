# from load_data import load_data
from load_data import load_data_deepsucc as load_data
# from load_data import load_data_gpsuc as load_data
from rf_train_deepsucc import rf_train
from correlation_test import spearman
from delete_columns_from_numpy import delete_columns_from_numpy_array
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold

def process_as_deepsucc(dataset, context_window):
    number_of_features = (int(dataset.shape[1]) - 1) // (2 * context_window + 1)
    number_of_features -= 1 #ignoring the lysine residue
    output = np.zeros((dataset.shape[0], 2*context_window, number_of_features), dtype=np.float)
    label = dataset[:,-1]
    print(output.shape,dataset.shape)

    for i in range(dataset.shape[0]):
        for k in range(number_of_features):
            for j in range(2*context_window + 1):
                if j == context_window:
                    continue
                output[i][j if j < context_window else j-1][k] = dataset[i][k*(2*context_window + 1) + j]
    return output,label

n_splits = 6

def fitness_deepsuc(individual,proteins, fastas, p_samples, n_samples, positives, negatives, isCompact = False, isAvg = True,num=30,all=False): # 556 element array with last 3 digits for context window
# def fitness(individual,proteins, val_proteins, fastas, val_fastas, train_p_samples, train_n_samples, test_p_samples, test_n_samples, positives, negatives, val_positives, val_negatives,isCompact = False,isAvg = True,num=30,all=False): # 556 element array with last 3 digits for context window
    context_window = 15 if isCompact else individual[-1] + individual[-2]*2 + individual[-3]*4 + 8 # 000 => 8 , 111 -> 15
    bits_for_context = 0 if isCompact else 3

    indices = []

    for i in range(len(individual)-bits_for_context):
        if individual[i]:
            indices.append(i)
    
    dataset = load_data(indices, context_window,proteins, fastas, p_samples, n_samples, positives, negatives, isCompact, isAvg)
    preds = np.zeros((dataset.shape[0],),dtype=np.float)
    # labels = np.zeros((dataset.shape[0],),dtype=np.float)
    # for i in range(21767):
    #     labels[i] = 1
    # print(dataset.shape)
    kf = StratifiedKFold(n_splits=n_splits)
    # np.random.shuffle(dataset)
    SN = SP = ACC = MCC = 0
    for train, test in kf.split(dataset, dataset[:,-1]):
        # print(dataset[train].shape)
        # print(dataset[test].shape)
        preds[test], sn, sp, acc, mcc = rf_train(dataset[train],dataset[test],num,all)
        SN += sn
        SP += sp
        ACC += acc
        MCC += mcc
    SN /= n_splits
    SP /= n_splits
    ACC /= n_splits
    MCC /= n_splits

    # c_0 = c_1 = 0

    # for i in range(21767):
    #     if preds[i] >= 0.5:
    #         c_1 += 1
    # for i in range(21767,43534,1):
    #     if preds[i] < 0.5:
    #         c_0 += 1
    # print(c_0,c_1)
    # np.savetxt('../DeepSucc/15_10_0.7_0.9_'+str(n_splits)+'_folds.txt',preds)
    # dataset, val_dataset = load_data(indices, context_window,proteins, val_proteins, fastas, val_fastas, train_p_samples, train_n_samples, test_p_samples, test_n_samples, positives, negatives, val_positives, val_negatives, isCompact, isAvg)
    # print(np.sum(dataset[:,-1]), np.sum(val_dataset[:,-1]),val_dataset.shape)
    # deleted_features = spearman(dataset, 0.8)
    # dataset = delete_columns_from_numpy_array(dataset,deleted_features)
    # val_dataset = delete_columns_from_numpy_array(val_dataset,deleted_features)
    # with open('15_0.7_0.9','wb') as f:
    #     pickle.dump(dataset[:,:-1],f)

    # with open('val_rf','wb') as f:
    #     pickle.dump(val_dataset[:,:-1],f)

    # inputs_dl_train = np.loadtxt('dl_output_train.txt')
    # inputs_dl_val = np.loadtxt('dl_output_val.txt')

    # dataset = np.concatenate((inputs_dl_train,dataset),axis = 1)
    # val_dataset = np.concatenate((inputs_dl_val,val_dataset),axis = 1)

    # return rf_train(dataset,val_dataset,num,all)
    return (SN,SP,ACC,MCC)