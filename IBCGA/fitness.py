from load_data import load_data
# from load_data import load_data_deepsucc as load_data
# from load_data import load_data_gpsuc as load_data
from rf_train import rf_train
from correlation_test import spearman
from delete_columns_from_numpy import delete_columns_from_numpy_array
import pickle
import numpy as np
from sklearn.model_selection import KFold

def fitness(individual,proteins, val_proteins, fastas, val_fastas, p_samples, n_samples, positives, negatives, val_positives, val_negatives,isCompact = False,isAvg = True,num=30,all=False): # 556 element array with last 3 digits for context window
# def fitness(individual,proteins, val_proteins, fastas, val_fastas, train_p_samples, train_n_samples, test_p_samples, test_n_samples, positives, negatives, val_positives, val_negatives,isCompact = False,isAvg = True,num=30,all=False): # 556 element array with last 3 digits for context window
    context_window = 10 if isCompact else individual[-1] + individual[-2]*2 + individual[-3]*4 + 8 # 000 => 8 , 111 -> 15
    bits_for_context = 0 if isCompact else 3

    indices = []

    for i in range(len(individual)-bits_for_context):
        if individual[i]:
            indices.append(i)
    
    dataset, val_dataset = load_data(indices, context_window,proteins, val_proteins, fastas, val_fastas, p_samples, n_samples, positives, negatives, val_positives, val_negatives, isCompact, isAvg)
    # train_rf = dataset[:,:-1]
    # val_rf = val_dataset[:,:-1]

    # train_label = dataset[:,-1]
    # val_label = val_dataset[:,-1]

    # train_rf = np.reshape(train_rf,(train_rf.shape[0],-1,20))
    # val_rf = np.reshape(val_rf,(val_rf.shape[0],-1,20))
    # dataset, val_dataset = load_data(indices, context_window,proteins, val_proteins, fastas, val_fastas, train_p_samples, train_n_samples, test_p_samples, test_n_samples, positives, negatives, val_positives, val_negatives, isCompact, isAvg)
    # print(np.sum(dataset[:,-1]), np.sum(val_dataset[:,-1]),val_dataset.shape)
    # deleted_features = spearman(dataset, 0.8)
    # dataset = delete_columns_from_numpy_array(dataset,deleted_features)
    # val_dataset = delete_columns_from_numpy_array(val_dataset,deleted_features)
    # with open('train_rf','wb') as f:
    #     pickle.dump(train_rf,f)
    #     pickle.dump(train_label,f)

    # with open('val_rf','wb') as f:
    #     pickle.dump(val_rf,f)
    #     pickle.dump(val_label,f)

    # inputs_dl_train = np.loadtxt('dl_output_train.txt')
    # inputs_dl_val = np.loadtxt('dl_output_val.txt')

    # dataset = np.concatenate((inputs_dl_train,dataset),axis = 1)
    # val_dataset = np.concatenate((inputs_dl_val,val_dataset),axis = 1)

    return rf_train(dataset,val_dataset,num,all)