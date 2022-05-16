from load_data import load_data
from rf_train_ensemble import rf_train

def fitness(individual,proteins, val_proteins, fastas, val_fastas, p_samples, n_samples, positives, negatives, val_positives, val_negatives,isCompact = False, isAvg = True,num=30,all=False): # 556 element array with last 3 digits for context window
    context_window = 15 if isCompact else individual[-1] + individual[-2]*2 + individual[-3]*4 + 8 # 000 => 8 , 111 -> 15
    bits_for_context = 0 if isCompact else 3

    indices = []

    for i in range(len(individual)-bits_for_context):
        if individual[i]:
            indices.append(i)
    
    dataset, val_dataset = load_data(indices, context_window,proteins, val_proteins, fastas, val_fastas, p_samples, n_samples, positives, negatives, val_positives, val_negatives, isCompact, isAvg)

    return rf_train(dataset,val_dataset,num,all)