import numpy as np
import pickle

with open('../Biophysico properties/aaindex1_extracted','rb') as f:
    values = pickle.load(f)

data = open("../Datasets/D2/train_proteins.txt",'r')

data = data.readlines()
proteins = []

for line in data:
    line = line[:-1]
    proteins.append(line)
print(len(proteins))

fastas = {}
data = open("../Datasets/D2/train_fasta.txt",'r')

data = data.readlines()

for line in data:
    line = line[:-1]
    if line[0]=='>':
        prot = line[1:]
    else:
        fastas[prot] = line
# print(fastas['P55096'])

positives = 0
negatives = 0

n = 0

p_samples = {}

data = open("../Datasets/D2/pos_train.csv")

data = data.readlines()

for row in data[1:]:
    row = row[:-1]
    row = row.split(',')

    pos = int(row[2])
    row[1] = row[1].split('|')[1]
    
    if row[1] not in p_samples:
        p_samples[row[1]] = []
    
    if pos not in p_samples[row[1]]:
        p_samples[row[1]].append(pos)
        positives += 1

n_samples = {}
data = open("../Datasets/D2/neg_train.csv")

data = data.readlines()

for row in data[1:]:
    row = row[:-1]
    row = row.split(',')

    pos = int(row[2])
    row[1] = row[1].split('|')[1]

    if row[1] not in n_samples:
        n_samples[row[1]] = []
    
    if pos not in n_samples[row[1]]:
        n_samples[row[1]].append(pos)
        negatives += 1

print(positives,negatives)

# indices = [33,34,35,67,69,73,76,80,85,92,93,94,95,112,116,117,126,127,128,129,130,145,150,152,197,
#             208,212,319,320,333,355,391,400,446,447,487,489,493,517,518,521,522,525,526]
indices = list(range(566))

for context_window in range(16,17):
    for i in indices:
        # file = open('../GPSucPaperCompare/Data/val_re.txt','w')
        dataset = np.zeros((1 * positives + negatives, (2 * context_window) + 1), dtype=float)
        # val_dataset = np.zeros((1 * val_positives + val_negatives, (2 * context_window) + 1), dtype=float)

        row = col = 0
        # val_row = val_col = 0
        for prot in p_samples:
            for p_indice in p_samples[prot]:
                if prot in proteins:
                    row += 1
                    col = 0
                    # if p_indice - context_window < 1:
                    #     col += (context_window - p_indice + 1)
                    # for j in range(max(p_indice - context_window, 1), min(p_indice + context_window + 1,len(fastas[prot])+1), 1):
                    for j in range(p_indice-context_window,p_indice+context_window+1,1):
                        if j == p_indice:
                            continue
                        actual_index = j - 1
                        if actual_index < 0:
                            actual_index += 2*((p_indice-1) - actual_index)
                        elif actual_index >= len(fastas[prot]):
                            actual_index -= -2*((p_indice - 1) - actual_index)
                        col += 1
                        dataset[row-1][col-1] = values[i][fastas[prot][actual_index]]
                    dataset[row-1][-1] = 1
                else:
                    print('check')
                    
        for prot in n_samples:
            for n_indice in n_samples[prot]:
                if prot in proteins:
                    row += 1
                    col = 0
                    # if n_indice - context_window < 1:
                    #     col += (context_window - n_indice + 1)
                    # for j in range(max(n_indice - context_window, 1), min(n_indice + context_window + 1,len(fastas[prot])+1), 1):
                    for j in range(n_indice-context_window,n_indice+context_window+1,1):
                        if j == n_indice:
                            continue
                        actual_index = j - 1
                        if actual_index < 0:
                            actual_index += 2*((n_indice - 1) - actual_index)
                        elif actual_index >= len(fastas[prot]):
                            actual_index -= -2*((n_indice - 1) - actual_index)
                        col += 1
                        dataset[row-1][col-1] = values[i][fastas[prot][actual_index]]
                    dataset[row-1][-1] = 0
                else:
                    print('check')
        index = [i for i in range(dataset.shape[0])]
        np.random.seed(10)
        np.random.shuffle(index)
        dataset = dataset[index]
        val_dataset = dataset[-3000:]
        dataset = dataset[:-3000]
        np.savetxt('..//Datasets//D2//Bio-physico//Train//'+str(context_window)+'_'+str(i)+'.txt', dataset, fmt='%f')
        np.savetxt('..//Datasets//D2//Bio-physico//Val//'+str(context_window)+'_'+str(i)+'.txt', val_dataset, fmt='%f')
        # break