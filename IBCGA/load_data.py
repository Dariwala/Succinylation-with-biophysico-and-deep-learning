import numpy as np
import pickle
from map import good_features_16 as good_features_16

# paper_name = 'GPSucPaperCompare' #LSTMCNNSuccPaperCompare, GPSucPaperCompare

def load_aa_index():
    with open('../Bio-physico analysis/aaindex1_extracted_non_empty','rb') as f:
        values = pickle.load(f)

    return values

def load_essentials():
    paper_name = 'LSTMCNNSuccPaperCompare'
    data = open("../"+paper_name+"/Data/train_proteins.txt",'r')

    data = data.readlines()
    proteins = []

    for line in data:
        line = line[:-1]
        proteins.append(line)
    # print(len(proteins))

    data = open("../"+paper_name+"/Data/val_proteins.txt",'r')

    data = data.readlines()
    val_proteins = []

    for line in data:
        line = line[:-1]
        val_proteins.append(line)
    # print(len(val_proteins))

    fastas = {}
    data = open("../"+paper_name+"/Data/train_fasta.txt",'r')

    data = data.readlines()

    for line in data:
        line = line[:-1]
        if line[0]=='>':
            prot = line[1:]
        else:
            fastas[prot] = line
    # print(fastas['P55096'])

    val_fastas = {}
    data = open("../"+paper_name+"/Data/val_fasta.txt",'r')

    data = data.readlines()

    for line in data:
        line = line[:-1]
        if line[0]=='>':
            prot = line[1:]
        else:
            val_fastas[prot] = line
    # print(fastas['P55096'])

    positives = 0
    negatives = 0
    val_positives = 0
    val_negatives = 0
    n = 0

    p_samples = {}

    data = open("../"+paper_name+"/Data/pos_train.csv")

    data = data.readlines()

    for row in data[1:]:
        row = row[:-1]
        row = row.split(',')

        pos = int(row[2])
        
        try:
            assert fastas[row[1]][pos-1] == 'K'
        except KeyError:
            assert val_fastas[row[1]][pos-1] == 'K'


        if row[1] not in p_samples:
            p_samples[row[1]] = []
        
        if pos not in p_samples[row[1]]:
            p_samples[row[1]].append(pos)
            if row[1] in proteins:
                positives += 1
            elif row[1] in val_proteins:
                val_positives += 1
            else:
                print('mathay haat')

    n_samples = {}
    data = open("../"+paper_name+"/Data/neg_train.csv")

    data = data.readlines()

    for row in data[1:]:
        row = row[:-1]
        row = row.split(',')

        pos = int(row[2])
        # assert row[4][pos-1] == 'K'

        if row[1] not in n_samples:
            n_samples[row[1]] = []
        
        if pos not in n_samples[row[1]]:
            n_samples[row[1]].append(pos)
            if row[1] in proteins:
                negatives += 1
            elif row[1] in val_proteins:
                val_negatives += 1
            else:
                print('mathay haat')

    return proteins, val_proteins, fastas, val_fastas, p_samples, n_samples, positives, negatives, val_positives, val_negatives

def load_essentials_deepsucc():
    data = open("../DeepSucc/General.txt",'r')

    data = data.readlines()
    proteins = []
    fastas = {}

    for line in data:
        line = line[:-1]
        if line[0] == '>':
            proteins.append(line[1:])
        else:
            fastas[proteins[-1]] = line

    positives = 0
    negatives = 0

    p_samples = {}
    n_samples = {}

    data = open("../DeepSucc/General_Train.csv")

    data = data.readlines()

    for row in data:
        row = row[:-1]
        row = row.split(',')

        pos = int(row[2])
        
        assert fastas[row[1]][pos-1] == 'K'
        
        if row[3] == '1':
            if row[1] not in p_samples:
                p_samples[row[1]] = []
            if pos not in p_samples[row[1]]:
                p_samples[row[1]].append(pos)
                if row[1] in proteins:
                    positives += 1
                else:
                    print('mathay haat')
        else:
            if row[1] not in n_samples:
                n_samples[row[1]] = []
            if pos not in n_samples[row[1]]:
                n_samples[row[1]].append(pos)
                if row[1] in proteins:
                    negatives += 1
                else:
                    print('mathay haat')

    return proteins, fastas, p_samples, n_samples, positives, negatives

def load_essentials_test():
    paper_name = 'LSTMCNNSuccPaperCompare'
    data = open("../"+paper_name+"/Data/train_proteins.txt",'r')

    data = data.readlines()
    proteins = []

    for line in data:
        line = line[:-1]
        proteins.append(line)
    
    data = open("../"+paper_name+"/Data/val_proteins.txt",'r')

    data = data.readlines()
    # proteins = []

    for line in data:
        line = line[:-1]
        proteins.append(line)

    data = open("../"+paper_name+"/Data/test_proteins.txt",'r')

    data = data.readlines()
    val_proteins = []

    for line in data:
        line = line[:-1]
        val_proteins.append(line)
    # print(len(val_proteins))

    fastas = {}
    data = open("../"+paper_name+"/Data/train_fasta.txt",'r')

    data = data.readlines()

    for line in data:
        line = line[:-1]
        if line[0]=='>':
            prot = line[1:]
        else:
            fastas[prot] = line
    
    data = open("../"+paper_name+"/Data/val_fasta.txt",'r')

    data = data.readlines()

    for line in data:
        line = line[:-1]
        if line[0]=='>':
            prot = line[1:]
        else:
            fastas[prot] = line
    # print(fastas['P55096'])

    val_fastas = {}
    data = open("../"+paper_name+"/Data/test_fasta.txt",'r')

    data = data.readlines()

    for line in data:
        line = line[:-1]
        if line[0]=='>':
            prot = line[1:]
        else:
            val_fastas[prot] = line
    # print(fastas['P55096'])

    positives = 0
    negatives = 0
    val_positives = 0
    val_negatives = 0
    n = 0

    p_samples = {}

    data = open("../"+paper_name+"/Data/pos_train.csv")

    data = data.readlines()

    for row in data[1:]:
        row = row[:-1] if paper_name=='LSTMCNNSuccPaperCompare' else row
        row = row.split(',')

        pos = int(row[2])
        
        if paper_name == 'GPSucPaperCompare':
            row[1] = row[1].split('|')[1]
        # print(row[1],fastas[row[1]][pos-2:pos+1])
        # print(pos)
        # assert fastas[row[1]][pos-1] == 'K'
        


        if row[1] not in p_samples:
            p_samples[row[1]] = []
        
        if pos not in p_samples[row[1]]:
            p_samples[row[1]].append(pos)
            # if row[1] in proteins:
            positives += 1
            # elif row[1] in val_proteins:
                # val_positives += 1
            # else:
                # print('mathay haat')

    n_samples = {}
    data = open("../"+paper_name+"/Data/neg_train.csv")

    data = data.readlines()

    for row in data[1:]:
        row = row[:-1] if paper_name=='LSTMCNNSuccPaperCompare' else row
        row = row.split(',')

        pos = int(row[2])
        # assert row[4][pos-1] == 'K'

        if paper_name == 'GPSucPaperCompare':
            row[1] = row[1].split('|')[1]

        if row[1] not in n_samples:
            n_samples[row[1]] = []
        
        if pos not in n_samples[row[1]]:
            n_samples[row[1]].append(pos)
            # if row[1] in proteins:
            negatives += 1
            # elif row[1] in val_proteins:
                # val_negatives += 1
            # else:
                # print('mathay haat')
    data = open("../"+paper_name+"/Data/pos_test.csv")

    data = data.readlines()

    for row in data[1:]:
        row = row[:-1] if paper_name=='LSTMCNNSuccPaperCompare' else row
        row = row.split(',')

        pos = int(row[2])
        
        if paper_name == 'GPSucPaperCompare':
            row[1] = row[1].split('|')[1]
        
        # assert val_fastas[row[1]][pos-1] == 'K'
        


        if row[1] not in p_samples:
            p_samples[row[1]] = []
        
        if pos not in p_samples[row[1]]:
            p_samples[row[1]].append(pos)
            # if row[1] in proteins:
            val_positives += 1
            # elif row[1] in val_proteins:
                # val_positives += 1
            # else:
                # print('mathay haat')

    # n_samples = {}
    data = open("../"+paper_name+"/Data/neg_test.csv")

    data = data.readlines()

    for row in data[1:]:
        row = row[:-1] if paper_name=='LSTMCNNSuccPaperCompare' else row
        row = row.split(',')

        pos = int(row[2])

        if paper_name == 'GPSucPaperCompare':
            row[1] = row[1].split('|')[1]
        # assert row[4][pos-1] == 'K'

        if row[1] not in n_samples:
            n_samples[row[1]] = []
        
        if pos not in n_samples[row[1]]:
            n_samples[row[1]].append(pos)
            # if row[1] in proteins:
            val_negatives += 1
            # elif row[1] in val_proteins:
                # val_negatives += 1
            # else:
                # print('mathay haat')

    return proteins, val_proteins, fastas, val_fastas, p_samples, n_samples, positives, negatives, val_positives, val_negatives

def load_essentials_gpsuc():
    paper_name = 'GPSucPaperCompare'
    data = open("../"+paper_name+"/Data/train_proteins.txt",'r')

    data = data.readlines()
    proteins = []

    for line in data:
        line = line[:-1]
        proteins.append(line)
    
    data = open("../"+paper_name+"/Data/test_proteins.txt",'r')

    data = data.readlines()
    val_proteins = []

    for line in data:
        line = line[:-1]
        val_proteins.append(line)
    # print(len(val_proteins))

    fastas = {}
    data = open("../"+paper_name+"/Data/train_fasta.txt",'r')

    data = data.readlines()

    for line in data:
        line = line[:-1]
        if line[0]=='>':
            prot = line[1:]
        else:
            fastas[prot] = line

    val_fastas = {}
    data = open("../"+paper_name+"/Data/test_fasta.txt",'r')

    data = data.readlines()

    for line in data:
        line = line[:-1]
        if line[0]=='>':
            prot = line[1:]
        else:
            val_fastas[prot] = line
    # print(fastas['P55096'])

    positives = 0
    negatives = 0
    val_positives = 0
    val_negatives = 0

    train_p_samples = {}

    data = open("../"+paper_name+"/Data/pos_train.csv")

    data = data.readlines()

    for row in data[1:]:
        row = row[:-1] if paper_name=='LSTMCNNSuccPaperCompare' else row
        row = row.split(',')

        pos = int(row[2])
        
        if paper_name == 'GPSucPaperCompare':
            row[1] = row[1].split('|')[1]
        # print(row[1],fastas[row[1]][pos-2:pos+1])
        # print(pos)
        # assert fastas[row[1]][pos-1] == 'K'
        


        if row[1] not in train_p_samples:
            train_p_samples[row[1]] = []
        
        if pos not in train_p_samples[row[1]]:
            train_p_samples[row[1]].append(pos)
            # if row[1] in proteins:
            positives += 1
            # elif row[1] in val_proteins:
                # val_positives += 1
            # else:
                # print('mathay haat')

    train_n_samples = {}
    data = open("../"+paper_name+"/Data/neg_train.csv")

    data = data.readlines()

    for row in data[1:]:
        row = row[:-1] if paper_name=='LSTMCNNSuccPaperCompare' else row
        row = row.split(',')

        pos = int(row[2])
        # assert row[4][pos-1] == 'K'

        if paper_name == 'GPSucPaperCompare':
            row[1] = row[1].split('|')[1]

        if row[1] not in train_n_samples:
            train_n_samples[row[1]] = []
        
        if pos not in train_n_samples[row[1]]:
            train_n_samples[row[1]].append(pos)
            # if row[1] in proteins:
            negatives += 1
            # elif row[1] in val_proteins:
                # val_negatives += 1
            # else:
                # print('mathay haat')
    data = open("../"+paper_name+"/Data/pos_test.csv")

    data = data.readlines()

    test_p_samples = {}

    for row in data[1:]:
        row = row[:-1] if paper_name=='LSTMCNNSuccPaperCompare' else row
        row = row.split(',')

        pos = int(row[2])
        
        if paper_name == 'GPSucPaperCompare':
            row[1] = row[1].split('|')[1]
        
        # assert val_fastas[row[1]][pos-1] == 'K'

        if row[1] not in test_p_samples:
            test_p_samples[row[1]] = []
        
        if pos not in test_p_samples[row[1]]:
            test_p_samples[row[1]].append(pos)
            # if row[1] in proteins:
            val_positives += 1
            # elif row[1] in val_proteins:
                # val_positives += 1
            # else:
                # print('mathay haat')

    # n_samples = {}
    data = open("../"+paper_name+"/Data/neg_test.csv")

    data = data.readlines()

    test_n_samples = {}

    for row in data[1:]:
        row = row[:-1] if paper_name=='LSTMCNNSuccPaperCompare' else row
        row = row.split(',')

        pos = int(row[2])

        if paper_name == 'GPSucPaperCompare':
            row[1] = row[1].split('|')[1]
        # assert row[4][pos-1] == 'K'
        if row[1] not in test_n_samples:
            test_n_samples[row[1]] = []
        
        if pos not in test_n_samples[row[1]]:
            test_n_samples[row[1]].append(pos)
            # if row[1] in proteins:
            val_negatives += 1
            # elif row[1] in val_proteins:
                # val_negatives += 1
            # else:
                # print('mathay haat')

    return proteins, val_proteins, fastas, val_fastas, train_p_samples, train_n_samples, test_p_samples, test_n_samples, positives, negatives, val_positives, val_negatives

def load_essentials_test_only_train():
    paper_name = 'LSTMCNNSuccPaperCompare'
    data = open("../"+paper_name+"/Data/train_proteins.txt",'r')

    data = data.readlines()
    proteins = []

    for line in data:
        line = line[:-1]
        proteins.append(line)
    
    # data = open("../LSTMCNNSuccPaperCompare/Data/val_proteins.txt",'r')

    # data = data.readlines()
    # # proteins = []

    # for line in data:
    #     line = line[:-1]
    #     proteins.append(line)

    data = open("../"+paper_name+"/Data/test_proteins.txt",'r')

    data = data.readlines()
    val_proteins = []

    for line in data:
        line = line[:-1]
        val_proteins.append(line)
    # print(len(val_proteins))

    fastas = {}
    data = open("../"+paper_name+"/Data/train_fasta.txt",'r')

    data = data.readlines()

    for line in data:
        line = line[:-1]
        if line[0]=='>':
            prot = line[1:]
        else:
            fastas[prot] = line
    
    # data = open("../LSTMCNNSuccPaperCompare/Data/val_fasta.txt",'r')

    # data = data.readlines()

    # for line in data:
    #     line = line[:-1]
    #     if line[0]=='>':
    #         prot = line[1:]
    #     else:
    #         fastas[prot] = line
    # print(fastas['P55096'])

    val_fastas = {}
    data = open("../"+paper_name+"/Data/test_fasta.txt",'r')

    data = data.readlines()

    for line in data:
        line = line[:-1]
        if line[0]=='>':
            prot = line[1:]
        else:
            val_fastas[prot] = line
    # print(fastas['P55096'])

    positives = 0
    negatives = 0
    val_positives = 0
    val_negatives = 0
    n = 0

    p_samples = {}

    data = open("../"+paper_name+"/Data/pos_train.csv")

    data = data.readlines()

    for row in data[1:]:
        row = row[:-1]
        row = row.split(',')

        pos = int(row[2])

        if row[1] not in proteins:
            continue
        
        
        assert fastas[row[1]][pos-1] == 'K'
        


        if row[1] not in p_samples:
            p_samples[row[1]] = []
        
        if pos not in p_samples[row[1]]:
            p_samples[row[1]].append(pos)
            # if row[1] in proteins:
            positives += 1
            # elif row[1] in val_proteins:
                # val_positives += 1
            # else:
                # print('mathay haat')

    n_samples = {}
    data = open("../"+paper_name+"/Data/neg_train.csv")

    data = data.readlines()

    for row in data[1:]:
        row = row[:-1]
        row = row.split(',')

        pos = int(row[2])
        if row[1] not in proteins:
            continue
        # assert row[4][pos-1] == 'K'

        if row[1] not in n_samples:
            n_samples[row[1]] = []
        
        if pos not in n_samples[row[1]]:
            n_samples[row[1]].append(pos)
            # if row[1] in proteins:
            negatives += 1
            # elif row[1] in val_proteins:
                # val_negatives += 1
            # else:
                # print('mathay haat')
    data = open("../"+paper_name+"/Data/pos_test.csv")

    data = data.readlines()

    for row in data[1:]:
        row = row[:-1]
        row = row.split(',')

        pos = int(row[2])
        
        
        assert val_fastas[row[1]][pos-1] == 'K'
        


        if row[1] not in p_samples:
            p_samples[row[1]] = []
        
        if pos not in p_samples[row[1]]:
            p_samples[row[1]].append(pos)
            # if row[1] in proteins:
            val_positives += 1
            # elif row[1] in val_proteins:
                # val_positives += 1
            # else:
                # print('mathay haat')

    # n_samples = {}
    data = open("../"+paper_name+"/Data/neg_test.csv")

    data = data.readlines()

    for row in data[1:]:
        row = row[:-1]
        row = row.split(',')

        pos = int(row[2])
        # assert row[4][pos-1] == 'K'

        if row[1] not in n_samples:
            n_samples[row[1]] = []
        
        if pos not in n_samples[row[1]]:
            n_samples[row[1]].append(pos)
            # if row[1] in proteins:
            val_negatives += 1
            # elif row[1] in val_proteins:
                # val_negatives += 1
            # else:
                # print('mathay haat')

    return proteins, val_proteins, fastas, val_fastas, p_samples, n_samples, positives, negatives, val_positives, val_negatives

def create_data(proteins, val_proteins, fastas, val_fastas, p_samples, n_samples, positive, negative, val_positive, val_negative, context_window, indices, values):
    dataset = np.zeros((positive+negative,len(indices)+1),dtype=float)
    val_dataset = np.zeros((val_positive+val_negative,len(indices)+1),dtype=float)
    r = c = 0
    vr = vc = 0
    for prot in p_samples:
        for pos in p_samples[prot]:
            pos -= 1
            if prot in proteins:
                assert c == len(indices) + 1 or c == 0
                c = 0
                for i in indices:
                    sum = 0.0
                    for j in range(pos - context_window, pos+context_window+1):
                        try:
                            sum += values[i][fastas[prot][j]]
                        except IndexError:
                            continue
                    dataset[r][c] = sum / (2 * context_window + 1)
                    c += 1
                dataset[r][c] = 1
                c += 1
                r += 1
            elif prot in val_proteins:
                assert vc == len(indices) + 1 or vc == 0
                vc = 0
                for i in indices:
                    sum = 0.0
                    for j in range(pos - context_window, pos+context_window+1):
                        try:
                            sum += values[i][val_fastas[prot][j]]
                        except IndexError:
                            continue
                    val_dataset[vr][vc] = sum / (2 * context_window + 1)
                    vc += 1
                val_dataset[vr][vc] = 1
                vc += 1
                vr += 1
            else:
                print('hayre')
    assert r == positive and vr == val_positive and c == len(indices)+1 and vc == len(indices)+1

    for prot in n_samples:
        for pos in n_samples[prot]:
            pos -= 1
            if prot in proteins:
                assert c == len(indices) + 1 or c == 0
                c = 0
                for i in indices:
                    sum = 0.0
                    for j in range(pos - context_window, pos+context_window+1):
                        try:
                            sum += values[i][fastas[prot][j]]
                        except IndexError:
                            continue
                    dataset[r][c] = sum / (2 * context_window + 1)
                    c += 1
                dataset[r][c] = 0
                c += 1
                r += 1
            elif prot in val_proteins:
                assert vc == len(indices) + 1 or vc == 0
                vc = 0
                for i in indices:
                    sum = 0.0
                    for j in range(pos - context_window, pos+context_window+1):
                        try:
                            sum += values[i][val_fastas[prot][j]]
                        except IndexError:
                            continue
                    val_dataset[vr][vc] = sum / (2 * context_window + 1)
                    vc += 1
                val_dataset[vr][vc] = 0
                vc += 1
                vr += 1
            else:
                print('hayre')
    assert r == positive + negative and vr == val_positive + val_negative    
    return dataset, val_dataset

def create_data_2(proteins, val_proteins, fastas, val_fastas, p_samples, n_samples, positive, negative, val_positive, val_negative, context_window, indices, values, isCompact):
    dataset = np.zeros((positive+negative, (2*context_window) * len(indices)+1),dtype=float)
    val_dataset = np.zeros((val_positive+val_negative, (2*context_window) * len(indices)+1),dtype=float)
    r = c = 0
    vr = vc = 0
    for prot in p_samples:
        for pos in p_samples[prot]:
            pos -= 1
            if prot in proteins:
                assert c == (2*context_window) * len(indices)+1 or c == 0
                c = 0
                for i in indices:
                    for j in range(pos - context_window, pos+context_window+1):
                        if j == pos:
                            continue
                        try:
                            actual_index = j
                            if actual_index < 0:
                                actual_index += 2*(pos - actual_index)
                            elif actual_index >= len(fastas[prot]):
                                actual_index -= -2*(pos - actual_index)
                            dataset[r][c] = values[good_features_16[i]][fastas[prot][actual_index]] if isCompact else values[i][fastas[prot][j]]
                            c += 1
                        except IndexError:
                            c += 1
                            print('note korlam')
                            continue
                    # dataset[r][c] = sum / (2 * context_window + 1)
                    # c += 1
                dataset[r][c] = 1
                c += 1
                r += 1
            elif prot in val_proteins:
                assert vc == (2*context_window) * len(indices)+1 or vc == 0
                vc = 0
                for i in indices:
                    for j in range(pos - context_window, pos+context_window+1):
                        if j == pos:
                            continue
                        try:
                            actual_index = j
                            if actual_index < 0:
                                actual_index += 2*(pos - actual_index)
                            elif actual_index >= len(val_fastas[prot]):
                                actual_index -= -2*(pos - actual_index)
                            val_dataset[vr][vc] = values[good_features_16[i]][val_fastas[prot][actual_index]] if isCompact else values[i][val_fastas[prot][j]]
                            vc += 1
                        except IndexError:
                            print('note korlam')
                            vc += 1
                            continue
                    # val_dataset[vr][vc] = sum / (2 * context_window + 1)
                    # vc += 1
                val_dataset[vr][vc] = 1
                vc += 1
                vr += 1
            else:
                print('hayre')
    assert r == positive and vr == val_positive and c == (2*context_window)* len(indices)+1 and vc == (2*context_window)* len(indices)+1

    for prot in n_samples:
        for pos in n_samples[prot]:
            pos -= 1
            if prot in proteins:
                assert c == (2*context_window) * len(indices)+1 or c == 0
                c = 0
                for i in indices:
                    # sum = 0.0
                    for j in range(pos - context_window, pos+context_window+1):
                        if j == pos:
                            continue
                        try:
                            actual_index = j
                            if actual_index < 0:
                                actual_index += 2*(pos - actual_index)
                            elif actual_index >= len(fastas[prot]):
                                actual_index -= -2*(pos - actual_index)
                            dataset[r][c] = values[good_features_16[i]][fastas[prot][actual_index]] if isCompact else values[i][fastas[prot][j]]
                            c += 1
                        except IndexError:
                            print('note korlam')
                            c += 1
                            continue
                    # dataset[r][c] = sum / (2 * context_window + 1)
                    # c += 1
                dataset[r][c] = 0
                c += 1
                r += 1
            elif prot in val_proteins:
                assert vc == (2*context_window) * len(indices)+1 or vc == 0
                vc = 0
                for i in indices:
                    # sum = 0.0
                    for j in range(pos - context_window, pos+context_window+1):
                        if j == pos:
                            continue
                        try:
                            actual_index = j
                            if actual_index < 0:
                                actual_index += 2*(pos - actual_index)
                            elif actual_index >= len(val_fastas[prot]):
                                actual_index -= -2*(pos - actual_index)
                            val_dataset[vr][vc] = values[good_features_16[i]][val_fastas[prot][actual_index]] if isCompact else values[i][val_fastas[prot][j]]
                            vc += 1
                        except IndexError:
                            vc += 1
                            continue
                    # val_dataset[vr][vc] = sum / (2 * context_window + 1)
                    # vc += 1
                val_dataset[vr][vc] = 0
                vc += 1
                vr += 1
            else:
                print('hayre')
    assert r == positive + negative and vr == val_positive + val_negative    
    return dataset, val_dataset

def create_data_deepsucc(proteins, fastas, p_samples, n_samples, positive, negative, context_window, indices, values, isCompact):
    dataset = np.zeros((positive+negative, (2*context_window) * len(indices)+1),dtype=float)
    r = c = 0
    for prot in p_samples:
        for pos in p_samples[prot]:
            pos -= 1
            # if prot in proteins:
            assert c == (2*context_window) * len(indices)+1 or c == 0
            c = 0
            for i in indices:
                for j in range(pos - context_window, pos+context_window+1):
                    if j == pos:
                        continue
                    try:
                        actual_index = j
                        if actual_index < 0:
                            actual_index += 2*(pos - actual_index)
                        elif actual_index >= len(fastas[prot]):
                            actual_index -= -2*(pos - actual_index)
                        dataset[r][c] = values[good_features_16[i]][fastas[prot][actual_index]] if isCompact else values[i][fastas[prot][j]]
                        c += 1
                    except IndexError:
                        c += 1
                        print('note korlam')
                        continue
                # dataset[r][c] = sum / (2 * context_window + 1)
                # c += 1
            dataset[r][c] = 1
            c += 1
            r += 1
    assert r == positive and c == (2*context_window)* len(indices)+1

    for prot in n_samples:
        for pos in n_samples[prot]:
            pos -= 1
            assert c == (2*context_window) * len(indices)+1 or c == 0
            c = 0
            for i in indices:
                # sum = 0.0
                for j in range(pos - context_window, pos+context_window+1):
                    if j == pos:
                        continue
                    try:
                        actual_index = j
                        if actual_index < 0:
                            actual_index += 2*(pos - actual_index)
                        elif actual_index >= len(fastas[prot]):
                            actual_index -= -2*(pos - actual_index)
                        dataset[r][c] = values[good_features_16[i]][fastas[prot][actual_index]] if isCompact else values[i][fastas[prot][j]]
                        c += 1
                    except IndexError:
                        print('note korlam')
                        c += 1
                        continue
                # dataset[r][c] = sum / (2 * context_window + 1)
                # c += 1
            dataset[r][c] = 0
            c += 1
            r += 1
    assert r == positive + negative
    return dataset

def create_data_gpsuc(proteins, val_proteins, fastas, val_fastas, train_p_samples, train_n_samples, test_p_samples, test_n_samples, positive, negative, val_positive, val_negative, context_window, indices, values, isCompact):
    dataset = np.zeros((positive+negative, (2*context_window) * len(indices)+1),dtype=float)
    val_dataset = np.zeros((val_positive+val_negative, (2*context_window) * len(indices)+1),dtype=float)
    r = c = 0
    vr = vc = 0
    for prot in train_p_samples:
        for pos in train_p_samples[prot]:
            pos -= 1
            if prot in proteins:
                assert c == (2*context_window) * len(indices)+1 or c == 0
                c = 0
                for i in indices:
                    for j in range(pos - context_window, pos+context_window+1):
                        if j == pos:
                            continue
                        try:
                            actual_index = j
                            if actual_index < 0:
                                actual_index += 2*(pos - actual_index)
                            elif actual_index >= len(fastas[prot]):
                                actual_index -= -2*(pos - actual_index)
                            dataset[r][c] = values[good_features_16[i]][fastas[prot][actual_index]] if isCompact else values[i][fastas[prot][j]]
                            c += 1
                        except IndexError:
                            c += 1
                            continue
                    # dataset[r][c] = sum / (2 * context_window + 1)
                    # c += 1
                dataset[r][c] = 1
                c += 1
                r += 1
            # elif prot in val_proteins:
            #     assert vc == (2*context_window+1) * len(indices)+1 or vc == 0
            #     vc = 0
            #     for i in indices:
            #         for j in range(pos - context_window, pos+context_window+1):
            #             try:
            #                 val_dataset[vr][vc] = values[good_features_16[i]][val_fastas[prot][j]] if isCompact else values[i][val_fastas[prot][j]]
            #                 vc += 1
            #             except IndexError:
            #                 vc += 1
            #                 continue
            #         # val_dataset[vr][vc] = sum / (2 * context_window + 1)
            #         # vc += 1
            #     val_dataset[vr][vc] = 1
            #     vc += 1
            #     vr += 1
            # else:
            #     print('hayre')
    for prot in test_p_samples:
        for pos in test_p_samples[prot]:
            pos -= 1
            # if prot in proteins:
            #     assert c == (2*context_window+1) * len(indices)+1 or c == 0
            #     c = 0
            #     for i in indices:
            #         for j in range(pos - context_window, pos+context_window+1):
            #             try:
            #                 # sum += values[i][fastas[prot][j]]
            #                 dataset[r][c] = values[good_features_16[i]][fastas[prot][j]] if isCompact else values[i][fastas[prot][j]]
            #                 c += 1
            #             except IndexError:
            #                 c += 1
            #                 continue
            #         # dataset[r][c] = sum / (2 * context_window + 1)
            #         # c += 1
            #     dataset[r][c] = 1
            #     c += 1
            #     r += 1
            if prot in val_proteins:
                assert vc == (2*context_window) * len(indices)+1 or vc == 0
                vc = 0
                for i in indices:
                    for j in range(pos - context_window, pos+context_window+1):
                        if j == pos:
                            continue
                        try:
                            actual_index = j
                            if actual_index < 0:
                                actual_index += 2*(pos - actual_index)
                            elif actual_index >= len(val_fastas[prot]):
                                actual_index -= -2*(pos - actual_index)
                            val_dataset[vr][vc] = values[good_features_16[i]][val_fastas[prot][actual_index]] if isCompact else values[i][val_fastas[prot][j]]
                            vc += 1
                        except IndexError:
                            vc += 1
                            continue
                    # val_dataset[vr][vc] = sum / (2 * context_window + 1)
                    # vc += 1
                val_dataset[vr][vc] = 1
                vc += 1
                vr += 1
            # else:
            #     print('hayre')
    # print(r, positive)    
    assert r == positive
    assert vr == val_positive
    assert c == (2*context_window)* len(indices)+1 
    assert vc == (2*context_window)* len(indices)+1

    for prot in train_n_samples:
        for pos in train_n_samples[prot]:
            pos -= 1
            if prot in proteins:
                assert c == (2*context_window) * len(indices)+1 or c == 0
                c = 0
                for i in indices:
                    # sum = 0.0
                    for j in range(pos - context_window, pos+context_window+1):
                        if j == pos:
                            continue
                        try:
                            actual_index = j
                            if actual_index < 0:
                                actual_index += 2*(pos - actual_index)
                            elif actual_index >= len(fastas[prot]):
                                actual_index -= -2*(pos - actual_index)
                            dataset[r][c] = values[good_features_16[i]][fastas[prot][actual_index]] if isCompact else values[i][fastas[prot][j]]
                            c += 1
                        except IndexError:
                            c += 1
                            continue
                    # dataset[r][c] = sum / (2 * context_window + 1)
                    # c += 1
                dataset[r][c] = 0
                c += 1
                r += 1
            # elif prot in val_proteins:
            #     assert vc == (2*context_window+1) * len(indices)+1 or vc == 0
            #     vc = 0
            #     for i in indices:
            #         # sum = 0.0
            #         for j in range(pos - context_window, pos+context_window+1):
            #             try:
            #                 val_dataset[vr][vc] = values[good_features_16[i]][val_fastas[prot][j]] if isCompact else values[i][val_fastas[prot][j]]
            #                 vc += 1
            #             except IndexError:
            #                 vc += 1
            #                 continue
            #         # val_dataset[vr][vc] = sum / (2 * context_window + 1)
            #         # vc += 1
            #     val_dataset[vr][vc] = 0
            #     vc += 1
            #     vr += 1
            # else:
            #     print('hayre')
    for prot in test_n_samples:
        for pos in test_n_samples[prot]:
            pos -= 1
            # if prot in proteins:
            #     assert c == (2*context_window+1) * len(indices)+1 or c == 0
            #     c = 0
            #     for i in indices:
            #         # sum = 0.0
            #         for j in range(pos - context_window, pos+context_window+1):
            #             try:
            #                 dataset[r][c] = values[good_features_16[i]][fastas[prot][j]] if isCompact else values[i][fastas[prot][j]]
            #                 c += 1
            #             except IndexError:
            #                 c += 1
            #                 continue
            #         # dataset[r][c] = sum / (2 * context_window + 1)
            #         # c += 1
            #     dataset[r][c] = 0
            #     c += 1
            #     r += 1
            if prot in val_proteins:
                assert vc == (2*context_window) * len(indices)+1 or vc == 0
                vc = 0
                for i in indices:
                    # sum = 0.0
                    for j in range(pos - context_window, pos+context_window+1):
                        if j == pos:
                            continue
                        try:
                            actual_index = j
                            if actual_index < 0:
                                actual_index += 2*(pos - actual_index)
                            elif actual_index >= len(val_fastas[prot]):
                                actual_index -= -2*(pos - actual_index)
                            val_dataset[vr][vc] = values[good_features_16[i]][val_fastas[prot][actual_index]] if isCompact else values[i][val_fastas[prot][j]]
                            vc += 1
                        except IndexError:
                            vc += 1
                            continue
                    # val_dataset[vr][vc] = sum / (2 * context_window + 1)
                    # vc += 1
                val_dataset[vr][vc] = 0
                vc += 1
                vr += 1
            # else:
            #     print('hayre')
    assert r == positive + negative and vr == val_positive + val_negative    
    return dataset, val_dataset

def create_data_gpsuc_val(proteins, val_proteins, fastas, val_fastas, train_p_samples, train_n_samples, test_p_samples, test_n_samples, positive, negative, val_positive, val_negative, context_window, indices, values, isCompact):
    dataset = np.zeros((positive+negative, (2*context_window) * len(indices)+1),dtype=float)
    # val_dataset = np.zeros((val_positive+val_negative, (2*context_window) * len(indices)+1),dtype=float)
    r = c = 0
    vr = vc = 0
    for prot in train_p_samples:
        for pos in train_p_samples[prot]:
            pos -= 1
            if prot in proteins:
                assert c == (2*context_window) * len(indices)+1 or c == 0
                c = 0
                for i in indices:
                    for j in range(pos - context_window, pos+context_window+1):
                        if j == pos:
                            continue
                        try:
                            actual_index = j
                            if actual_index < 0:
                                actual_index += 2*(pos - actual_index)
                            elif actual_index >= len(fastas[prot]):
                                actual_index -= -2*(pos - actual_index)
                            dataset[r][c] = values[good_features_16[i]][fastas[prot][actual_index]] if isCompact else values[i][fastas[prot][j]]
                            c += 1
                        except IndexError:
                            c += 1
                            continue
                    # dataset[r][c] = sum / (2 * context_window + 1)
                    # c += 1
                dataset[r][c] = 1
                c += 1
                r += 1
    assert r == positive
    assert c == (2*context_window)* len(indices)+1 

    for prot in train_n_samples:
        for pos in train_n_samples[prot]:
            pos -= 1
            if prot in proteins:
                assert c == (2*context_window) * len(indices)+1 or c == 0
                c = 0
                for i in indices:
                    # sum = 0.0
                    for j in range(pos - context_window, pos+context_window+1):
                        if j == pos:
                            continue
                        try:
                            actual_index = j
                            if actual_index < 0:
                                actual_index += 2*(pos - actual_index)
                            elif actual_index >= len(fastas[prot]):
                                actual_index -= -2*(pos - actual_index)
                            dataset[r][c] = values[good_features_16[i]][fastas[prot][actual_index]] if isCompact else values[i][fastas[prot][j]]
                            c += 1
                        except IndexError:
                            c += 1
                            continue
                    # dataset[r][c] = sum / (2 * context_window + 1)
                    # c += 1
                dataset[r][c] = 0
                c += 1
                r += 1
    assert r == positive + negative
    with open('index','rb') as f:
        index = pickle.load(f)
    dataset = dataset[index]
    val_dataset = dataset[-3000:]
    dataset = dataset[:-3000]
    # import pickle
    with open('val_label_ind','wb') as f:
        pickle.dump(val_dataset[:,-1],f)
    return dataset, val_dataset

def load_data(indices,context_window,proteins, val_proteins, fastas, val_fastas, p_samples, n_samples, positives, negatives, val_positives, val_negatives, isCompact = False, isAvg = True):
    values = load_aa_index()
    # print(len(values))
    if isAvg:
        dataset, val_dataset = create_data(proteins, val_proteins, fastas, val_fastas, p_samples, n_samples, positives, negatives, val_positives, val_negatives, context_window, indices, values)
    else:
        dataset, val_dataset = create_data_2(proteins, val_proteins, fastas, val_fastas, p_samples, n_samples, positives, negatives, val_positives, val_negatives, context_window, indices, values, isCompact)

    return dataset, val_dataset


def load_data_gpsuc(indices,context_window,proteins, val_proteins, fastas, val_fastas, train_p_samples, train_n_samples, test_p_samples, test_n_samples, positives, negatives, val_positives, val_negatives, isCompact = False, isAvg = True):
    values = load_aa_index()
    
    dataset, val_dataset = create_data_gpsuc(proteins, val_proteins, fastas, val_fastas, train_p_samples, train_n_samples, test_p_samples, test_n_samples, positives, negatives, val_positives, val_negatives, context_window, indices, values, isCompact)

    return dataset, val_dataset

def load_data_deepsucc(indices,context_window,proteins, fastas, p_samples, n_samples, positives, negatives, isCompact = False, isAvg = True):
    values = load_aa_index()
    
    dataset = create_data_deepsucc(proteins, fastas, p_samples, n_samples, positives, negatives, context_window, indices, values, isCompact)

    return dataset