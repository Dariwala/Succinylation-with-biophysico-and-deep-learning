import pickle
from fitness import fitness
from load_data import load_essentials
# from load_data import load_essentials_test as load_essentials
import sys
# from load_data import load_essentials_gpsuc as load_essentials

proteins, val_proteins, fastas, val_fastas, p_samples, n_samples, positives, negatives, val_positives, val_negatives = load_essentials()
# proteins, val_proteins, fastas, val_fastas, train_p_samples, train_n_samples, test_p_samples, test_n_samples, positives, negatives, val_positives, val_negatives = load_essentials()

with open('Combinations/bests_optimzed(16)_50_'+str(sys.argv[1])+'_'+str(sys.argv[2])+'_1_20_5_0_mcc_rf_updated','rb') as f:
    while True:
        try:
            r,best = pickle.load(f)
            # if r != 20:
            #     continue
            # for i in range(len(best)):
            #     if best[i] == 0:
            #         best[i] = 1
            fit = fitness(best,proteins, val_proteins, fastas, val_fastas, p_samples, n_samples, positives, negatives, val_positives, val_negatives, True, False, 300, True)
            # fit = fitness(best,proteins, val_proteins, fastas, val_fastas, train_p_samples, train_n_samples, test_p_samples, test_n_samples, positives, negatives, val_positives, val_negatives, True, False, 300, True)
            print(r,fit[0],fit[1],fit[2],fit[3])
        except EOFError:
            break