import pickle
from fitness_deepsucc import fitness_deepsuc as fitness
from load_data import load_essentials_deepsucc as load_essentials
# from load_data import load_essentials_gpsuc as load_essentials

proteins, fastas, p_samples, n_samples, positives, negatives = load_essentials()

with open('Combinations/bests_optimzed(15)_50_0.7_0.9_1_20_5_0_mcc_rf_updated','rb') as f:
    while True:
        try:
            r,best = pickle.load(f)
            if r != 9:
                continue
            print(best)
            # fit = fitness(best,proteins, fastas, p_samples, n_samples, positives, negatives, True, False, 300, True)
            # print(r,fit)
        except EOFError:
            break