from fitness import fitness
from mutation import mutation
from crossover import crossover
from init_pop import init_pop
from tournament_selection import tournament_selection
from increase_r_by_1 import increase_r_by_1
from load_data import load_essentials
import numpy as np
import pickle

proteins, val_proteins, fastas, val_fastas, p_samples, n_samples, positives, negatives, val_positives, val_negatives = load_essentials()

Npop = 50
Pc = 0.8
Pm = 0.5
rs = 1
rend = 20
iterations = 5
extra_bits = 0
isAvg = False
isCompact = True

r = rs

individuals = init_pop(Npop,rs, isCompact)
best = []
max_fit = -np.inf
best_ind = -1

while r <= rend:
    local_best = []
    local_max_fit = -np.inf
    local_best_ind = -1
    for _ in range(iterations):
        # print('shuru')
        fits = []
        for i in range(len(individuals)):
            # print('individual' + str(i))
            fit = fitness(individuals[i],proteins, val_proteins, fastas, val_fastas, p_samples, n_samples, positives, negatives, val_positives, val_negatives, isCompact, isAvg)
            fits.append(fit)

            if fit > max_fit:
                max_fit = fit
                best = individuals[i]
                best_ind = i
            if fit > local_max_fit:
                local_max_fit = fit
                local_best = individuals[i]
                local_best_ind = i
        
        crossover_number = int(Pc * len(individuals))

        for _ in range(int(crossover_number/2)):
            # print('crossover')
            p1, i1 = tournament_selection(individuals, fits)
            p2, i2 = tournament_selection(individuals, fits)

            c1, c2, f1, f2 = crossover(p1,p2,r,extra_bits,proteins, val_proteins, fastas, val_fastas, p_samples, n_samples, positives, negatives, val_positives, val_negatives)

            if f1 != -1 and f2 != -1:
                individuals[i1] = c1
                individuals[i2] = c2

                fits[i1] = f1
                fits[i2] = f2
        # print('mutation shuru')
        individuals = mutation(individuals,int(Pm * len(individuals)),best_ind,extra_bits)
        # print('mutation shesh')
    for i in range(len(individuals)):
        fit = fitness(individuals[i],proteins, val_proteins, fastas, val_fastas, p_samples, n_samples, positives, negatives, val_positives, val_negatives, isCompact, isAvg)

        if fit > max_fit:
            max_fit = fit
            best = individuals[i]
            best_ind = i
        if fit > local_max_fit:
            local_max_fit = fit
            local_best = individuals[i]
            local_best_ind = i
    with open('bests_optimzed_w(0.7)_50_8_6_1_20_5_0','ab') as f:
        pickle.dump([r,local_best],f)
    print(r, local_max_fit)
    if r < rend:
        individuals = increase_r_by_1(individuals,extra_bits)
    r += 1

# with open('bests','ab') as f:
#     pickle.dump([-1,local_best],f)