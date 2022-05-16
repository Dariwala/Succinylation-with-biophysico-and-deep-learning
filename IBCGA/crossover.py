from numpy.core.numeric import cross
from init_pop import init_pop
from generate_OA import generate_OA
from fitness import fitness
import numpy as np

def list_ones(individual):
    ones = []

    for i in range(len(individual)):
        if individual[i] == 1:
            ones.append(i)
    return ones

def create_child(individual1, individual2, factor_level,cut_points):
    prev = 0
    individual = []
    for i in range(len(factor_level)):
        # try:
            # assert len(list_ones(individual1[prev:cut_points[i]+1])) == len(list_ones(individual2[prev:cut_points[i]+1]))
        # except AssertionError:
            # print(prev,i,cut_points[i]+1)
            # print(individual1[prev:cut_points[i]+1],individual2[prev:cut_points[i]+1])
        if factor_level[i] == 1:
            individual += individual1[prev:cut_points[i]+1]
        else:
            individual += individual2[prev:cut_points[i]+1]
        prev = cut_points[i]+1
    return individual

def crossover(individual1, individual2,r,extra_bits,proteins, val_proteins, fastas, val_fastas, p_samples, n_samples, positives, negatives, val_positives, val_negatives):
    if individual1 == individual2:
        return individual1, individual2, -1, -1
    elif individual1[:-extra_bits] == individual2[:-extra_bits]:
        return individual1, individual2, -1, -1
    # try:
    #     assert len(list_ones(individual1[:-extra_bits])) == r
    #     assert len(list_ones(individual2[:-extra_bits])) == r
    # except AssertionError:
    #     exit(1)
    
    N = 0
    cut_points = []
    o1 = o2 = 0
    isEqual = True
    for ind in range(len(individual1)-extra_bits):
        if individual1[ind] == 1:
            o1 += 1
        if individual2[ind] == 1:
            o2 += 1

        if individual1[ind] != individual2[ind]:
            isEqual = False

        if o1 == o2 and isEqual == False:
            isEqual = True
            N += 1
            cut_points.append(ind)
    

    if isEqual:
        cut_points[-1] = len(individual1) - 1
    else:
        N += 1
        cut_points.append(len(individual1) - 1)
    
    try:
        assert N > 1
    except AssertionError:
        # print(N,'Assertion error here')
        return individual1, individual2, -1, -1

    assert N == len(cut_points)

    oa = generate_OA(N)
    S = np.zeros((N,2),dtype=float)
    # MED = []

    children = []

    for i in range(oa.shape[0]):
        factor_level = oa[i]
        child = create_child(individual1,individual2, list(factor_level), cut_points)
        children.append(child)
        fit = fitness(child,proteins, val_proteins, fastas, val_fastas, p_samples, n_samples, positives, negatives, val_positives, val_negatives)

        for i in range(N):
            # print(factor_level)
            # print(factor_level[i]-1)
            S[i][factor_level[i]-1] += fit
    
    best_factor = []
    min_MED = np.inf
    worst_factor = -1
    for i in range(N):
        if S[i][0] > S[i][1]:
            best_factor.append(1)
        else:
            best_factor.append(2)
        diff = np.abs(S[i][0] - S[i][1])
        if diff < min_MED:
            min_MED = diff
            worst_factor = i
    child1 = create_child(individual1,individual2, best_factor, cut_points)

    if best_factor[worst_factor] == 1:
        best_factor[worst_factor] = 2
    else:
        best_factor[worst_factor] = 1
    child2 = create_child(individual1,individual2, best_factor, cut_points)

    # print(len(list_ones(child1[:-extra_bits])))
    # print(len(child1))
    # print(len(list_ones(child2[:-extra_bits])))
    # print(len(child2))
    # try:
    #     assert len(list_ones(child1[:-extra_bits])) == r
    #     assert len(list_ones(child2[:-extra_bits])) == r
    # except AssertionError:
    #     print('hayhay')
    #     print(individual1,individual2,cut_points)
    #     exit(1)
    
    return child1, child2, fitness(child1,proteins, val_proteins, fastas, val_fastas, p_samples, n_samples, positives, negatives, val_positives, val_negatives), fitness(child2,proteins, val_proteins, fastas, val_fastas, p_samples, n_samples, positives, negatives, val_positives, val_negatives)

# individuals = init_pop(10,2)
# c1,c2,f1,f2 = crossover(individuals[0],individuals[1],2,3)