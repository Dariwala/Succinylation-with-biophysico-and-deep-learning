from fitness import fitness
import random

def tournament_selection(individuals, fits):
    i = random.randrange(len(individuals))
    best = individuals[i]
    ind = i
    b_fit = fits[ind]

    j = random.randrange(len(individuals))
    if fits[j] > b_fit:
        best = individuals[j]
        ind = j

    return best, ind