import random
import copy

def swap_mutation(individual,extra_bits):
    i = random.randrange(len(individual)-extra_bits)

    while True:
        j = random.randrange(len(individual)-extra_bits)
        if individual[i] != individual[j]:
            break
    r = individual[i]
    individual[i] = individual[j]
    individual[j] = r
    return individual

def mutation(individuals, desired_number, best_ind,extra_bits):
    fake_pop = [i for i in range(len(individuals))]
    del fake_pop[best_ind]
    random.shuffle(fake_pop)

    for i in range(desired_number):
        individuals[fake_pop[i]] = swap_mutation(individuals[fake_pop[i]],extra_bits)
    return individuals
    