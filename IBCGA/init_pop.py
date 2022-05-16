import random
# from selection_for_crossover import selection_for_crossover
from mutation import mutation

def init_pop(Npop, r, isCompact = False):
    binary_gene_length = 40 if isCompact else 553
    bits_for_context = 0 if isCompact else 3
    individuals = []
    for _ in range(Npop):
        individual = []
        for _ in range(bits_for_context+binary_gene_length):
            individual.append(0)
        cons = [i for i in range(binary_gene_length)]
        # print(cons)
        random.shuffle(cons)
        # print(cons)

        for j in range(r):
            # print(cons[j])
            individual[cons[j]] = 1
        
        for j in range(1,bits_for_context+1):
            ra = random.random()
            if ra < 0.5:
                individual[-j] = 1
        individuals.append(individual)
    return individuals

# individuals = init_pop(5,1)
# print(individuals)
# print(mutation(individuals,1))
# crossover_pop = selection_for_crossover(individuals,5)
# print(crossover_pop)