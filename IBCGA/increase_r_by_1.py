import random
def increase_r_by_1(individuals, extra_bits):
    for i in range(len(individuals)):
        while True:
            j = random.randrange(len(individuals[i])-extra_bits)
            if individuals[i][j] == 0:
                individuals[i][j] = 1
                break
    return individuals