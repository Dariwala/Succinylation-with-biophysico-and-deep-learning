import numpy as np

def generate_OA(N):
    n = int(2 ** np.ceil(np.log2(N+1)))
    # print(n, type(n))
    oa = np.zeros((n,N),dtype=int)

    for i in range(n):
        for j in range(N):
            level = 0
            k = j 
            mask = int(n / 2)
            while k > 0:
                if k%2 == 1 and ((i-1)&mask) != 0:
                    level = (level+1)%2
                k = np.floor(k / 2)
                mask = int(mask / 2)
            oa[i][j] = level + 1
    return oa

# generate_OA(3)