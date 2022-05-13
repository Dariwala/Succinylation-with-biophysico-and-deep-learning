import pickle

lines = open('aaindex1').read().split('\n')[:-1]

values = []
index = -1

# aaindex_for_15 = [126, 145]

for i in range(len(lines)):
    if lines[i] == '//':
        index += 1

        # if index in aaindex_for_15:
        #     print(i)
        # if index == 67 or index == 94 or index == 129 or index == 145:
        #     print(i)
        value = lines[i-2].split()
        v = {}
        try:
            v['A'] = float(value[0])
        except ValueError:
            v['A'] = 0
            print(index)
            continue
        try:
            v['R'] = float(value[1])
        except ValueError:
            v['R'] = 0
            print(index)
            continue
        v['N'] = float(value[2])
        v['D'] = float(value[3])
        v['C'] = float(value[4])

        v['Q'] = float(value[5])
        v['E'] = float(value[6])
        try:
            v['G'] = float(value[7])
        except ValueError:
            v['G'] = 0
            print(index)
            continue
        v['H'] = float(value[8])
        v['I'] = float(value[9])

        value = lines[i-1].split()

        v['L'] = float(value[0])
        v['K'] = float(value[1])
        v['M'] = float(value[2])
        v['F'] = float(value[3])
        try:
            v['P'] = float(value[4])
        except ValueError:
            v['P'] = 0
            print(index)
            continue

        v['S'] = float(value[5])
        v['T'] = float(value[6])
        v['W'] = float(value[7])
        v['Y'] = float(value[8])
        v['V'] = float(value[9])

        v['X'] = 0
        v['B'] = 0
        v['U'] = 0
        v['Z'] = 0

        values.append(v)

        # if len(values) == 513:
        #     print(i, 'haha')

# print(len(values))
# print(values[512])

# for i in [33,76,87,93,145,400,447,507,510]:
#     stri = ''
#     for aa in 'ARNDCQEGHILKMFPSTWYV':
#         stri += str(values[i][aa]) + ' '
#     print(stri)

with open('aaindex1_extracted_non_empty','wb') as f:
    pickle.dump(values,f)