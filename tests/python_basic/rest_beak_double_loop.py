
eos = 0

tokens = [[1, 2, 3, 4, 5, eos, 6, 7], 
 [2, 3, 1, eos, 5, eos, 6, 7],
 [4, 5, 6, 8, 9, eos]
 ]

for row in tokens:
    for t in row:
        if t == eos:
            break
        print(t)
    print()