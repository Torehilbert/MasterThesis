import os
import matplotlib.pyplot as plt
import numpy as np


PATH_RUNS = r"D:\Speciale\Code\output\Performance Trainings\CRANDOM_4"


if __name__ == "__main__":
    
    # inspect performances
    conts = os.listdir(PATH_RUNS)

    P = np.zeros(shape=(10,10), dtype="float32")
    for cont in conts:
        ch1 = int(cont[0])
        ch2 = int(cont[1])

        with open(os.path.join(PATH_RUNS, cont, 'test_acc.txt'), 'r') as f:
            p = float(f.readline().split(" ")[0])
            P[ch1, ch2] = p
            P[ch2, ch1] = p

    xticklabels = []
    xticklabels_sorted = []
    ys = []
    ys_sorted = []

    for r in range(P.shape[0]):
        for c in range(P.shape[1]):
            if(c > r):
                xticklabels.append("%d-%d" % (r,c))
                ys.append(P[r,c])

    for (_, idx) in sorted((e,i) for i,e in enumerate(ys)):
        xticklabels_sorted.append(xticklabels[idx])
        ys_sorted.append(ys[idx])


    plt.figure(figsize=(10,5))
    plt.bar(xticklabels_sorted, ys_sorted)
    plt.ylim([0.92,0.97])
    plt.xticks(rotation=-75)
    plt.show()
