import numpy as np
import matplotlib.pyplot as plt

PATH = r"D:\Speciale\Code\output\DSC_New\Runs_3264_relu\2020-12-08--14-43-12_DSC_3264_relu\matrices\Z_norm.csv"


def permute(M, idx1, idx2):
    N = M.shape[0]
    M_copy = np.zeros_like(M)

    inds = [i for i in range(N)]
    inds[idx1] = idx2
    inds[idx2] = idx1

    for r in range(N):
        for c in range(N):
            M_copy[r,c] = M[inds[r], inds[c]]
    
    return M_copy


def measure_contrast(M):
    N = M.shape[0]
    diff = 0
    for r in range(0, N):
        for c in range(r+1, N):
            diff += M[r,c] * (c - r+1)
    return diff

def get_permutations(N):
    perms = []
    for i in range(N):
        for c in range(i+1, N):
            perms.append((i,c))
    return perms


if __name__ == "__main__":
    # CONCLUSION: NOT BAD! :)
    
    M = np.loadtxt(PATH, delimiter=",")
    #M = np.array([[1,2,3],[2,4,5],[3,5,6]], dtype=int)
    
    plt.figure()
    plt.imshow(M)

    M_perm = np.array(M, copy=True)
    for it in range(1000):
        perms = get_permutations(M_perm.shape[0])
        diff_ref = measure_contrast(M_perm)
        diffs = []
        for i,p in enumerate(perms):
            M_new = permute(M_perm, p[0], p[1])
            diffs.append(measure_contrast(M_new))

        idx_perm = np.argmin(diffs)
        M_perm = permute(M_perm, perms[idx_perm][0], perms[idx_perm][1])

    plt.figure()
    plt.imshow(M_perm)


    plt.show()