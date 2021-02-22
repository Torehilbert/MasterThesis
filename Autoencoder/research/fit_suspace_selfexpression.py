import numpy as np
import matplotlib.pyplot as plt

def loss_fn(x, xre, c):
    return np.mean(np.square(x - xre)) + np.sum(np.square(c))


if __name__ == "__main__":
    A = np.array([[1,2],[2,2]])
    B = np.array([[3,3]])

    h = 0.01
    lr = 0.001
    N = 25000
    roll_coef = 0.000

    points = []
    points.append([0,1])
    points.append([0,0.5])
    points.append([0.5,0])
    points.append([1,0])
    points = np.array(points)
    x0 = np.array([[1,1]])

    loss_best = np.inf
    c_best = None
    x0re_best = None


    for i in range(1000000):
        c = np.random.randn(1, points.shape[0])
        x0re = np.matmul(c, points)
        loss = loss_fn(x0, x0re, c)

        if loss < loss_best:
            loss_best = loss
            c_best = c
            x0re_best = x0re
    
    print(c_best)
    print(x0re_best)