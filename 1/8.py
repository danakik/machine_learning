import numpy as np

a = np.array(
    [[2, 3, -1],
     [4, 5, 2],
     [-1, 0, 7]])

b = np.array(
    [[-1, 0, 5],
     [0, 1, 3],
     [2, -2, 4]])

x = 2 * (a + b) * (2 * b - a)

print(x)