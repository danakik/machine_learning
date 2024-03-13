import numpy as np

coefficients = np.array([
    [1, 1, 2, 3],
    [3, -1, -2, -2],
    [2, -3, -1, -1],
    [1, 2, 3, -1]
])
constants = np.array([1, -4, -6, -4])

result = np.linalg.solve(coefficients, constants)

print(f'x1 = {round(result[0], 4)}')
print(f'x2 = {round(result[1], 4)}')
print(f'x3 = {round(result[2], 4)}')
print(f'x4 = {round(result[3], 4)}')
