import numpy as np

a = np.random.randint(0, 10, (1, 20))

x = a.reshape(4, 5)

print(f'Масив 1: \n{a},\n Масив 2: \n{x + 10}')