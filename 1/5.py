import numpy as np

a = np.random.randint(0, 10, (1, 15))
num = np.mean(a)
num = round(num, 3)
x = a - num
print(f'Масив: {a},\n Середю : {num},\n {np.sort(x)}')
