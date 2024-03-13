import numpy as np

a = np.random.randint(-15, 15, (1, 20))
x = np.where(a < 0, -1, np.where(a > 0, 1, 0))


print(f'{a} \n {x}')