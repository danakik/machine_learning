import numpy as np

x = np.array([1, 2, 3, 4, 5])
y = np.array([6, 7, 8, 9, 10])

a = np.concatenate((x, y))


print(f'Масив 1: {x}\n'
      f'Масив 2: {y}\n'
      f'Додавання: {x + y}\n'
      f'Віднімання: {y - x}\n'
      f'Множення: {x * y}\n'
      f'Ділення: {np.round((y / x), 3)}\n'
      f'Масив 1, 2: {a}\n'
      f'min a: {min(a)}\n'
      f'max a: {max(a)}')
