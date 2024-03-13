import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return 2**x * np.sin(10*x)


x = np.linspace(-3, 3, 1000)


y = f(x)

plt.plot(x, y, label='Y(x) = 2^x * sin(10x)')
plt.title('Графік функції Y(x)')
plt.xlabel('x')
plt.ylabel('Y(x)')
plt.legend()
plt.grid(True)
plt.show()
