import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def fun(X, y):
    classes = np.unique(y)
    mean = {c: np.mean(X[y == c], axis=0) for c in classes}
    std = {c: np.std(X[y == c], axis=0) for c in classes}
    class_prior = {c: np.mean(y == c) for c in classes}
    return mean, std, class_prior

def predict(X, mean, std, class_prior):
    posterior = np.zeros((X.shape[0], len(class_prior)))
    for idx, c in enumerate(class_prior):
        likelihood = (1 / (np.sqrt(2 * np.pi * std[c]))) * np.exp(-0.5 * ((X - mean[c]) ** 2 / std[c] ** 2) )
        posterior[:, idx] = np.prod(likelihood, axis=1) * class_prior[c]
    predicted_y = np.argmax(posterior, axis=1)
    predicted_y = np.where(predicted_y == 0, -1, 1)
    return predicted_y


 
data_x = np.array([(4.9, 3.3), (5.6, 4.5), (6.4, 4.3), (6.7, 5.7), (6.3, 5.0), (5.2, 3.9), (5.5, 3.7), (5.6, 3.6), (5.5, 3.8), (6.1, 4.7), (7.4, 6.1), (6.0, 5.1), (5.5, 4.4), (5.9, 5.1), (6.5, 5.8), (6.5, 4.6), (6.7, 4.4), (6.3, 5.6), (5.9, 4.8), (6.0, 4.5), (5.6, 4.1), (5.6, 4.9), (4.9, 4.5), (6.2, 4.5), (6.1, 4.7), (6.1, 4.9), (6.2, 5.4), (5.7, 4.2), (6.1, 5.6), (5.8, 4.0), (6.6, 4.6), (5.6, 4.2), (7.2, 6.1), (7.7, 6.7), (5.6, 3.9), (7.7, 6.9), (6.0, 4.0), (6.1, 4.0), (7.6, 6.6), (5.1, 3.0), (6.3, 6.0), (6.7, 5.7), (6.8, 5.9), (6.4, 5.5), (7.0, 4.7), (5.8, 5.1), (5.8, 5.1), (6.4, 5.3), (6.3, 4.9), (6.4, 5.3), (5.7, 3.5), (7.2, 5.8), (6.4, 5.6), (5.7, 4.5), (6.0, 4.5), (7.7, 6.1), (6.2, 4.3), (7.1, 5.9), (7.3, 6.3), (5.0, 3.3), (6.3, 5.1), (5.8, 3.9), (6.4, 4.5), (6.3, 5.6), (6.8, 5.5), (6.9, 5.4), (5.5, 4.0), (5.7, 4.1), (6.5, 5.5), (6.3, 4.7), (5.0, 3.5), (6.7, 5.8), (6.9, 4.9), (7.7, 6.7), (5.8, 4.1), (6.4, 5.6), (6.7, 5.2), (6.7, 4.7), (5.4, 4.5), (6.8, 4.8), (5.7, 4.2), (5.5, 4.0), (6.3, 4.9), (6.5, 5.2), (5.8, 5.1), (6.0, 4.8), (6.2, 4.8), (6.5, 5.1), (7.9, 6.4), (6.7, 5.0), (6.7, 5.6), (6.0, 5.0), (6.1, 4.6), (5.7, 5.0), (7.2, 6.0), (6.3, 4.4), (5.9, 4.2), (6.9, 5.1), (6.6, 4.4), (6.9, 5.7)])
data_y = np.array([-1, -1, -1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, 1])
    
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=2)

mean, std, class_prior = fun(x_train, y_train)
predicted_y = predict(x_test, mean, std, class_prior)
print(predicted_y)
print(y_test)
accuracy = np.mean(predicted_y == y_test)
print(f'Число неправильних класифікацій: {len(y_test) - np.sum(predicted_y == y_test)}')
print(f'Відсоток неправильних класифікацій: {100 * (1 - accuracy):.2f}%')

class_minus_1 = x_test[y_test == -1]
class_1 = x_test[y_test == 1]

""" plt.figure(figsize=(8, 6))
plt.scatter(class_minus_1[:, 0], class_minus_1[:, 1], c='red', label='-1')
plt.scatter(class_1[:, 0], class_1[:, 1], c='blue', label='1')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Test Data')
plt.legend()
plt.grid(True)
plt.show() """
