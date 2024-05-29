import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def true_function(x):
    return -x**4 + 100*x**2 + x

x_values = np.arange(0, 10.1, 0.1)
y_values = true_function(x_values)

x_train = x_values[::2]
y_train = y_values[::2]

X_train = np.column_stack([x_train ** i for i in range(14)])

lr = LinearRegression()
lr.fit(X_train, y_train)

X_values = np.column_stack([x_values ** i for i in range(14)])
predictions_lr = lr.predict(X_values)


alpha_lasso = 1
lasso_coefficients = np.linalg.lstsq(X_train, y_train)[0]
lasso_predictions = np.dot(X_values, lasso_coefficients)


alpha_ridge = 1
ridge_coefficients = np.linalg.inv(X_train.T @ X_train + alpha_ridge * np.identity(X_train.shape[1])) @ X_train.T @ y_train
ridge_predictions = np.dot(X_values, ridge_coefficients)

r2_lr = r2_score(y_values, predictions_lr)
r2_lasso = r2_score(y_values, lasso_predictions)
r2_ridge = r2_score(y_values, ridge_predictions)

print(f'r2 Linear Regression: {r2_lr}')
print(f'r2 Lasso Regression: {r2_lasso}')
print(f'r2 Ridge Regression: {r2_ridge}')