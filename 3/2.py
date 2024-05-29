import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = pd.read_csv('3/petrol_consumption.xls')

X = data.drop('Petrol_Consumption', axis=1)
y = data['Petrol_Consumption']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
mse_train = mean_squared_error(y_train, y_train_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)

print('Train Mean Squared Error:', round(mse_train, 3))
print('Train Mean Absolute Error:', round(mae_train, 3))
print('Train Root Mean Squared Error:', round(rmse_train, 3))

r2_train = r2_score(y_train, y_train_pred)
print('Train R^2 Score:', round(r2_train, 3))

y_test_pred = model.predict(X_test)
mse_test = mean_squared_error(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)

print('Test Mean Squared Error:', round(mse_test, 3))
print('Test Mean Absolute Error:', round(mae_test, 3))
print('Test Root Mean Squared Error:', round(rmse_test, 3))

r2_test = r2_score(y_test, y_test_pred)
print('Test R^2 Score:', round(r2_test, 3))


plt.figure(figsize=(10, 6))
plt.scatter(y_train, y_train_pred, color='blue', label='Training Data')
plt.scatter(y_test, y_test_pred, color='red', label='Testing Data')
plt.legend(loc='upper left')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='black', lw=3)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()
