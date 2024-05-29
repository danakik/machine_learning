import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('3/student_scores.csv')
x = data[['Hours']]
y = data['Scores']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

print('Coefficients:', model.coef_)
print('Intercept:', round(model.intercept_, 3))

mse_train = mean_squared_error(y_train, pred_train)
mse_test = mean_squared_error(y_test, pred_test)
r2_train = r2_score(y_train, pred_train)
r2_test = r2_score(y_test, pred_test)

print('Train Squared Error:', round(mse_train, 3))
print('Test Squared Error:', round(mse_test, 3))
print('Train R^2 Score:', round(r2_train, 3))
print('Test R^2 Score:', round(r2_test, 3))

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, pred_test, color='black', linewidth=3)
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title('Linear Regression (Test Data)')

plt.subplot(1, 2, 2)
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, pred_train, color='black', linewidth=3)
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title('Linear Regression (Train Data)')

plt.tight_layout()
plt.show()
