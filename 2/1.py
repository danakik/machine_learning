import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('2/data.csv', delimiter=',')

#print(data.describe())

#print(data.head(3))
#print(data.tail(2))


#print(data['DebtRatio'])
#print(data['MonthlyIncome'])
data['DebtRatio'] = (data['DebtRatio'] < 1) * data['MonthlyIncome']
print(data['DebtRatio'])

data.rename(columns={'DebtRatio': 'Debt'}, inplace=True)

#print(data['MonthlyIncome'])
mean_monthly_income = data['MonthlyIncome'].mean()
data['MonthlyIncome'].fillna(mean_monthly_income, inplace=True)
#print(data['MonthlyIncome'])

probability_dependents = data['SeriousDlqin2yrs'].groupby(data['NumberOfDependents']).mean()
probability_real_estate_loans = data['SeriousDlqin2yrs'].groupby(data['NumberRealEstateLoansOrLines']).mean()
#print(probability_dependents)
#print(probability_real_estate_loans)



""" no_delinquency = data[data['SeriousDlqin2yrs'] == 0]
delinquency = data[data['SeriousDlqin2yrs'] == 1]

plt.scatter(no_delinquency['age'], no_delinquency['Debt'], color='blue', label='No Delinquency')
plt.scatter(delinquency['age'], delinquency['Debt'], color='red', label='Delinquency')


plt.title('Scatter Plot of Age vs Debt')
plt.xlabel('Age')
plt.ylabel('Debt')
plt.legend()

#plt.show() """



data['MonthlyIncome'] = data['MonthlyIncome'].clip(upper=25000)
no_delinquency = data[data['SeriousDlqin2yrs'] == 0]['MonthlyIncome']
delinquency = data[data['SeriousDlqin2yrs'] == 1]['MonthlyIncome']

plt.hist(no_delinquency, bins=30, density=True, color='blue', alpha=0.5, label='No Delinquency')
plt.hist(delinquency, bins=30, density=True, color='red', alpha=0.5, label='Delinquency')


plt.title('Normalized Density Histogram of Monthly Income')
plt.xlabel('Monthly Income')
plt.ylabel('Density')
plt.legend()
#plt.show()

data['MonthlyIncome'] = data['MonthlyIncome'].clip(upper=25000)

non_binary_features = ['age', 'MonthlyIncome', 'NumberOfDependents']

scatter_matrix = pd.plotting.scatter_matrix(data[non_binary_features], figsize=(10, 8), diagonal='kde')

for ax in scatter_matrix.ravel():
    ax.set_xlabel(ax.get_xlabel(), fontsize=10, rotation=45)
    ax.set_ylabel(ax.get_ylabel(), fontsize=10, rotation=0)

#plt.show()