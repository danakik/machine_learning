import pandas as pd

data = pd.read_csv("2/titanic.csv")

# Операція 1: Кількість чоловіків та жінок
male_count = (data['sex'] == 'male').sum()
female_count = (data['sex'] == 'female').sum()
print("Кількість чоловіків:", male_count)
print("Кількість жінок:", female_count)

# Операція 2: Частка пасажирів, які вижили
survived_percentage = round((data['survived'].mean()) * 100, 2)
print("Частка пасажирів, які вижили:", survived_percentage)

# Операція 3: Частка пасажирів першого класу серед всіх пасажирів
first_class_percentage = round((data['pclass'] == 1).mean() * 100, 2)
print("Частка пасажирів першого класу:", first_class_percentage)

# Операція 4: Середній та медіана віку пасажирів
average_age = round(data['age'].mean(), 2)
median_age = round(data['age'].median(), 2)
print("Середній вік пасажирів:", average_age)
print("Медіана віку пасажирів:", median_age)
