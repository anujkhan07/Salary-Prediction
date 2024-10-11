# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary.csv')
x = dataset.iloc[:, :-1].values  # Independent variable (Years of experience)
y = dataset.iloc[:, -1].values   # Dependent variable (Salary)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test set results
y_predict = regressor.predict(x_test)

# Visualising the Training set results
plt.scatter(x_train, y_train, color='blue')
plt.plot(x_train, regressor.predict(x_train), color='red')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(x_test, y_test, color='blue')
plt.plot(x_train, regressor.predict(x_train), color='red')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Let's predict a salary of an employee with a given number of years of experience
years = input("Enter the years of experience: ")
float_years = float(years)

prediction = regressor.predict([[float_years]])
print("The predicted salary is: ", prediction)
