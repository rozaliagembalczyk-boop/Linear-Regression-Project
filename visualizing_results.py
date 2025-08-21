import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

fueleconomy_df = pd.read_csv("FuelEconomy.csv")

X = fueleconomy_df[['Horse Power']]
y = fueleconomy_df['Fuel Economy (MPG)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
regressor = LinearRegression(fit_intercept =True)
regressor.fit(X_train,y_train)

plt.scatter(X_train, y_train, color = 'lightblue')
plt.plot(X_train, regressor.predict(X_train), color = 'red')
plt.ylabel('MPG')
plt.xlabel('Horse Power (HP)')
plt.title('HP vs. MPG (Training dataset)')
plt.show()

# VISUALIZE TEST SET RESULTS
plt.scatter(X_test, y_test, color = 'gray')
plt.plot(X_test, regressor.predict(X_test), color = 'red')
plt.ylabel('MPG')
plt.xlabel('Horse Power (HP)')
plt.title('HP vs. MPG (Testing dataset)')
plt.show()

HP = [[245]]
y_predict = regressor.predict(HP)
print(y_predict)