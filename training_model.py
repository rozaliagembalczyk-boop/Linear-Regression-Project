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

print('Linear Model Coefficient (m): ', regressor.coef_)
print('Linear Model Coefficient (b): ', regressor.intercept_)

y_predict = regressor.predict( X_test)
print(y_predict)
print(y_test)
