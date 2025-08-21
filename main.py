import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#importing dataset
fueleconomy_df = pd.read_csv("FuelEconomy.csv")
print(fueleconomy_df.head(6))
print(fueleconomy_df.tail())
print(fueleconomy_df.describe())

#visualization
sns.jointplot(x='Horse Power', y='Fuel Economy (MPG)', data = fueleconomy_df)
plt.show()
sns.pairplot(fueleconomy_df)
plt.show()
sns.lmplot(x='Horse Power', y='Fuel Economy (MPG)', data=fueleconomy_df)
plt.show()

#testing data
X = fueleconomy_df[['Horse Power']]
y = fueleconomy_df['Fuel Economy (MPG)']
#print(X)
#print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#print(X_train.shape)

#training model
regressor = LinearRegression(fit_intercept =True)
regressor.fit(X_train,y_train)

print('Linear Model Coefficient (m): ', regressor.coef_)
print('Linear Model Coefficient (b): ', regressor.intercept_)

y_predict = regressor.predict( X_test)
print(y_predict)
print(y_test)

plt.scatter(X_train, y_train, color = 'gray')
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