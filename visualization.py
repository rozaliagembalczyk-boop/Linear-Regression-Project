import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#importing dataset
fueleconomy_df = pd.read_csv("FuelEconomy.csv")

#visualization: Horsepower vs Fuel Economy
g = sns.jointplot(x='Horse Power', y='Fuel Economy (MPG)', data=fueleconomy_df)
g.fig.suptitle("Horsepower vs Fuel Economy (Jointplot)", y=1)  # y adjusts spacing
plt.show()

#visualization: Pairwise Relationships in Fuel Economy Data
h = sns.pairplot(fueleconomy_df)
h.fig.suptitle("Pairwise Relationships in Fuel Economy Data", y=1)
plt.show()

#visualization: Horsepower vs Fuel Economy with Regression Line
m = sns.lmplot(x='Horse Power', y='Fuel Economy (MPG)', data=fueleconomy_df)
m.fig.suptitle("Horsepower vs Fuel Economy with Regression Line", y=1)
plt.show()
