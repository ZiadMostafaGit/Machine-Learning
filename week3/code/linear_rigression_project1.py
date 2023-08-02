import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

df = pd.read_csv(
    "/home/ziad/GitHub/Machine-Learning-journey/week3/data/Housing_Data (2).csv")

df.dropna(inplace=True)

df = df.drop("Address", axis=1)

x = df.drop("Price", axis=1)

y = df["Price"]

scaler = StandardScaler()
y = y.values.reshape(-1, 1)
x_norm = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
y_norm = pd.DataFrame(scaler.fit_transform(y))


x_norm, x_test, y_norm, y_test = train_test_split(
    x_norm, y_norm, test_size=0.3)


modil = LinearRegression()

modil.fit(x_norm, y_norm)

w = modil.coef_

b = modil.intercept_
prediction = modil.predict(x_test)


mae = metrics.mean_absolute_error(y_test, prediction)
mse = metrics.mean_squared_error(y_test, prediction)
r2 = metrics.r2_score(y_test, prediction)

# Print the results
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R2 score: {r2:.2f}")

# plt.scatter(y_test, prediction)
# plt.hist(y_test-prediction)
# plt.show()
