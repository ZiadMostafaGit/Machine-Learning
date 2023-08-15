from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl

import sklearn.metrics as metrics
dataframe=pd.read_excel("/home/ziad/GitHub/Machine-Learning-journey/week3/data/gld_price_data.xlsx")
dataframe.dropna()
# dataframe=dataframe.drop("SLV",axis=1)
# dataframe=dataframe.drop("USO",axis=1)
# dataframe=dataframe.drop("SPX",axis=1)
# dataframe=dataframe.drop("Date",axis=1)
x=dataframe.drop("EUR/USD",axis=1)


y=dataframe["EUR/USD"]

print(x.columns)

scaler=StandardScaler()

y=y.values.reshape(-1,1)

# x_norm=pd.DataFrame(scaler.fit_transform(x),columns=x.columns)
# y_norm=pd.DataFrame(scaler.fit_transform(y))
# columns=x.columns

x_norm,x_test,y_norm,y_test=train_test_split(x,y)

model=LinearRegression()

# model = Ridge(alpha=1.0,max_iter=10000)

model.fit(x_norm,y_norm)


prodiction=model.predict(x_test)


mae = metrics.mean_absolute_error(y_test, prodiction)
mse = metrics.mean_squared_error(y_test, prodiction)
r2 = metrics.r2_score(y_test, prodiction)
# Print the results
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R2 score: {r2:.2f}")
# plt.scatter(prodiction,y_test)
# plt.show()
