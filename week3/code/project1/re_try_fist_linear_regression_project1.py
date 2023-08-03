from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl
plt.style.use('./deeplearning.mplstyle')
DATA_FRAME = pd.read_excel(
    "/home/ziad/GitHub/Machine-Learning-journey/week3/data/car data 1.xlsx")


# def predict(w, b, x):
#     y = []
#     m = x.values.shape[0]  # fix typo here
#     for i in range(m):
#         y.append(np.dot(w, x.values[i])+b)
#     return y


DATA_FRAME.dropna()

# print(DATA_FRAME.columns)


# print(DATA_FRAME)


# print(DATA_FRAME.columns)


DATA_FRAME = DATA_FRAME.drop("Car_Name", axis=1)
DATA_FRAME = DATA_FRAME.drop("Owner", axis=1)
DATA_FRAME = DATA_FRAME.drop("Fuel_Type", axis=1)
DATA_FRAME = DATA_FRAME.drop("Seller_Type", axis=1)
DATA_FRAME = DATA_FRAME.drop("Transmission", axis=1)

x = DATA_FRAME.drop("Selling_Price", axis=1)

y = DATA_FRAME["Selling_Price"]

scaler = StandardScaler()


x_norm = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
y_norm = pd.DataFrame(scaler.fit_transform(y))


print(x_norm)
print(y_norm)


# print(x.columns)
# print(x.shape)


# print("*"*100)


# print(y.shape)
# DATA_FRAME = DATA_FRAME.drop("Address", axis=1)


# x = DATA_FRAME.drop("Price", axis=1)
# y = DATA_FRAME["Price"]

# scaler = StandardScaler()

# x_norm = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)


# # print(x_norm.head)
# y = y.values.reshape(-1, 1)
# y_norm = pd.DataFrame(scaler.fit_transform(y))
# # print("*"*100)
# # print(y_norm.head)

# col = x_norm.columns
# # plt.scatter(x_norm[col[1]], x_norm[col[2]])

# # print(col)

# model = LinearRegression()
# # plt.show()


# x_norm, x_test, y_norm, y_test = train_test_split(x_norm, y_norm)


# model.fit(x_norm, y_norm)


# pre = model.predict(x_test)


# # Compute evaluation metrics
# mse = mean_squared_error(y_test, pre)
# rmse = np.sqrt(mse)
# mae = mean_absolute_error(y_test, pre)
# r2 = r2_score(y_test, pre)


# print(f"MSE: {mse:.2f}")
# print(f"RMSE: {rmse:.2f}")
# print(f"MAE: {mae:.2f}")
# print(f"R2: {r2:.2f}")


# plt.scatter(y_test, pre)
# plt.xlabel("Actual values")
# plt.ylabel("Predicted values")
# plt.title("Actual vs. predicted values")
# plt.show()


# plt.scatter(pre, y_test, marker='x', c='r', label="Actual Value")
# plt.title("THE PRE DATA AND THE ACTUAL DATA")
# plt.xlabel("predicted data")
# plt.ylabel("actual values")
# plt.show()


# plt.scatter(pre, y_test, marker='x', c='r', label="Actual Value")
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
#          'k--', lw=2)  # Add a line of best fit
# plt.title("THE PRE DATA AND THE ACTUAL DATA")
# plt.xlabel("Predicted values")
# plt.ylabel("Actual values")
# plt.legend()
# plt.show()


# plt.scatter(x["Kms_Driven"], y)

# plt.xlabel("Kms_Driven")
# plt.ylabel("selling prices")
# plt.show()


# modil = LinearRegression()

# x_norm = x_norm.values.reshape(-1, -1)
# y_norm = y_norm.values.reshape(-1, 1)


# modil.fit(x_norm, y_norm)

# print(modil.coef_)
# print(modil.intercept_)


# print(x)
# print(y_norm)
