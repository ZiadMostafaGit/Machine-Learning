import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sas
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
from scipy import stats
df = pd.read_csv(
    "/home/ziad/GitHub/Machine-Learning-journey/week3/data/california_housing_test.csv")

df.dropna(inplace=True)
print(df.shape)


z_scores = stats.zscore(df)

_scores = stats.zscore(df)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
new_df = df[filtered_entries]
print(new_df)
