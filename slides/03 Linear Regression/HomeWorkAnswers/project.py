import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns






df=pd.read_csv("/home/ziad/github/Machine-Learning/slides/03 Linear Regression/HomeWorkAnswers/train.csv")

# print(df.head())

df_sample=df.sample(10)

sns.pairplot(df_sample)
plt.show()

