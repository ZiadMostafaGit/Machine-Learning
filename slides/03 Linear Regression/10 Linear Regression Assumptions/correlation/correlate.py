import numpy as np
import matplotlib.pyplot as plot
from scipy import stats
import seaborn as sns

from data_helper import load_data


df, _, x, t, _ = load_data('data/dataset_200x4_regression.csv')

# You can use pandas to get the correlation matrix
correlation_matrix = df.corr()
round(correlation_matrix, 2)
print(correlation_matrix)

# plot the matrix heatmap
sns.heatmap(correlation_matrix)
plot.show()

# we can also use compute correlation of 2 columns of data
# using pandas
ans = df['Feat1'].corr(df['Target'])    # panda series
print(ans)
# using stats
ans = stats.pearsonr(df['Feat1'], df['Target'])[0]  # second value is p-value
print(ans)

ans = stats.pearsonr(x[:, 0], t)[0]
print(ans)



'''
           Feat1     Feat2     Feat3    Target
Feat1   1.000000  0.054809  0.056648  0.901208
Feat2   0.054809  1.000000  0.354104  0.349631
Feat3   0.056648  0.354104  1.000000  0.157960
Target  0.901208  0.349631  0.157960  1.000000

0.9012079133023306
0.9012079133023306

'''
