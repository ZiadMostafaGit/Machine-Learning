import numpy as np
from sklearn.preprocessing import MinMaxScaler

data = np.array([ [1, 10],
                 [2, 20],
                 [3, 21],
                 [4, 22],
                 [5, 90]])
print(data)
'''
[[ 1 10]
 [ 2 20]
 [ 3 21]
 [ 4 22]
 [ 5 90]]
'''

processor = MinMaxScaler()
data_scaled = processor.fit_transform(data)

print(data_scaled)
'''
[[0.     0.    ]
 [0.25   0.125 ]
 [0.5    0.1375]
 [0.75   0.15  ]
 [1.     1.    ]]
'''

test_data = np.array([[1, 5],
                     [2.5, 20],
                     [6, 100]])

print(processor.transform(test_data))
'''
[[ 0.     -0.0625]
 [ 0.375   0.125 ]
 [ 1.25    1.125 ]]
'''



from sklearn.preprocessing import StandardScaler

processor = StandardScaler()
data_scaled = processor.fit_transform(data)

print(data_scaled)
print(processor.transform(test_data))


