import torch
import numpy as np
from numpy.random import random

# requires_grad=True => update weights
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# Define a simple function and compute gradient
y = x * 2
z = y.mean()
z.backward()        # Compute gradients
print(x.grad)       # Gradients value

a = torch.from_numpy(np.array([1, 2, 3]))
a = torch.ones(3, 3)    # zeros / eye(3)
print(a.shape, a.dtype)

b = a.view(1, 9)     # reshape to 1 x 9
c = a.view(1, -1)    # reshape to 1 x 9
c = c.float()
print(c.device, c, c.dtype)   # cpu 1 1 1  1 1



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
c = c.to(device)
print(c.device, c)  # cuda:0

arr = random((2, 3, 4, 5))  # numpy

tensor = torch.from_numpy(arr).to(device)
arr = tensor.cpu().numpy()  # if not on gpu, move first

'''
Many like numpy: add, multiply, dot, broadcast, slice

Educate yourself and play
- torch.topk, unsqueeze, expand, repeat, gather
'''
