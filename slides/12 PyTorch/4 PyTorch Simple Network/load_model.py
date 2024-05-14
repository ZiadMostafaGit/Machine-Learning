

import torch
import os
from simple_regressor_v2 import SimpleNet

version = 'v2'


def try1():
    model1 = torch.load(f'model_{version}.pth')
    print(model1)


def try2():
    # loading v2 model vs v1 definition mismatch in keys
    input_dim = 65
    model2 = SimpleNet(input_dim)   # reconstruct
    dict = torch.load(f'model_weights_{version}.pth')
    #model2.load_state_dict(dict)
    #print(model2)
    print(dict.keys())



if __name__ == '__main__':
    #try1()
    try2()


