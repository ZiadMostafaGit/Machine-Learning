import numpy as np
import os

npy_root_dir = '/home/moustafa/0hdd/00Udemy/MachineLearning/0code/9-multi-class/homework'

def softmax_batch(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


if __name__ == '__main__':
    # batch of 3 examples, each is 4 features
    arr2d = np.array([[1, 2, 3, 4],
                     [1, 3, 7, 8],
                     [2, 6, 10, 5]])

    your_answer = softmax_batch(arr2d)
    right_answer = np.load(os.path.join(npy_root_dir, 'softmax.npy'))

    if np.allclose(your_answer, right_answer, atol=1e-6):
        print("Good job")
    else:
        print("Something wrong")
        print("your answer")
        print(your_answer)
        print("Optimal answer")
        print(right_answer)
