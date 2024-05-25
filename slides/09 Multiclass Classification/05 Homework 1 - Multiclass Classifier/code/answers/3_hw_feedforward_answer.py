import numpy as np
import os

npy_root_dir = '/home/moustafa/0hdd/00Udemy/MachineLearning/0code/9-multi-class/homework'


def softmax_batch(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def forward(X_batch, W1, b1, W2, b2, W3, b3):   # use tanh
    net1 = np.dot(X_batch, W1) + b1
    out1 = np.tanh(net1)
    net2 = np.dot(out1, W2) + b2
    out2 = np.tanh(net2)
    net3 = np.dot(out2, W3) + b3
    out3 = softmax_batch(net3)                  # softmax

    return net1, out1, net2, out2, net3, out3



def check(your_answer, right_answer, name):
    if your_answer.shape != right_answer.shape:
        print(f"\nSomething wrong for {name}")
        print("your answer shape")
        print(your_answer.shape)
        print("Optimal answer shape")
        print(right_answer.shape)
        exit()


    if np.allclose(your_answer, right_answer, atol=1e-6):
        print("Good job")
    else:
        print(f"\nSomething wrong for {name}")
        print("your answer")
        print(your_answer.shape, your_answer)
        print("Optimal answer")
        print(right_answer.shape, right_answer)
        exit()

if __name__ == '__main__':
    X_batch = np.load(os.path.join(npy_root_dir, 'X_batch.npy'))
    W1 = np.load(os.path.join(npy_root_dir, 'W1.npy'))
    b1 = np.load(os.path.join(npy_root_dir, 'b1.npy'))
    W2 = np.load(os.path.join(npy_root_dir, 'W2.npy'))
    b2 = np.load(os.path.join(npy_root_dir, 'b2.npy'))
    W3 = np.load(os.path.join(npy_root_dir, 'W3.npy'))
    b3 = np.load(os.path.join(npy_root_dir, 'b3.npy'))

    opt_net1 = np.load(os.path.join(npy_root_dir, 'net1.npy'))
    opt_out1 = np.load(os.path.join(npy_root_dir, 'out1.npy'))
    opt_net2 = np.load(os.path.join(npy_root_dir, 'net2.npy'))
    opt_out2 = np.load(os.path.join(npy_root_dir, 'out2.npy'))
    opt_net3 = np.load(os.path.join(npy_root_dir, 'net3.npy'))
    opt_out3 = np.load(os.path.join(npy_root_dir, 'out3.npy'))

    net1, out1, net2, out2, net3, out3 = forward(X_batch, W1, b1, W2, b2, W3, b3)

    check(net1, opt_net1, "net1")
    check(out1, opt_out1, "out1")
    check(net2, opt_net2, "net2")
    check(out2, opt_out2, "out2")
    check(net3, opt_net3, "net3")
    check(out3, opt_out3, "out3")
