import numpy as np
import os

npy_root_dir = '/home/moustafa/0hdd/00Udemy/MachineLearning/0code/9-multi-class/homework'


def dtanh(y):   # tanh derivative
    return 1 - y ** 2


def backward(X_batch, y_batch, W2, W3, out1, out2, out3):
    dE_dnet3 = out3 - y_batch
    dE_dout2 = np.dot(dE_dnet3, W3.T)
    dE_dnet2 = dE_dout2 * dtanh(out2)
    dE_dout1 = np.dot(dE_dnet2, W2.T)
    dE_dnet1 = dE_dout1 * dtanh(out1)

    dW3 = np.dot(out2.T, dE_dnet3)
    db3 = np.sum(dE_dnet3, axis=0, keepdims=True)
    dW2 = np.dot(out1.T, dE_dnet2)
    db2 = np.sum(dE_dnet2, axis=0, keepdims=True)
    dW1 = np.dot(X_batch.T, dE_dnet1)
    db1 = np.sum(dE_dnet1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2, dW3, db3, dE_dnet3, dE_dout2, dE_dnet2, dE_dnet1


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
    y_batch = np.load(os.path.join(npy_root_dir, 'y_batch.npy'))
    W2 = np.load(os.path.join(npy_root_dir, 'W2.npy'))
    W3 = np.load(os.path.join(npy_root_dir, 'W3.npy'))
    out1 = np.load(os.path.join(npy_root_dir, 'out1.npy'))
    out2 = np.load(os.path.join(npy_root_dir, 'out2.npy'))
    out3 = np.load(os.path.join(npy_root_dir, 'out3.npy'))

    dW1, db1, dW2, db2, dW3, db3, dE_dnet3, dE_dout2, dE_dnet2, dE_dnet1 = \
        backward(X_batch, y_batch, W2, W3, out1, out2, out3)

    opt_dW1 = np.load(os.path.join(npy_root_dir, 'dW1.npy'))
    opt_db1 = np.load(os.path.join(npy_root_dir, 'db1.npy'))
    opt_dW2 = np.load(os.path.join(npy_root_dir, 'dW2.npy'))
    opt_db2 = np.load(os.path.join(npy_root_dir, 'db2.npy'))
    opt_dW3 = np.load(os.path.join(npy_root_dir, 'dW3.npy'))
    opt_db3 = np.load(os.path.join(npy_root_dir, 'db3.npy'))
    opt_dE_dnet3 = np.load(os.path.join(npy_root_dir, 'dE_dnet3.npy'))
    opt_dE_dout2 = np.load(os.path.join(npy_root_dir, 'dE_dout2.npy'))
    opt_dE_dnet2 = np.load(os.path.join(npy_root_dir, 'dE_dnet2.npy'))
    opt_dE_dnet1 = np.load(os.path.join(npy_root_dir, 'dE_dnet1.npy'))

    check(dW1, opt_dW1, 'dW1')
    check(db1, opt_db1, 'db1')
    check(dW2, opt_dW2, 'dW2')
    check(db2, opt_db2, 'db2')
    check(dW3, opt_dW3, 'dW3')
    check(db3, opt_db3, 'db3')
    check(dE_dnet3, opt_dE_dnet3, 'dE_dnet3')
    check(dE_dout2, opt_dE_dout2, 'dE_dout2')
    check(dE_dnet2, opt_dE_dnet2, 'dE_dnet2')
    check(dE_dnet1, opt_dE_dnet1, 'dE_dnet1')
