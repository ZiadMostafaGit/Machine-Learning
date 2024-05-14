import numpy as np


def cross_entropy_batch(y_true, y_pred):
    ...     # TODO


if __name__ == '__main__':
    # One-hot encoded true labels
    y_true = np.array([[1, 0],
                       [0, 1],
                       [0.8, 0.2]])     # soft labels for last example

    # Predicted probabilities
    y_pred = np.array([[0.9, 0.1],
                       [0.2, 0.8],
                       [0.7, 0.3]])

    your_answer = cross_entropy_batch(y_true, y_pred)
    right_answer = 0.284879527662735

    if np.isclose(your_answer, right_answer, atol=1e-6):
        print("Good job")
    else:
        print("Something wrong")
        print("your answer")
        print(your_answer)
        print("Optimal answer")
        print(right_answer)
