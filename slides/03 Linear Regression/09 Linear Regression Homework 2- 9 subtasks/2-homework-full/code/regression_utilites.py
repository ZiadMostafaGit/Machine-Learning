import numpy as np

def visualize_cost(cost_history):
    import matplotlib.pyplot as plt

    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.grid()

    plt.plot(list(range(len(cost_history))), cost_history, '-r')

    plt.show()

def do_predictions(X, t, optimal_weights):
    examples, _ = X.shape
    pred = np.dot(X, optimal_weights)

    error = pred - t
    cost = np.sum(error ** 2) / (2 * examples)
    print(f'Cost function is {cost}')


def column_investigation(data):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.pairplot(data, x_vars=['Feat1', 'Feat2', 'Feat3'], y_vars='Target', height=4, aspect=1, kind='scatter')


    plt.show()
