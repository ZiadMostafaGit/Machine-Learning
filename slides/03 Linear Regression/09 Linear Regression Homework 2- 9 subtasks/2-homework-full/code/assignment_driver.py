import argparse
import numpy as np

from gradient_descent_linear_reg import gradient_descent_linear_regression
from regression_utilites import *
from data_helper import *
from normal_eq import normal_equations_solution

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Linear Regression Homework')

    parser.add_argument('--dataset', type=str, default='dataset_200x4_regression.csv')

    parser.add_argument('--preprocessing', type=int, default=1,       # p4
                        help='0 for no processing, '
                             '1 for min/max scaling and '
                             '2 for standrizing')

    parser.add_argument('--choice', type=int, default=2,
                        help='0 for linear verification, '            # p0
                             '1 for training with all features, '     # p1 / p3 / p7
                             '2 for training with the best feature, ' # p5
                             '3 for normal equations, '               # p6
                             '4 for sikit)')

    # Use below to explore for p2
    parser.add_argument('--step_size', type=float, default=0.01, help='Learning rate (default: 0.01)')
    parser.add_argument('--precision', type=float, default=0.0001, help='Requested precision (default: 0.0001)')
    parser.add_argument('--max_iter', type=int, default=10000, help='number of epochs to train (default: 1000)')

    args = parser.parse_args()

    #np.random.seed(0)
    # When we are done, we might use random seed 0 to FIX the randomizations in our program

    if args.choice == 0:
        # Perfect line: 45 degree
        X = np.array([0, 0.2, 0.4, 0.8, 1.0])
        x_vis = X
        t = 5 + X
        X = X.reshape((-1, 1))
    else:
        df, data, X, t = load_data(args.dataset, args.preprocessing)
        examples, features = X.shape

    # add dummy one for the intercept
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    if args.choice == 0:
        # p1 / p7
        print('Verify linear line')

        optimal_weights, cost_history = gradient_descent_linear_regression(X, t,
                                    step_size=0.1, precision = 0.00001, max_iter=1000)

        # Number of iterations ended at 695 - with cost 4.4770104998508613e-08 - optimal weights [4.99957534 1.00078798]

        # Compute predictions to visualize the line
        pred = np.dot(X, optimal_weights)

        # p3
        import matplotlib.pyplot as plt

        plt.scatter(x_vis, t, marker='o', color='red')     # plot input data points (x, t)
        plt.plot(x_vis, pred, color = 'blue')              # our predicted line (x, predicted_t)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()
    elif args.choice == 1:
        print('Learn using the 3 features')
        optimal_weights, cost_history = gradient_descent_linear_regression(X, t,
                                    step_size=args.step_size, precision = args.precision, max_iter=args.max_iter)

        do_predictions(X, t, optimal_weights)
        visualize_cost(cost_history)

        # Number of iterations ended at 2941 - with cost 0.0035358756746826903 - optimal weights [0.12690647 0.5644904  0.11481544 0.27794466]
        # Cost function is 0.0035348763195645234


    elif args.choice == 2:
        print('Learn using the 1st feature')

        X = X[:, :1]
        optimal_weights, cost_history = gradient_descent_linear_regression(X, t,
                               step_size=args.step_size, precision = args.precision, max_iter=args.max_iter)

        do_predictions(X, t, optimal_weights)

        column_investigation(df)

        # Number of iterations ended at 315 - with cost 0.021579007954330674 - optimal weights [0.54254992]
        # Cost function is 0.02157802236475603
    elif args.choice == 3:
        optimal_weights = normal_equations_solution(X, t)
        print(optimal_weights)

        do_predictions(X, t, optimal_weights)

        # For minmax scaled data
        # [0.12060381 0.6338432  0.20894728 0.00150253]
        # Cost function is 0.0020971589652777674

        # For NOT scaled data
        # [1.19099373e-01 2.14353466e-03 4.21264675e-03 1.32148928e-05]
        # Cost function is 0.002097158965277767
        # So, we don't need in theory to scale the data
        # however, it is still good for numerical stability
    elif args.choice == 4:
        from sklearn import linear_model
        from sklearn.metrics import mean_squared_error

        # LinearRegression from sklearn uses OLS (normal equations)
        # We can remove our added 1 column and use fit_intercept = True
        X_ = X[:, 1:]
        model = linear_model.LinearRegression(fit_intercept = True)

        # Train the model using the training sets
        model.fit(X_, t)
        optimal_weights = np.array([model.intercept_, *model.coef_])
        print(f'Scikit parameters: {optimal_weights}')

        # Make predictions using the testing set
        pred_t = model.predict(X_)
        # divide by 2 to MATCH our cost function
        error = mean_squared_error(t, pred_t) / 2
        print(f'Error: {error}')
        do_predictions(X, t, optimal_weights)

        # Sikit uses normal equations
        # Specifically use: numpy.linalg.lstsq

        # Using minmax scaled data
        # Scikit parameters: [0.12060381 0.6338432  0.20894728 0.00150253]
        # Error: 0.0020971589652777674
        # Cost function is 0.0020971589652777674

