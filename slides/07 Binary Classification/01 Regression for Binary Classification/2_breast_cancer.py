# adapted from: https://github.com/artemmavrin/logitboost/blob/5b28555/docs/examples/Breast_Cancer.ipynb

import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPRegressor


def visualize(X_train, y_train):
    from sklearn.manifold import TSNE

    # Map N features to 2 features so that you can visualize
    tsne = TSNE(n_components=2, random_state=0)
    X_train_tsne = tsne.fit_transform(X_train)

    plt.figure(figsize=(10, 8))
    mask_benign = (y_train == 0)     # 'benign'
    mask_malignant = (y_train == 1)  # 'malignant'

    plt.scatter(X_train_tsne[mask_benign, 0], X_train_tsne[mask_benign, 1],
                marker='s', c='g', label='benign', edgecolor='k', alpha=0.7)
    plt.scatter(X_train_tsne[mask_malignant, 0], X_train_tsne[mask_malignant, 1],
                marker='o', c='r', label='malignant', edgecolor='k', alpha=0.7)

    plt.title('t-SNE plot of the training data')
    plt.xlabel('1st embedding axis')
    plt.ylabel('2nd embedding axis')
    plt.legend(loc='best', frameon=True, shadow=True)

    plt.tight_layout()
    plt.show()
    plt.close()


def do_classification(X_train, X_test, y_train, y_test):
    model = LogisticRegression(solver = 'lbfgs')
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    print('Training accuracy: %.4f' % accuracy_train)
    print('Test accuracy:     %.4f' % accuracy_test)

    report_train = classification_report(y_train, y_pred_train)
    report_test = classification_report(y_test, y_pred_test)
    print('Training\n%s' % report_train)
    print('Testing\n%s' % report_test)


def do_linear_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    y_pred_train = (y_pred_train > 0.5).astype(int)
    y_pred_test = (y_pred_test > 0.5).astype(int)

    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    print('Training accuracy: %.4f' % accuracy_train)
    print('Test accuracy:     %.4f' % accuracy_test)

    report_train = classification_report(y_train, y_pred_train)
    report_test = classification_report(y_test, y_pred_test)
    #print('Training\n%s' % report_train)
    #print('Testing\n%s' % report_test)


def do_neuralnetwork(X_train, X_test, y_train, y_test):
    # tricky to find good values
    hidden_layer_sizes = (20, 10, 5)
    model = MLPRegressor(hidden_layer_sizes, random_state=10, max_iter=10000)
    model.fit(X_train, y_train.reshape(-1))

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    y_pred_train = (y_pred_train > 0.5).astype(int)
    y_pred_test = (y_pred_test > 0.5).astype(int)

    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    print('Training accuracy: %.4f' % accuracy_train)
    print('Test accuracy:     %.4f' % accuracy_test)

    report_train = classification_report(y_train, y_pred_train)
    report_test = classification_report(y_test, y_pred_test)
    #print('Training\n%s' % report_train)
    #print('Testing\n%s' % report_test)


if __name__ == '__main__':
    data = load_breast_cancer()
    X = data.data
    y = data.target_names[data.target]
    n_classes = data.target.size

    y = (y == 'malignant').astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        shuffle=True, stratify=y,
                                                        random_state=0)


    visualize(X_train, y_train)
    do_classification(X_train, X_test, y_train, y_test)
    #do_linear_regression(X_train, X_test, y_train, y_test)
    #do_neuralnetwork(X_train, X_test, y_train, y_test)




