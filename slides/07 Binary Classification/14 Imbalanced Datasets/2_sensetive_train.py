from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification
from collections import Counter


if __name__ == '__main__':

    X, y = make_classification(n_samples=5000, n_features=5,
                               n_informative=2, n_redundant=3,
                               n_clusters_per_class=1, weights=[0.01], random_state=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    cnter = Counter(y_train)
    ir = cnter[0] / cnter[1]    # 0.016

    model = LogisticRegression(solver='lbfgs', class_weight={0:1,1:ir})
    model.fit(X_train, y_train)

    # or for grid search
    parameters = {'model__class_weight':
                      [{1: w} for w in [0.01, 0.05, 0.1, 0.5, 1]]}

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_test_prop = model.predict_proba(X_test)[:, 1]

    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    print('Training accuracy: %.4f' % accuracy_train)
    print('Test accuracy:     %.4f' % accuracy_test)

    report_train = classification_report(y_train, y_pred_train)
    report_test = classification_report(y_test, y_pred_test)
    print('Training\n%s' % report_train)
    print('Testing\n%s' % report_test)
