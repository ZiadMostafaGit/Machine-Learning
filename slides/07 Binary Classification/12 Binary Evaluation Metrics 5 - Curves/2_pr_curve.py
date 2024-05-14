import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.datasets import make_classification


def visualize_pr(precision, recall, threshold, versus_threshold = True):
    if versus_threshold:   # threshold vs recall/precision
        plt.plot(threshold, precision, 'b--', label='Precision')
        plt.plot(threshold, recall, 'r--', label='Recall')
        plt.xlabel('Threshold')
        plt.legend(loc='lower left')
    else:
        plt.plot(recall, precision, 'r--')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # also plot_precision_recall_curve

    plt.show()


def pr_compute(precision, recall, threshold):
    # precision is increasing
    target_precision = 0.935
    best_threshold_idx = np.argmax(precision >= target_precision)
    best_threshold = threshold[best_threshold_idx]
    print(f'Threshold {best_threshold} has precision >= {target_precision} '
          f'has recall {recall[best_threshold_idx]}')
    # Threshold 0.8002828287526059 has precision >= 0.935 has recall 0.670020120724346

    # recall is decreasing
    target_recall = 0.82
    best_threshold_idx = np.argmin(recall >= target_recall)
    best_threshold = threshold[best_threshold_idx]
    print(f'Threshold {best_threshold} has recall >= {target_recall} '
          f'has precision {precision[best_threshold_idx]}')

    # Threshold 0.604938756634422 has recall >= 0.82 has precision 0.8790496760259179


def evalaute(y_gt, y_prop, n_classes):
    # The last precision and recall values are 1. and 0. respectively
    # and do not have a corresponding threshold
    precision, recall, threshold = precision_recall_curve(y_gt, y_prop)
    precision, recall = precision[:-1], recall[:-1]

    #visualize_pr(precision, recall, threshold, versus_threshold = True)
    #pr_compute(precision, recall, threshold)

    print(average_precision_score(y_gt, y_prop))    # popular approximation
    print(roc_auc_score(y_gt, y_prop))              # another approximation  [not fully similar)

    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html





if __name__ == '__main__':
    # create random binary classification task
    n_classes = 2
    X, y = make_classification(n_samples=5000, n_classes = n_classes, n_informative=4, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    y_prop = model.predict_proba(X_val)[:, 1]   # 2nd column: prob class 1

    evalaute(y_val, y_prop, n_classes)




