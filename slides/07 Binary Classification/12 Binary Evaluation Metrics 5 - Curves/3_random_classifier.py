import numpy as np
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, roc_auc_score


def eval(ground_truth, random_predictions):
    accuracy = accuracy_score(ground_truth, random_predictions)
    precision = precision_score(ground_truth, random_predictions)
    recall = recall_score(ground_truth, random_predictions)
    f1 = f1_score(ground_truth, random_predictions)

    aucpr = average_precision_score(ground_truth, random_predictions)
    aucroc = roc_auc_score(ground_truth, random_predictions)

    print(f"Accuracy: {accuracy:.2f} - Precision: {precision:.2f} - Recall: {recall:.2f} - "
          f"AUC-PR: {aucpr:.2f} - AUC-ROC: {aucroc:.2f} \n")


def get_gt(total, positives_percent = 0.5):
    # return array of ones and zeros for ground truth. Ones based on positives_percent
    ones = int(total * positives_percent)
    zeros = total - ones

    return np.concatenate([np.array([1] * ones), np.array([0] * zeros)])


if __name__ == '__main__':
    N = 10000
    random_predictions = [random.choice([0, 1]) for _ in range(N)]

    percent = [0.01, 0.25, 0.50, 0.75, 0.98]

    for p in percent:
        print(f'Positive examples are {p}%')
        eval(get_gt(N, p), random_predictions)


if __name__ == '__main__1':
    N = 10
    # Generate random predictions (assuming binary classification)


    ground_truth = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    eval(ground_truth, random_predictions)

    ground_truth = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    eval(ground_truth, random_predictions)

    ground_truth = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    eval(ground_truth, random_predictions)

    ground_truth = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    eval(ground_truth, random_predictions)

    ground_truth = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    eval(ground_truth, random_predictions)
