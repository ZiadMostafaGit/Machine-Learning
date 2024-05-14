import numpy as np
from sklearn.metrics import average_precision_score

def average_precision_score_our(ground_truth, scores):
    # Sort the ground truth in descending order of scores
    sorted_indices  = np.argsort(scores)[::-1]
    sorted_ground_truth = [ground_truth[i] for i in sorted_indices]
    scores = [scores[i] for i in sorted_indices]
    print(scores)
    print(sorted_ground_truth)
    precision_sum, true_positives = 0, 0

    for k in range(len(ground_truth)): # tip: bug to miss index = 0
        if sorted_ground_truth[k]:  # tp
            true_positives += 1
            pr = true_positives / (k + 1)
            print(k, pr)
            precision_sum += pr

    return precision_sum / sum(sorted_ground_truth)


if __name__ == '__main__':
    scores =       [0.8, 0.6, 0.3, 0.2, 0.9, 0.75, 0.81, 0.92]  # avoid duplicates
    ground_truth = [1  , 0  , 0  , 1  , 0  , 0   , 1    , 1]


    print(average_precision_score_our(ground_truth, scores))
    print(average_precision_score(ground_truth, scores))
