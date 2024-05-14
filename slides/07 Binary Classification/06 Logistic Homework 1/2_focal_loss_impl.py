# ChatGPT
import numpy as np


def focal_loss_with_class_weight(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Compute focal loss for binary classification problem.

    y_true: array-like, true labels; shape should be (n_samples,)
    y_pred: array-like, predicted probabilities for being in the
        positive class; shape should be (n_samples,)
    alpha: float, weight for the positive class to improve further
    gamma: float, focusing parameter for focal loss
    """

    # Ensure the prediction is within the range [eps, 1-eps] to avoid log(0)
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1.0 - eps)

    # Compute focal loss
    pos_loss = -alpha * np.power(1 - y_pred, gamma) * np.log(y_pred)
    neg_loss = -(1 - alpha) * np.power(y_pred, gamma) * np.log(1 - y_pred)

    # Combine losses
    focal_loss = np.where(y_true == 1, pos_loss, neg_loss)

    return np.mean(focal_loss)


# Test the function
y_true = np.array([0, 1, 1, 0, 1])  # true labels
y_pred = np.array([0.1, 0.8, 0.9, 0.3, 0.7])  # predicted probabilities for the positive class

loss_value = focal_loss_with_class_weight(y_true, y_pred, alpha=0.25, gamma=2)
print(f"Focal Loss: {loss_value}")
