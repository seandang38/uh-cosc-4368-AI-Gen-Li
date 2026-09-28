"""
COSC 4368
Homework 1: Regression from Scratch

Name: Sean Dang 
CougarNet ID: 2518173

Instructions:
- Complete every TODO.
- Do not change function names, parameters, or return values.
- NumPy is allowed.
- Do not use scikit-learn, PyTorch, TensorFlow, Keras, or statsmodels.
"""

import numpy as np


# ============================================================
# Part A: Linear Regression
# ============================================================

def linear_predict(x, w, b):
    """
    Compute linear regression predictions.

    Parameters
    ----------
    x : np.ndarray
        One-dimensional input array.
    w : float
        Model weight.
    b : float
        Model bias.

    Returns
    -------
    np.ndarray
        Predictions w*x + b.
    """
    return w * x + b


def mse_loss(y_true, y_pred):
    """
    Compute mean squared error.

    Returns
    -------
    float
        Mean squared error.
    """
    return float(np.mean((y_true - y_pred) ** 2))


def linear_gradients(x, y, w, b):
    """
    Compute gradients of MSE with respect to w and b.

    Returns
    -------
    dw : float
    db : float
    """
    n = len(x)
    y_hat = linear_predict(x, w, b)
    diff = y_hat - y
    dw = (2 / n) * np.sum(diff * x)
    db = (2 / n) * np.sum(diff)

    return float(dw), float(db)

def train_linear_regression(x, y, learning_rate, epochs):
    """
    Train one-feature linear regression using gradient descent.

    Initialize w = 0.0 and b = 0.0.

    Returns
    -------
    w : float
    b : float
    """
    w = 0.0
    b = 0.0

    for _ in range(epochs):
        dw, db = linear_gradients(x, y, w, b)
        w = w - learning_rate * dw
        b = b - learning_rate * db

    return w, b


# ============================================================
# Part B: Logistic Regression
# ============================================================

def sigmoid(z):
    """
    Compute the sigmoid function.

    Must work for both a scalar and a NumPy array.
    """
    return 1.0 / (1.0 + np.exp(-z))


def logistic_predict_proba(x, w, b):
    """
    Return P(y=1 | x) for each input.
    """
    return sigmoid(w * x + b)


def logistic_predict(x, w, b, threshold=0.5):
    """
    Convert logistic-regression probabilities into class labels.

    Predict 1 when probability >= threshold, otherwise 0.

    Returns
    -------
    np.ndarray
        Integer array containing 0 and 1.
    """

    probs = logistic_predict_proba(x, w, b)
    return (probs >= threshold).astype(int)


def binary_cross_entropy(y_true, y_prob):
    """
    Compute binary cross-entropy loss.

    Hint:
        Clip y_prob to [1e-12, 1 - 1e-12] before taking logs.
    """
    eps = 1e-12
    y_prob = np.clip(y_prob, eps, 1.0 - eps)
    bce_loss = -np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob))

    return float(bce_loss)


def logistic_gradients(x, y, w, b):
    """
    Compute logistic-regression gradients for BCE loss.

    Returns
    -------
    dw : float
    db : float
    """
    n = len(x)
    p = logistic_predict_proba(x, w, b)
    diff = p - y
    dw = (1.0 / n) * np.sum(diff * x)
    db = (1.0 / n) * np.sum(diff)

    return float(dw), float(db)


def train_logistic_regression(x, y, learning_rate, epochs):
    """
    Train one-feature logistic regression using gradient descent.

    Initialize w = 0.0 and b = 0.0.

    Returns
    -------
    w : float
    b : float
    """
    w = 0.0
    b = 0.0

    for _ in range(epochs):
        dw, db = logistic_gradients(x, y, w, b)
        w = w - learning_rate * dw
        b = b - learning_rate * db

    return w, b


def accuracy(y_true, y_pred):
    """
    Compute classification accuracy.

    Returns
    -------
    float
        Value between 0.0 and 1.0.
    """
    return float(np.mean(y_true == y_pred))


# ============================================================
# Part C: Concept Check
# ============================================================

# Replace each empty string with A, B, C, or D.
answer_q1 = "B"
answer_q2 = "B"


# ============================================================
# Optional local practice
# ============================================================

if __name__ == "__main__":
    # You may use this section to test your own code.
    # The autograder will call the functions above directly.

    x_linear = np.array([1, 2, 3, 4, 5], dtype=float)
    y_linear = np.array([3, 5, 7, 9, 11], dtype=float)

    x_logistic = np.array(
        [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        dtype=float,
    )
    y_logistic = np.array(
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        dtype=float,
    )

    print("Use this section for your own experiments.")
