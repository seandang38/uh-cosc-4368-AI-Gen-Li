"""
Public tests for COSC 4368 Homework 1.

Usage:
1. Rename your completed starter file to hw1.py
2. Put this file in the same folder.
3. Run:

    python hw1_public_tests.py

Passing the public tests does NOT guarantee a perfect score.
The instructor may use additional test cases.
"""

import numpy as np
import hw1


def check(name, condition):
    if condition:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name}")


# A1
x = np.array([1.0, 2.0, 3.0])
expected = np.array([3.0, 5.0, 7.0])
check(
    "linear_predict",
    np.allclose(hw1.linear_predict(x, 2.0, 1.0), expected),
)

# A2
y_true = np.array([1.0, 2.0, 3.0])
y_pred = np.array([1.0, 2.0, 5.0])
check(
    "mse_loss",
    np.isclose(hw1.mse_loss(y_true, y_pred), 4.0 / 3.0),
)

# B1
check("sigmoid(0)", np.isclose(hw1.sigmoid(0.0), 0.5))

# B2
p = hw1.logistic_predict_proba(
    np.array([0.0, 1.0]),
    w=1.0,
    b=0.0,
)
check(
    "logistic_predict_proba",
    np.allclose(p, np.array([0.5, 1.0 / (1.0 + np.exp(-1.0))])),
)

# B3
pred = hw1.logistic_predict(
    np.array([-1.0, 0.0, 1.0]),
    w=1.0,
    b=0.0,
    threshold=0.5,
)
check(
    "logistic_predict",
    np.array_equal(pred, np.array([0, 1, 1])),
)

# B7
check(
    "accuracy",
    np.isclose(
        hw1.accuracy(
            np.array([0, 1, 1, 0]),
            np.array([0, 1, 0, 0]),
        ),
        0.75,
    ),
)

print("\nPublic tests finished.")
