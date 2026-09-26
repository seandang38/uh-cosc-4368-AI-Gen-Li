# COSC 4368 — Homework 1: Regression from Scratch

**Total:** 100 points  
**Due:** 9.27  
**Submission:** Submit one file named `hw1.py` on Canvas.

## Overview

In this homework, you will implement the core components of **linear regression** and **logistic regression** using NumPy.

The goal is to connect the mathematical ideas discussed in class with code. You will implement prediction functions, loss functions, gradients, gradient descent, and classification decisions.

You do **not** need to build a large software project. Complete only the functions marked `TODO` in the provided starter file.

## Learning Objectives

After completing this assignment, you should be able to:

1. Compute predictions for linear regression.
2. Compute mean squared error (MSE).
3. Compute gradients for linear regression.
4. Train a linear regression model using gradient descent.
5. Implement the sigmoid function.
6. Convert logistic regression scores into probabilities and class predictions.
7. Compute binary cross-entropy (BCE) loss.
8. Compute gradients for logistic regression.
9. Train a logistic regression model using gradient descent.
10. Evaluate binary classification accuracy.

## Allowed Libraries

You may use:

- Python standard library
- NumPy

You may **not** use machine-learning libraries to implement the models, including:

- scikit-learn
- PyTorch
- TensorFlow
- Keras
- statsmodels

The purpose of this homework is to implement the basic regression algorithms yourself.

## Important Rules

1. Do **not** change any required function name.
2. Do **not** change function parameters.
3. Do **not** change the required return values.
4. Do **not** remove provided code unless instructed.
5. Your submitted file must be named exactly `hw1.py`.
6. Your code must run with Python 3 and NumPy.
7. Do not hard-code answers for the provided datasets. Your functions will also be tested on additional inputs.
8. Follow the course policy regarding collaboration and outside assistance.

---

# Part A — Linear Regression (40 points)

For one input feature, linear regression predicts

\[
$\hat{y}$ = wx + b
\]

where:

- \(x\) is the input,
- \(w\) is the weight,
- \(b\) is the bias,
- $(\hat{y})$ is the prediction.

## A1. Linear Prediction — 5 points

Complete:

```python
def linear_predict(x, w, b):
```

Return the predicted values:

\[
$\hat{y}$ = wx+b
\]

### Example

```python
x = np.array([1.0, 2.0, 3.0])
linear_predict(x, 2.0, 1.0)
```

Expected result:

```text
[3.0, 5.0, 7.0]
```

---

## A2. Mean Squared Error — 10 points

Complete:

```python
def mse_loss(y_true, y_pred):
```

Use

\[
MSE=$\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2$
\]

Return one floating-point number.

### Example

```python
y_true = np.array([1.0, 2.0, 3.0])
y_pred = np.array([1.0, 2.0, 5.0])
```

The MSE is:

\[
$\frac{0^2+0^2+(-2)^2}{3}=\frac{4}{3}$
\]

---

## A3. Linear Regression Gradients — 10 points

Complete:

```python
def linear_gradients(x, y, w, b):
```

First compute

\[
$\hat y_i$ = $wx_i$+b
\]

Then compute

\[
$\frac{\partial L}{\partial w}$=
$\frac{2}{n}\sum_{i=1}^{n}x_i(\hat y_i-y_i)$
\]

and

\[
$\frac{\partial L}{\partial b}$= $\frac{2}{n}\sum_{i=1}^{n}(\hat y_i-y_i)$
\]

Return:

```python
dw, db
```

---

## A4. Train Linear Regression — 15 points

Complete:

```python
def train_linear_regression(x, y, learning_rate, epochs):
```

Initialize:

```python
w = 0.0
b = 0.0
```

For each epoch:

1. Compute `dw` and `db`.
2. Update

\[
$w \leftarrow w-\alpha dw$
\]

\[
$b \leftarrow b-\alpha db$
\]

where \(\alpha\) is the learning rate.

Return:

```python
w, b
```

Your function should work for datasets other than the example below.

### Practice Dataset

```python
x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([3, 5, 7, 9, 11], dtype=float)
```

This dataset approximately follows

\[
y=2x+1
\]

Training should therefore produce parameters close to:

```text
w = 2
b = 1
```

---

# Part B — Logistic Regression (50 points)

Logistic regression first computes

\[
z=wx+b
\]

and converts the score into a probability using the sigmoid function:

\[
$\sigma(z)=\frac{1}{1+e^{-z}}$
\]

---

## B1. Sigmoid — 5 points

Complete:

```python
def sigmoid(z):
```

The function must work with both:

- a single number,
- a NumPy array.

For example:

```python
sigmoid(0)
```

should return approximately:

```text
0.5
```

---

## B2. Probability Prediction — 5 points

Complete:

```python
def logistic_predict_proba(x, w, b):
```

Compute:

1. \(z=wx+b\)
2. \(p=$\sigma(z)$\)

Return the probability that each example belongs to class 1.

---

## B3. Binary Prediction — 5 points

Complete:

```python
def logistic_predict(x, w, b, threshold=0.5):
```

Use the probabilities from `logistic_predict_proba`.

For each probability \(p\):

- predict `1` if \(p \geq\) threshold,
- predict `0` otherwise.

Return a NumPy array containing integers `0` and `1`.

---

## B4. Binary Cross-Entropy Loss — 10 points

Complete:

```python
def binary_cross_entropy(y_true, y_prob):
```

Use:

\[
BCE=
$-\frac{1}{n}
\sum_{i=1}^{n}
\left[
y_i\log(p_i)
+
(1-y_i)\log(1-p_i)
\right]$
\]

To avoid taking `log(0)`, clip probabilities before computing the logarithm:

```python
eps = 1e-12
y_prob = np.clip(y_prob, eps, 1 - eps)
```

Return one floating-point number.

---

## B5. Logistic Regression Gradients — 10 points

Complete:

```python
def logistic_gradients(x, y, w, b):
```

First compute:

\[
$p_i$=$\sigma(wx_i+b)$
\]

Then:

\[
$\frac{\partial L}{\partial w}$=
$\frac{1}{n}
\sum_{i=1}^{n}
x_i(p_i-y_i)$
\]

\[
$\frac{\partial L}{\partial b}$=
$\frac{1}{n}
\sum_{i=1}^{n}
(p_i-y_i)$
\]

Return:

```python
dw, db
```

---

## B6. Train Logistic Regression — 10 points

Complete:

```python
def train_logistic_regression(x, y, learning_rate, epochs):
```

Initialize:

```python
w = 0.0
b = 0.0
```

For each epoch:

1. Compute the logistic regression gradients.
2. Update `w`.
3. Update `b`.

Return:

```python
w, b
```

### Practice Dataset

Suppose we want to predict whether a student passes an exam based on hours studied.

```python
hours_studied = np.array(
    [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    dtype=float,
)

passed = np.array(
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    dtype=float,
)
```

For testing, you may use:

```python
w, b = train_logistic_regression(
    hours_studied,
    passed,
    learning_rate=0.1,
    epochs=2000,
)
```

The resulting model should correctly classify most or all of these training examples.

---

## B7. Accuracy — 5 points

Complete:

```python
def accuracy(y_true, y_pred):
```

Accuracy is:

\[
Accuracy=
\frac{\text{number of correct predictions}}
{\text{total number of predictions}}
\]

Return a value between `0.0` and `1.0`.

---

# Part C — Concept Check (10 points)

Enter your answers in the two variables at the bottom of `hw1.py`.

## C1 — 5 points

A logistic regression model outputs a probability of `0.73` for one example.

What does `0.73` represent?

**A.** The predicted class must be 0  
**B.** The model's estimated probability that the example belongs to class 1  
**C.** The MSE loss  
**D.** The model weight  

Set:

```python
answer_q1 = "A"
```

or `"B"`, `"C"`, or `"D"`.

---

## C2 — 5 points

Suppose the model probabilities stay unchanged, but the classification threshold is increased from `0.5` to `0.8`.

What is most likely to happen?

**A.** More examples will be classified as class 1  
**B.** Fewer examples will be classified as class 1  
**C.** All predicted probabilities will decrease  
**D.** The learned values of `w` and `b` will automatically change  

Set:

```python
answer_q2 = "A"
```

or `"B"`, `"C"`, or `"D"`.

---

# Grading

| Item | Points |
|---|---:|
| A1. `linear_predict` | 5 |
| A2. `mse_loss` | 10 |
| A3. `linear_gradients` | 10 |
| A4. `train_linear_regression` | 15 |
| B1. `sigmoid` | 5 |
| B2. `logistic_predict_proba` | 5 |
| B3. `logistic_predict` | 5 |
| B4. `binary_cross_entropy` | 10 |
| B5. `logistic_gradients` | 10 |
| B6. `train_logistic_regression` | 10 |
| B7. `accuracy` | 5 |
| C1–C2. Concept questions | 10 |
| **Total** | **100** |

## Grading Notes

- Correctness is the primary grading criterion.
- The autograder may test your functions using inputs different from the examples in this handout.
- Small floating-point differences are acceptable.
- A function that only works for the provided practice dataset will not receive full credit.
- Code that imports a prohibited machine-learning library may receive no credit for the corresponding implementation.

---

# Before You Submit

Make sure:

- [ ] Your file is named `hw1.py`.
- [ ] All required functions are implemented.
- [ ] You did not rename any function.
- [ ] You did not change function parameters.
- [ ] You did not use prohibited machine-learning libraries.
- [ ] Your file runs without errors.
- [ ] You completed `answer_q1` and `answer_q2`.
- [ ] You ran the provided public tests.
