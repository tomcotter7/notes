# Machine Learning Basics

## Learning Algorithms

ML tasks are usually described in terms of how the ML system should process an **example**. We typically represent an example as a vector $x \in \mathbb{R}^n$ where each entry $x_i$ of the vector is a feature of the example.

The error rate is the proportion of examples that the model misclassifies. Also known as the expected 0-1 loss.

## Linear Regression

We can define this a $y^{\hat{}} = w^Tx + b$, where $w \in \mathbb{R}^n$ is the weight vector and $y^{\hat{}}$ is the predicted value.

If a feature $x_i$ receives a positive weight $w_i$, then increasing the value of that feature will increase the predicted value.

One way of measuring model performance is to use the Mean Squared Error (MSE) loss function. This is defined as:

$ MSE_{test} = \frac{1}{m} \sum_{i=1}^{m} (y_i - y^{\hat{}}_i)^2 $

Where $m$ is the number of examples in the test set, $y_i$ is the true value of the $i$th example in the test, and $y^{\hat{}}_i$ is the predicted value of the $i$th example in the test set. Intuitively, we can see that this is 0 when the two values are equal. We can also see that this is the L2 norm of the difference between the true and predicted values.

To minimize the MSE, we can simple solve for where it's gradient with respect to the weights is 0.

$\nabla_w MSE = 0$

$\nabla_w \frac{1}{m} \sum_{i=1}^{m} (y^{\hat{}} - y)^2 = 0$

$\frac{1}{m} \nabla_w \sum_{i=1}^{m} (X^Tw - y)^2 = 0$

$\nabla_w (X^Tw - y)^T(X^Tw - y) = 0$

-- the following jump is a rule from matrix calculus --

$2X^T(X^Tw - y) = 0$

$w = (X^TX)^{-1}X^Ty$

Where $X$ is the matrix of examples, $y$ is the vector of true values, and $w$ is the vector of weights.

## Capacity, Overfitting, and Underfitting

When training ML models, we make the assumption that the training and test data are drawn from the same probability distribution as each other. We call this underlying distribution the **data generating distribution** or $p_{data}$.

Underfitting occurs when the model is not able to obtain a sufficiently low error value on the training set. Overfitting occurs when the gap between the training error and test error is too large.

We can adjust a models capacity in order to reduce the risk of (over|under)fitting. For example, we can generalise linear regression to include polynomials in its hypothesis space, in which doing so increases the models capacity.

## No Free Lunch Theorem

This argues that no machine learning algorithm is universally better than any other. This is because the performance of any algorithm depends on the data being used to train it. Therefore, the best algorithm for a given problem is the one that is best suited to the data.

## Regularization

The behavious of our algorithm is strongly affected by not just the number of functions in the hypothesis space, but by the specific identity of those functions. We can also give the algorithm for one function over another. For example, we can modify linear regression to use weight decay. Which means we minimize $J(w)$ to both have a small training error and for the weights to have a small L2 norm. Minimizing $J(w)$ results in a tradeoff between the two goals.

*Regularization is any modification we make to a learning algorithm that is intended to reduce its generalization error but not its training error.*


