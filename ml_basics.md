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



