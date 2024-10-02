# Numerical Computation - Chapter 4 of Ian Goodfellow's Deep Learning

## Overflow and Underflow

Underflow occurs when numbers near zero are rounded to zero. Overflow occurs when numbers with large magnitude are approximated as infinity or negative infinity. 

We need to stabilize the softmax function to avoid overflow and underflow. The softmax function is defined as follows:

$softmax(x)_i = \frac{exp(x_i)}{\sum_{j=1}^{n}exp(x_j)}$

When all $x_i$ are some constant $c$, the output is $\frac{1}{n}$. If $c$ is very negative, $exp(x_j)$ will underflow, resulting in the denominator becoming zero and the output being nan. We can stabilize the softmax function by evalutiong $softmax(z)$ where $z = x - max_i(x_i)$.

## Poor Conditioning

Poorly conditioned functions are function that change a lot even when the input changes a little.

## Gradient-Based Optimization

The partial derivation $\frac{\partial}{\partial x_i}f(x)$ measures how $f$ changes as only the variable $x_i$ increases at point x. The gradient of $f$ is the vector containing all the partial derivatives , denoted as $\nabla_xf(x)$.

The directional derivative in a direction $u$ (a unit vector) is the slope of the function $f$ in the direction $u$. In other words, the directional derivative is the derivative of the function $f(x + \epsilon u)$ with respect to $\epsilon$ at $\epsilon = 0$. Using the chain rule, we can see that $\frac{\partial}{\partial \epsilon}f(x + \epsilon u) evaluates to $u^{T}\nabla_xf(x)$, when $\epsilon = 0$.

Chain rule $f(g(x)) = f'(g(x))g'(x)$, in this case $f(x)$ is just $f$ (i.e. f'(x) is $\nabla_xf(x)$) and $g(x)$ is $x + \epsilon u$ (i.e. g'(x) is $u$).

To minimize $f$, we would like to find the direction in which $f$ decreases the fastest. We can do this using the directional derivative:

$min_{u, u^{T}u = 1}u^{T}\nabla_xf(x)$ = $min_{u, u^{T}u = 1}||u||_2||\nabla_xf(x)||_2cos(\theta)$

where $\theta$ is the angle between $u$ and the gradient. Since $u$ is a unit vector, $||u||_2 = 1$. Therefore, this simplifies to: $min_u cos(\theta)$. This is minimized when $u$ points in the opposite direction of the gradient because $cos(180) = -1$.

### Beyond the Gradient: Jacobian and Hessian Matrices

Sometimes, we need to find all the partial derivatives of a function whose input and output are both vectors. The matrix containing all such partial derivatives is known as a Jacobian matrix. Specifically if we have a function $f: \mathbb{R}^m \rightarrow \mathbb{R}^n$, the Jacobian matrix $J \in \mathbb{R}^{n \times m}$ is defined such that $J_{i,j} = \frac{\partial}{\partial x_j}f_i(x)$.

We also sometimes are interested in a derivative of a derivative. For example, for a function $f: \mathbb{R}^n \rightarrow \mathbb{R}$, the derivative with respect to $x_i$ of the derivative of $f$ with respect to $x_j$ is denoted as $\frac{\partial^2}{\partial x_i \partial x_j}f(x)$.

We can think of the second derivative as measuring curvature. If a function has a second derivative of 0, then there is no curvature. It is a perfectly flat line, and it's value can be predicted using only the gradient. If the second deriviative is negative, the function curves downward, so the function which actually decrease by more than the step size. If the second derivative is positive, the function curves upward, so the function will decrease by less than the step size.

The Hessian matrix is the Jacobian matrix of the gradient. $H_{i, j} = H_{j, i}$, i.e the Hessian matrix is symmetric when the second partial derivatives are continuous (as the operators are commutative).

Optimization algorithms that only use the first derivative are called first-order optimization algorithms. Optimization algorithms that use the second derivative are called second-order optimization algorithms.

Again, even in multiple dimensions, we can check to see if a certain point is a minimum, maximum or saddle point by looking at the eigenvalues of the Hessian. If all the eigenvalues are positive, the point is a local minimum. If all the eigenvalues are negative, the point is a local maximum. If there is at least 1 positive and 1 negative eigenvalue, the point is a saddle point. 

## Constrained Optimization

Sometimes, we may wish to find the minimum value of f(x) for values in x in some set $S$. This is known as constrained optimization. The Karush-Kuhn-Tucker (KKT) provides a very general solution to constrained optimization. With the KKT we introduce a new function called the generalized Lagrangian function.

To define the langrangian, we need to describe $S$ in terms of equations and inequalities. We want a descriptions of $S$ in terms of $m$ functions $g^{i}$ and $n$ functions $h^{j}$ such that $S = \{x | g^{i}(x) = 0, h^{j}(x) \leq 0\}$.

The generalized Lagrangian function is defined as:

$L(x, \alpha, \beta) = f(x) + \sum_{i=1}^{m}\alpha_{i}g^{i}(x) + \sum_{j=1}^{n}\beta_{j}h^{j}(x)$

we can now solve the constrained optimization problem using unconstrained optimization of the generalized langrangian.

The KKT conditions are:
    - The gradient of the generalized Lagrangian with respect to x is 0.
    - All constraints are satisfied.
    - The inequality constrains exhibit complementary slackness, i.e. $\beta_{j}h^{j}(x) = 0$.
