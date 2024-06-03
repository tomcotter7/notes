# Linear Algebra

## Chain Rule & Linear Regression

If $h(x) = f(g(x))$, then $h'(x) = f'(g(x)) \cdot g'(x)$.

This is used with linear regression to calculate the gradient of the loss function with respect to the parameters.

Let's say our loss function is $Residual^2 = (Observed - Predicted)^2$, and we are using linear regression to predict the value of $y$ given $x$.

We can write this as $y = \beta_0 + \beta_1x$.

Therefore, our loss function is $Residual^2 = (Observed - (\beta_0 + \beta_1x))^2$.

However, we can also write $Residual^2 = (Inside)^2$, which means we can use the chain chain as $f(g(x)) = (Inside)^2$ and $g(x) = (Observed - (\beta_0 + \beta_1x))$.

Therefore, $\frac{\partial Residual^2}{\partial \beta_1} = \frac{\partial Residual^2}{\partial Inside} \cdot \frac{\partial Inside}{\partial \beta_1}$.

We can do the same with $\beta_0$.

## Vectors, Matrices & Tensors

### Vectors

A vector is an array of numbers, $v = [v_1, v_2, ..., v_n]$. If each element of the vector is in $\mathbb{R}$, and the vector has $n$ elements, then the vector is in $\mathbb{R}^n$. This set is form by taking the Cartesian product of $\mathbb{R}$ with itself $n$ times. We can use the $-$ sign to denote the complement of a set, therefore $x_{-1}$ is the vector containing all elements of $x$ except for $x_1$.

### Matrices & Tensors

A matrix is a 2-d array of numbers. If a real-valued matrix **A** has a height of *m* and a width of *n* (i.e *m* rows and *n* columns), then we say that $A \in \mathbb{R}^{m x n}$.
    - Within a matrix, we can identify the *i-th row of A* by $A_{i,:}$ and the *i-th column* by $A_{:,i}$.

Matrices are a special version of a **tensor**, which is just an array of numbers arranged on a regular grid with a variable number of axis.

The tranpose of a matrix is the mirror image of a matrix across the diagonal. It can be defined as $(A^{T})_{i,j} = A_{j,i}$.

Adding matrices is possible if they have the same shape, and is done element-wise, i.e $C = A + B$ implies $C_{i,j} = A_{i,j} + B_{i,j}$. Adding a scalar or multiplying by a scalar is also done element-wise, i.e $D = a \cdot B + c$ implies $D_{i,j} = a \cdot B_{i,j} + c$.

We also have the addition of a matrix and a vector, which yields another matrix. $C = A + b$ implies $C_{i,j} = A_{i,j} + b_j$. In other words, the vector b (a column vector) is added to each row of the matrix A.

## Multiplying Matrices & Vectors

In order for a matrix product to be defined, A must have the same number of columns as B has rows. If A is of shape $m x n$ and B is of shape $n x p$, then the product of A and B is of shape $m x p$. The product is defined as $C_{i,j} = \sum_{k} A_{i,k}B_{k,j}$. This is because $k$ is the same for both A and B. The dot product between two vectors *x* and *y* is the matrix product $x^{T}y$. Let's define that:

- x (column vector) = $[1 2 3]$
- y (column vector) = $[4 5 6]$
- x^{T} (row vector) = $[1 2 3]$
- $Z_{1,1} = \sum_{k} x^{T}_{1,k}y_{k,1} = 1 \cdot 4 + 2 \cdot 5 + 3 \cdot 6$
- $Z_{1,1} = 32$
- $Z_{2,1}$ and $Z_{1, 2}$ don't exist because the vectors are 1 column. Therefore, the dot product is 32.

Observe that we can think of the calculation for $C_{i,j}$ as computing the dot product between the *i-th row of A* and the *j-th column of B* (which are both vectors).

The standard product of two matrices is not jsut a matrix containing the products of the individual elements. This exists and is called the **Hadamard product**. It is denoted by $A \odot B$.
