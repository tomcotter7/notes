# Linear Algebra - Chapter 2 of Ian Goodfellow's Deep Learning

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

## Identity & Inverse Matrices

An identity matrix is matrix that does not change any vector when we multiply that vector by that matrix. Formalluy, $I_{n} \in R^{n x n}$ and \forall x \in R^{n}, I_{n}x = x$.

The matrix inverse of $A$ is denoted as $A^{-1}$, $A^{-1}A = I_{n}$.

## Norms

We can measure the size of vectors using a **norm**. Formally, $L^{p}$ norm is given by: $||x||_{p} = (\sum_{i}|x_{i}|^{p})^{\frac{1}{p})$ for $p \in \mathbb{R}, p \geq 1$. On an intuitive level, the norm of a vector x measures the distance from the origin to the point x. The $L^{2}$ norm (the Euclidean norm), is often used, as well as the squared $L^{2}$ norm, which can be calculated as $x^{T}x$.

We also often use the $L^{1}$ norm, which essentially becomes an absolute sum of all the elements. This is useful when the difference between zero and non-zero elements is very important. *Every time an element of x moves away from 0 by* $\epsilon$, *the* $L^{1}$ *norm increases by* $\epsilon$.

We can also measure the size of a matrix using the Frobenius norm: $||A||_{F} = \sqrt{\sum_{i,j}A_{i,j}^{2}}$. This is analogous to the $L^{2}$ norm of a vector.

## Special Kinds of Matrices & Vectors

Diagonal matrices are diagonal if and only if $D_{i, j} = 0$ for all $i \neq j$.

A symmetric matrix is a matrix that is equal to its transpose: $A = A^{T}$.

A unit vector is a vector with a norm of 1.

A vector x is orthogonal to a vector y if $x^{T}y = 0$. If the vectors also have unit norms, then they are orthonormal.

An orthogonal matrix is a square matrix whose rows are mutually orthonormal and whose columns are mutually orthonormal.

## Eigendecomposition

An eigenvector of a square matrix A is a nonzero vector v such that multiplication by A alters only the scale of v: $Av = \lambda v$. Here, $\lambda$ is the eigenvalue corresponding to the eigenvector v.

Suppose that a matrix A has $n$ linearly independent eigenvectors, $v^{(1)}, ..., v^{(n)}$, and corresponding eigenvalues $\lambda_{1}, ..., \lambda_{n}$. We can concatenate the eigenvectors to create a matrix $V$ and the eigenvalues to create a vector $\lambda$. The eigendecomposition of A is then given by $A = V diag(\lambda)V^{-1}$.

Every real symmetric matrix can be decomposed into an expression using only real-valued eigenvectors and eigenvalues. $A = Q \Lambda Q^{T}$, where $Q$ is an orthogonal matrix composed of the eigenvectors of A, and $\Lambda$ is a diagonal matrix containing the eigenvalues of A.

## Singular Value Decomposition

If a matrix is not square, then eigendecomposition is not defined. However, we can still use a singular value decomposition (SVD). For any $m x n$ matrix A, there exists an SVD: $A = UDV^{T}$. Here, U is a orthogonal matrix of shape $m x m$, D is a diagonal matrix of shape $m x n$, and V is an orthogonal matrix of shape $n x n$. D is not necessarily square.

- D is the singular values of A, also known as the square roots of the eigenvalues of $AA^{T}$.
- U is the eigenvectors of $AA^{T}$.
- V is the eigenvectors of $A^{T}A$.

## The Trace Operator

The trace operator gives the sum of the diagonal elements of a matrix: $tr(A) = \sum_{i}A_{i, i}$.

## The Determinant

The determinant of a square matrix is a function mapping matrices to real scalars. It is denoted as $det(A)$ or $|A|$. The determinant of a matrix is the product of all the eigenvalues of the matrix.

You can also calculate the determinant of a matrix in other ways, for example a 2x2 matrix:
A = $\begin{bmatrix} a & b \\ c & d \end{bmatrix}$
$det(A) = ad - bc$

This method is called the Laplace expansion.
