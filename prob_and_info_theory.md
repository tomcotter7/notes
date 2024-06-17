# Probability & Information Theory

## Random Variables

A random variable is a variable that can take on different values randomly. On its own,  a random variable is just a description of that states that are possible; it must be coupled with a probability distribution that specifies how likely each of these states are.

## Probability Distributions

A probability distribution is a description of how likely a random variable is to take on each of its possible states.

### Discrete Variables and Probability Mass Functions

A probability distribution over discrete variables may be described using a probability mass function (PMF). The probability function maps from a state of a random variable to the probability of that random variable taking on that state.

PMFs can act on multiple variables at the same time (joint probability distribution). $P(x = x, y = y)$ denotes the probability that x = x and y = y simultaneously. This can be simplified down to $P(x, y)$. PMFs must satisfy the following properties:

- The domain of $P$ must be the set of all possible states of x. 
- $\forall x \in x, 0 \leq P(x) \leq 1$.
- $\sum_{x \in x} P(x) = 1$, normalisation.

### Continuous Variables and Probability Density Functions

A probability distribution over continuous variables may be described using a probability density function (PDF). It must satisfy the following properties:

- The domain of $P$ must be the set of all possible states of x.
- $\forall x \in x, P(x) \geq 0$. We do not require that $P(x) \leq 1$.
- $\int P(x)dx = 1$, normalisation.

A probability density function does not give the probablity of a specific state directly; instead it gives the probability of landing inside an infinitesimally small region with volume $dx$.

## Marginal Probability

The probability over a subset of variables is known as the marginal probability distribution. For example,

- We have discrete random variables $x$ and $y$.
- We know $P(x, y)$.
- We can find $P(x)$ with the sum rule: $\forall x \in x, P(x = x) = \sum_{y} P(x = x, y = y)$.

For continuous variables, we can use the integral rule: $P(x) = \int P(x, y)dy$.

## Conditional Probability

The conditional probability of $x = x$ given that $y = y$ is denoted as $P(x = x | y = y)$. It is defined as:
$P(x = x | y = y) = \frac{P(x = x, y = y)}{P(y = y)}$. This is only defined when $P(y = y) > 0$.

## The Chain Rule of Conditional Probabilities

Any joint probability distribution over many random variables may be decomposed into conditional distributions over only one variable:

$P(x^{(1)}, ..., x^{(n)}) = P(x^{(1)})\prod_{i=2}^{n}P(x^{(i)} | x^{(1)}, ..., x^{(i-1)})$.

## Independence and Conditional Independence

Two random variables $x$ and $y$ are independent if their probability distribution can be expressed as a product of two factors, one involving only $x$ and one involving only $y$:

$\forall x \in x, y \in y, p(x = x, y = y) = p(x = x)p(y = y)$.

Two random variables $x$ and $y$ are conditionally independent given a random variable $z$ if the conditional probability distribution over $x$ and $y$ factorises in this way for every value of $z$:

$\forall x \in x, y \in y, z \in z, p(x = x, y = y | z = z) = p(x = x | z = z)p(y = y | z = z)$.

## Expectation, Variance and Covariance

The expected value of some function $f(x)$ with respect to a probability distribution $P(x)$ is the average value that f takes on when x is drawn from P - $E_{x \sim P}[f(x)] = \sum_{x}P(x)f(x)$. For continuous variables, this is $\int P(x)f(x)dx$.

Expectations are linear: $E_{x}[a f(x) + bg(x)] = aE_{x}[f(x)] + bE_{x}[g(x)]$, when $a$ and $b$ are not dependent on $x$.

The variance gives a measure of how much the values of a function of a random variable x vary as we sample different values of x from its probability distribution - $Var(f(x)) = E[(f(x) - E[f(x)])^2]$. The standard deviation is the square root of the variance.

The covariance gives us some sense of how much two values are linearly related to each other, as well as the scale of these variables - $Cov(f(x), g(y)) = E[(f(x) - E[f(x)])(g(y) - E[g(y)])]$. High absolute values of the covariance mean that the values change very much and are both far from their respective means at the same time. If the sign of the covariance is positive, then both variables tend to take on relatively high values simultaneously. If the sign of the covariance is negative, then one variable tends to take on a relatively high value when at the times that the other takes on a relatively low value and vice versa.

Other measures such as correlation normalize the contribution of each variable in order to measure only how much the variables are related, not how much they vary individually.

For two variable to have zero covariance, there must be no linear dependence between them. Independence is a stronger requirement that zero covariance, because independence also excludes non-linear relationships.

The covariance matrix of a random vector $x \in R^n$ is an $n x n$ matrix, such that - $Cov(x)_{i, j} = Cov(x_i, x_j)$. The diagonal elements of the covariance give the variance of each element of x, $Cov(x_i, x_i) = Var(x_i)$.

