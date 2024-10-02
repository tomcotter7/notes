# Probability & Information Theory - Chapter 3 of Ian Goodfellow's Deep Learning

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

## Common Probability Distributions

### Bernoulli Distribution

The Bernoulli distribution is a distribution over a single binary random variable. It is controlled by a single parameter $\phi \in [0, 1]$, which gives the probability of the random variable being equal to 1.

### Multinoulli Distribution

The multinoulli, or categorical, distribution is a distribution over a single discrete variable with k different states, where k is finite.

### Gaussian Distribution

Also known as the normal distribution.

$\mathcal{N}(x; \mu, \sigma^2) = \sqrt(\frac{1}{2\pi\sigma^2})exp(-\frac{1}{2\sigma^2}(x - \mu)^2)$.

The two params, $\mu$ and $\sigma^2$, are the mean and variance of the distribution. The $\mu$ gives the co-ordinate of the central peak, this is also the mean of the distribution: $E[x] = \mu$.

When we evaluate the PDF, we need to square and invert $\sigma$. If we are repeatedly doing this, a more efficient way is to use a parameter $\Beta \in (0, \infty)$ to control the precision (or inverse variance) of the distribution:

$\mathcal{N}(x; \mu, \Beta^{-1}) = \sqrt(\frac{\Beta}{2\pi})exp(-\frac{\Beta}{2}(x - \mu)^2)$.

Out of all possible probability distributions with the same variance, the normal distribution encodes the maximum amount of uncertainty over the real numbers. We can thus think of the normal distribution as the one that puts the least amount of prior knowledge into the model.

The normal distribution generalizes to $R^n$ as the multivariate normal distribution.It may be parameterized with a positive definite symmetric matrix $\Sigma$, which is the covariance matrix of the distribution.

$\mathcal{N}(x; \mu, \Sigma) = \sqrt(\frac{1}{(2\pi)^n|\Sigma|})exp(-\frac{1}{2}(x - \mu)^T\Sigma^{-1}(x - \mu))$.

Since the covariance matrix is not a computationally efficient wy to parameterize the distribution, we use the precision matrix $\Beta$ instead:

$\mathcal{N}(x; \mu, \Beta^{-1}) = \sqrt(\frac{det(\Beta)}{(2\pi)^n})exp(-\frac{1}{2}(x - \mu)^T\Beta(x - \mu))$.

### Exponential and Laplace Distributions

The exponential distribution has a sharp point at x = 0.

$p(x; \lambda) = \lambda 1_{x \geq 0}exp(-\lambda x)$.

The indicator function $1_{x \geq 0}$ assigns a probability of 0 to all negative values of x (and 1 to all non-negative values). 

The Laplace distribution allows us to place a sharp peak of probability mass at an arbitrary point $\mu$.

$Laplace(x; \mu, \gamma) = \frac{1}{2\gamma}exp(-\frac{|x - \mu|}{\gamma})$.

### The Dirac Distribution and Empirical Distribution

In some cases, we wish to specify that all the mass in a probability distribution clusters around a single point. We can do thi with a Dirac delta function $\delta(x)$: $p(x) = \delta(x - \mu)$.

The Dirac delta function is defined such that it is zero valued everywhere except 0, yet integrates to 1.

A common use of the Dirac delta distribution is as a component of an empirical distribution. $p(x) = \frac{1}{m}\sum_{i=1}^{m}\delta(x - x^{(i)})$. This puts probability mass 1/m at each of the points $x^{(i)}$. You can imagine an empirical distribution as essential a training set.

### Mixtures of Distributions

You can combine distributions to construct a mixture distribution. On each trial, the choice of which component distribution should generate the sample is determined by sampling a component identity from a multinoulli distribution:

$P(x) = \sum_{i}P(c = i)P(x | c = i)$, where $P(c)$ is the multinoulli distribution over the component identities. The empirical distribution is a special case of a mixture distribution, as it is a mixture of Dirac delta functions.

## Bayes Rule

$P(x | y) = \frac{P(y | x)P(x)}{P(y)}$.

## Information Theory

THe basic intuition behind information theory is that learning that an unlikely event has occurred is more informative than learning that a likely event has occurred.

- Likely events should have low information content, and in the extreme case, events that are guaranteed to happen should have no information content whatsoever.
- Less likely events should have higher information content.
- Independent events should have additive information. For example, finding out that a tossed coin has come up heads twice should convey twice as much information as finding out that a tossed coin has come up heads once.

The self-information of an event x = x is: $I(x) = -logP(x)$. In ML, we use the natural logarithm, and as such the information is measured in "nats". One nat is the amount of information gained by observing an event of probability $\frac{1}{e}$.

Self-information deals with a single outcome. We can quantify the amount of uncertainty in an entire probability distribution using the Shannon entropy: $H(x) = E_{x \sim P}I(x) = -E_{x \sim P}logP(x)$. In other words, the Shannon entropy of a distribution is the expected amount of information in an event drawn from that distribution. 

If we have two separate probability distributions $P(x)$ and $Q(x)$ over the same random variable x, we can measure how different these two distributions are using the Kullback-Leibler (KL) divergence: $D_{KL}(P||Q) = E_{x \sim P}[log\frac{P(x)}{Q(x)}] = E_{x \sim P}[logP(x) - logQ(x)]$. The KL divergence is not symmetric, $D_{KL}(P||Q) \neq D_{KL}(Q||P)$. It is defined as the extra amount of information needed to send a message containing symbols drawn from P when we use a code that was designed to minimize the length of messages drawn from Q.

Cross-Entropy is $H(P, Q) = H(P) + D_{KL}(P||Q)$. Or more simply, $H(P, Q) = -E_{x \sim P}logQ(x)$.

## Structured Probabilistic Models

Instead of using a single function to represent a probability distribution, we can split a probability distribution into many factors that we multiply together. For example, suppose we have three random variables: a, b & c. Supposes that a influences b, and b influences c, but that a and c are independent given b. We can represent this as: $P(a, b, c) = P(a)P(b | a)P(c | b)$. We can greatly reduce the cost of representing a distribution  if we are able to find a factorization into distributions over fewer variables. We can describe these factorizations with graphs (creating structure probabilistic models).

Directed models use graphs with directed edges & represent factorizations into conditional probability distributions. $p(x) = \prod_{i}p(x_i | Pa_{\mathcal{g}}(x_i))$. The parents of a node are the nodes with edges pointing directly into that node.

Undirected models represent factorizations into a set of functions; these functions are not probability distributions of any kind. Any set of nodes that are all connected to each other in the graph is called a "clique".
