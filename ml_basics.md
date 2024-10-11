# Machine Learning Basics - Chapter 5 of Ian Goodfellow's Deep Learning

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

## Estimators, Bias, and Variance

### Point Estimation

Point estimation is the attempt to provide the single "best" prediction of some quantity of interest. Usually, this is the parameters of a model, such as the weights in a linear regression model. To distinguish estimates and true values, we use the hat symbol. For example $\hat{\theta}$ is the point estimate of the parameter $\theta$.

Let ${x_1, x_2, ..., x_n}$ be a set of $n$ independent and identically distributed (i.i.d) data points. A point estimator is any function of the data: $\hat{\theta} = g(x_1, x_2, ..., x_n)$. This defintion is very general and does require that g returns a value that is close to the true $\theta$. Therefore, whilst any function is an estimator, a good estimator is a function whose output is close to the true underlying $\theta$ that generated the training data.

### Function Estimation

In this case, we want to predict a variable $y$ on a given input vector $x$. We assume that there is a function $f(x)$ that describes the approximate relationship between $x$ and $y$. A function estimator $\hat{f}$ is the same a point estimation but in funciton space.

### Bias

The bias of an estimator is defined as $bias(\hat{\theta}_m) = E[\hat{\theta}_m] - \theta$. Where $E$ means the expected value operator. Therefore, if an estimator is unbiased (has a bias of 0), then the expected value of the estimator is equal to the true value of the parameter being estimated.

### Variance and Standard Error

Another property of the estimator that we might want to consider is how much we expect to vary as function of the data sample. This provides a measure of how we would expect the estimate we compute from data to vary as we independently resample the dataset from the underlying data-generating process.

Bias measures the expected diaviation from the true value of the function or parameter. Variannce provides a measure of the deviation from the expected estimataor value that any particular sampling of the data is likely to cause. MSE is the sum of the variance and the square of the bias.

### Consistency

As the number of data points $m$ increase, we would like the estimator to converge to the true value of the parameter. Formally: $plim_{m \rightarrow \infty} \hat{\theta}_m = \theta$. 

$plim$ indicates a convergence in probability, meaning that for any $\epsilon < 0$, $P(|\hat{\theta}_m - \theta| > \epsilon) \rightarrow 0$ as $m \rightarrow \infty$. This is known as consistency. Consistency ensures that the bias induced by the estimator diminshes as the number of data examples grows.

## Maximum Likelihood Estimation

We would like to have some principle in which we can derivce specific functions, which are good estimators for different models. The most common of these principles is the maximum likelihood principle.

Consider a set of $m$ examples $\mathcal{X} = {x_1,...,x_m}$ drawn independently from the true but unknown data generating distribution $p_{data}$. Let $p_{model}(x;\theta)$ be a parametric family of probability distributions over the same space indexed by $\theta$. In other words, $p_{model}(x;\theta)$ maps any configuration $x$ to a real number estimating the true probability p_{data}(x).

Basically, given any $x$ and $\theta$, $p_{model}(x;\theta)$ is the probability that $x$ was generated by the true data generating distribution $p_{data}$ ($\theta$ is our true model parameters).

The maximum likelihood estimator for $\theta$ is then defined as:

$\theta_{ML} = argmax_{\theta} p_{model}(\mathcal{X};\theta) = argmax_{\theta} \prod_{i=1}^{m} p_{model}(x_i;\theta)$

The product is inconvient, therefore we take the log of the likelihood function (which does not change the arg max, but makes the product a sum):

$\theta_{ML} = argmax_{\theta} \sum_{i=1}^{m} log p_{model}(x_i;\theta)$

Because the argmax does not change when we rescale the cost function, we can divide by m to obtain a version of the criterion that is expresssed as an expectation with respect to the empirical distribution $\hat{p}_{data}$ defined by the training data:

$\theta_{ML} = argmax_{\theta} E_{x \sim \hat{p}_{data}} log p_{model}(x;\theta)$

One way to interpret this is to view it as minimizing the dissimilarity between the empirical distribution $\hat{p}_{data}$ and the model distribution. The degree of dissimilarity is measured by the KL divergence:

$D_{KL}(\hat{p}_{data} || p_{model}) = E_{x \sim \hat{p}_{data}}[log \hat{p}_{data}(x) - log p_{model}(x)]$. The term of the left is only dependent on the data, therefore when we train the model to minimize the KL divergence, we need only minimize the right term.

$ -E_{x \sim \hat{p}_{data}}[log p_{model}(x)]$

### Conditional Log-Likelihood and Mean Squared Error

The maximum likelihood estimator can readily be generalized to esimtate a conditional probability $P(y | x; \theta)$ in order to predict $y$ given $x$. If $X$ represents all our inputs, and $Y$ represents all our outputs, then the conditional maximum likelihood estimator is (assuming the examples are independent):

$\theta_{ML} = argmax_{\theta} \sum_{i=1}^{m} log P(y_i | x_i; \theta)$

THe main appeal of the maximum likelihood estimator is that it can be shown to be the best estimator asymptotically, as the number of exmaples $m \rightarrow \inf$, in terms of its rate of convergence of $m$ increases. The maximum likelihood esimtator has the property of consistency, as long as the data generating distribution $p_{data}$ is in the model family $p_{model}(\theta)$.

## Bayesian Statistics

The frequentist perspective is that the true parameter value $\theta$ is fixed but unknown, while the point estimate $\hat{\theta}$ is a random variable. The Bayesian uses probability to reflect degrees of certainty in states of knowledge. The true parameter $\theta$ is unknown or uncertain, therefore is represented as a random variable.

In Bayesian, we essentially build a `prior` distribution, which is our initial belief about the parameter $\theta$. We then update the prior after seeing the training examples to produce a `posterior`, in which the training examples cause it to lose entropy and concentrate around a few highly likely values of the parameters.

Before observing the data, we represent of knowledge of $\theta$ using the prior probability distribution, $p(\theta)$. After observing the data, we update our knowledge of $\theta$ using the posterior probability distribution, $p(\theta | \mathcal{X})$. Therefore, after having observed $\mathcal{X}$, if we are still quite uncertain about the value of $\theta$ then this uncertainty will be incorporated into our predictions.

### Maximum a Posteriori (MAP) Esimation

The MAP estimate chooses the point of maximal posterior probability.

$\theta_{MAP} = argmax_{\theta} p(\theta | x) = argmax_{\theta}log p(x | \theta}) + log p(\theta)$

$logp(\theta)$ is the prior distribution, and $log p(x | \theta)$ is the log likelihood term.

## Supervised Learning Algorithms

### Probabilistic Supervised Learning

We want to estimate a probability distribution $p(y | x)$, which we can do by finding the best parameter vector $\theta$ for a parametric family of distributions $p(y | x; \theta)$.

Linear regression corresponds to the family: $p(y | x;\theta) = \mathcal{N}(y; \theta^{T}x, I)$, which basically means "for any $x$, we expect $y$ to be normally distributed around $\theta^{T}x$"

A logistic sigmoid function is defined as:

$p(y = 1 | x; \theta) = \sigmoid(\theta^{T}x)$

Logistic regression (as defined above) has no closed form solution, we must minimize the negative log likelihood using gradient descent.

### Support Vector Machines

This type of model is similar to linear regression in that it is driven by a linear function $w^{T}x + b$. The SVM predicts that the [postive|negative] class is present when $w^{T}x + b$ is [positive|negative]. The key innovation of SVMs is the kernel trick, in which it can be shown that many ML algorithms can be rewritten in terms of dot products between examples.

For example, linear regression can be rewritten as:

$w^{T}x + b = b + \Sigma_{i=1}^{m}\alpha_{i}x^{T}x^(i)$, where $x^{(i)}$ is a training example, and $\alpha$ is a vector of coefficients. Rewriting in this way enables us to replace $x$ with the output of a given feature function $\phi (x)$a and the dot product with a function $k(x, x^i) = \phi (x)^{T}\phi (x^i)}$. After replacing the dot product with kernal evaluations, we can make predictions using the function $f(x) = b + \Sigma_{i=1}^{m}\alpha_{i}k(x, x^i)$. The kernel-based function is exactly equivalent to preprocessing the data by applying $\phi (x)$ to all inputs, and then learning a linear model in the new transformed space.

This is powerful in two ways. 1, we can learn models that are nonlinear as a funtion of x using convex optimization techniques that are guaranteed to converge efficiently. This is possible because we consider $\phi$ fixed and optimize only $\alpha$. 2, the kernel function often admits an implementation that is significantly more efficient than naively constructing two $\phi (x)$ vectors and explicitly taking their dot product.

In many cases, $k(x, x')$ is a non-linear, tractable function of $x$ even when $\phi (x)$ is intractable.

Other linear models use this, such as kernel machines. However, SVM have an advantage in that the vector $\alpha$ in a SVM contains mostly zeros. Therefore, classifying a new example requires evaluating the kernel functions only for the training examples that have nonzero $\alpha_{i}$. These training examples are known as support vectors.
>>>>>>> 681883f52c2fe092d9f95892148353c0bffc1a2f
