# Traditional ML

## Clustering

### Gaussian Mixture Models (GMM)

GMM is a probabilistic K-means.

KMeans has the limitations:

- It assumes that the clusters are spherical
- It's a hard clustering algorithm, meaning that each point is assigned to a single cluster

Essentially, we want to approximate the gaussian distributions present in the data, definining the number of clusters (i.e. the number of gaussians) beforehand.

Steps of GMM:
    - Decide the number of clusters - using BIC or AIC
    - Initialize the mean, covariance and weights of the gaussians
    - Use the Expectation-Maximization algorithm to do the following:
        - E-step: Calculate the probability of each point belonging to each cluster (distribution), then evaluate the likelihod function given the current estimate for the parameters
        - M-step: Update the mean, covariance and weights of the gaussians to maximize the likelihood function.
    - Repeat the E and M steps until convergence

[Here](https://towardsdatascience.com/gaussian-mixture-model-clearly-explained-115010f7d4cf) is a good article on GMM.

The optimal number of clusters can be found using an elbow test of the BIC/AIC scores - see [this](https://stats.stackexchange.com/questions/368560/elbow-test-using-aic-bic-for-identifying-number-of-clusters-using-gmm) StackExchange post for more information.

## Dimensionality Reduction

### UMAP

UMAP is a manifold learning technique for dimension reduction. Find the paper [here](https://arxiv.org/abs/1802.03426.pdf).

## Classification

### Logistic Regression

For binary classification:

$y_pred = 1 / (1 + exp(-h))$

where $h = w_0 + w_1x_1 + w_2x_2 + ... + w_nx_n$, or the dot product of the weight vector and the feature vector - $h = w^TX$

Here, we can use Binary Cross-Entropy loss:

$J(w) = -\frac{1}{N} \Sigma_{i=0}^{N} y_{true} * log(y_{pred} + \epsilon) + (1 - y_{true} * log(1 - y_{pred} + \epsilon)$

Where $\epsilon$ is some small number to prevent $log(0)$ from happening.

We can derive the gradient using the chain rule:

$\frac{\delta J(w)}{\delta w} = \frac{\delta J(w)}{\delta y_{pred}} \cdot \frac{\delta y_{pred}}{\delta h} \cdot \frac{\delta h}{\delta w}$

$\frac{\delta h}{\delta w} = X$
$\frac{\delta y_{pred}}{\delta h} = y_{pred}(1 - y_{pred})$ - because of the derivative of the exponent.
$\frac{\delta J(w)}{\delta y_{pred}} = \frac{y_{pred} - y_{true}}{y_{pred}(1 - y_{pred})$ - because of the derivative of logs.

Therefore, $\frac{\delta J(w)}{\delta w} = X^{T}(y_{pred} - y_{true})
