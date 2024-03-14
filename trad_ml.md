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
