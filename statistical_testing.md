# Statistical Testing

## Paired Ratio Comparison

You would use this test, for example, when analysing if the difference in task accuracy (correct / total) between model_A and model_B is statistically significant or not. For this, it is useful to use a paired bootstrapped test, and sometimes with clusters, depending on your data.
For example, if the test set has multiple rows from the same user, these can be correlated, and therefore all rows from one user should be resampled together as one cluster.

A worked example:

$model_A$ vs. $model_B$:
- For each user in the test set, sum the correct predictions for each model ($k_A$, $k_B$), and the total number of rows for that user ($n$).
- Bootstrap $B$ times (typically 1,000–10,000), where a "boot" is a sample of users (selected by random sampling with replacement, so one user can appear multiple times). Use the **same** sampled users for both models in each boot. This is what makes the test paired: user difficulty cancels out, because both models see the same easy and hard users.
- For each boot, calculate accuracy as a ratio of sums, $\sum k / \sum n$ (not the mean of per-user accuracies), for each model, then take the difference.
- The point estimate is the difference on the non-bootstrapped data (a.k.a. the full sample).
- Calculate the CI from the bootstrap differences, using the middle 95% (the 2.5th to 97.5th percentiles), a.k.a. `np.percentile(diffs, [2.5, 97.5])`.
- This gives a 95% range of plausible values for the difference if you sampled different users.
- If this CI does not include 0, the difference is significant at $\alpha = 0.05$ (two-sided).

```python
idx = rng.integers(0, n_users, size=n_users)  # one sample of users...
acc_A = k_A[idx].sum() / n[idx].sum()         # ...used for A
acc_B = k_B[idx].sum() / n[idx].sum()         # ...and for B
diffs.append(acc_B - acc_A)
```

In this case, I prefer to also use some $\delta$ for practical significance. Compare $\delta$ to the CI, not to the point estimate:
- The CI is fully outside $[-\delta, \delta]$: the change matters.
- The CI is fully inside $[-\delta, \delta]$: the change is negligible, even if it is stat. sig.
- The CI includes $\pm\delta$: the result is unclear.
