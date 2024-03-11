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

