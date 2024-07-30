# Deep Learning

## Activation Functions

### SwiGLU

Defining swish: $swish(x) = x \cdot sigmoid(\beta x)$, where $\beta$ is a learned parameter. Swish can be better than ReLU because it provides a smoother transition around 0, which can lead to better optimization.

GLU - Gated linear unit, $glu(x) = sigmoid(W1 \cdot x + b) x (V \cdot x + c), where x is a component-wise product (hadamard product).

SwigGlu is a combination of both:

$SwiGLU(x) = Swish(W1 \cdot x + b) x (V \cdot x + c)$.

In pytorch:

```python
class SwiGLU(nn.Module):
    
    def __init__(self, w1, w2, w3) -> None:
        super().__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
    
    def forward(self, x):
        x1 = F.linear(x, self.w1.weight)
        x2 = F.linear(x, self.w2.weight)
        hidden = F.silu(x1) * x2
        return F.linear(hidden, self.w3.weight)
```
where F.silu is the swish activation function with $\beta = 1$.
