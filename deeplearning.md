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

## Layers

### Convolutional

If padding is N, add N elements to the start & end of each row & column in the input. For example, if the input is 5 x 5 and the padding is 1, then the padded input is a 7 x 7

If the kernel size is N, this is the size of the data looked at to produce the output. For example, is the kernel size is 3, then each 3x3 square in the input will be combined to produce 1 element in the output.

If the stride is N, then the kernel moves N elements across for each step. For example, let's say the kernel starts in the top-left corner, and the top-left of the kernel looks at element (0, 0). If the stride is 1, then the next step has the top left of the kernel looking at (0, 1).

### MaxPool

Take the maximum value from each "kernel" and discard the rest.

