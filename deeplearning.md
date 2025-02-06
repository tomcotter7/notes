# Deep Learning

## Activation Functions

### SwiGLU

Defining swish: $swish(x) = x \cdot sigmoid(\beta x)$, where $\beta$ is a learned parameter. Swish can be better than ReLU because it provides a smoother transition around 0, which can lead to better optimization.

GLU - Gated linear unit, $glu(x_i) = sigmoid(W1 \cdot x_i + b) x (V \cdot x_i + c), where x is a component-wise product (hadamard product).

SwiGLU is a variant that replaces the sigmoid in GLU with a Swish activation function.

$SwiGLU(x_i) = swish((W1 \cdot x_i) + b) x ((V \cdot x_i) + c)$.

In PyTorch:

```python
class SwiGLU(nn.Module):
    
    def __init__(self, w1, w2, b, c) -> None:
        super().__init__()
        self.w1 = w1
        self.w2 = w2
        self.b = b
        self.c
    
    def forward(self, x):
        x1 = F.linear(x, self.w1.weight, self.b)
        x2 = F.linear(x, self.w2.weight, self.c)
        return F.silu(x1) * x2
```
where F.silu is the swish activation function with $\beta = 1$.

## Layers

### Convolutional

If padding is N, add N elements to the start & end of each row & column in the input. For example, if the input is 5 x 5 and the padding is 1, then the padded input is a 7 x 7

If the kernel size is N, this is the size of the data looked at to produce the output. For example, is the kernel size is 3, then each 3x3 square in the input will be combined to produce 1 element in the output.

If the stride is N, then the kernel moves N elements across for each step. For example, let's say the kernel starts in the top-left corner, and the top-left of the kernel looks at element (0, 0). If the stride is 1, then the next step has the top left of the kernel looking at (0, 1).

We can calculate the output size of a convolutional layer via this formula: `output = [(input + 2×padding - kernel) / stride] + 1`. We can calculate the number of parameters via: `parameters = (kernel height × kernel width × input channels × output channels) + bias`

If we take a look in PyTorch: `layer = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)`, this layer would take in a 3 channel image (likely RGB), and then apply 64 3x3 kernels to it to produce 64 output channels. This would be 1728 + 64 parameters, because we have 64 bias's

### MaxPool

Take the maximum value from each "kernel" and discard the rest. An example with a 2x2 MaxPool is:

```
Input (4x4):    Using 2x2 MaxPool:
1  3  2  4         
5  7  0  2    →    7  4    (takes max value from each 2x2 region)
1  2  3  4         6  8
2  6  7  8
```

