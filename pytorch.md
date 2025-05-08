# PyTorch

## torch.Tensor

[Docs](https://pytorch.org/docs/stable/tensors.html)

`torch.tensor()` always copies data. To avoid copying when using a numpy array, use `torch.as_tensor()`.

A tensor of shape [[[1, 2, 3], [4, 5, 6]]] has shape `(1, 2, 3)`. This is because there is 1 dimension of a 2 x 3 matrix. This is a 3D tensor.

Another example of a 3D tensor is [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]. This tensor has shape `(2, 2, 2)`. This is because there are 2 dimensions of a 2 x 2 matrix.

Easy way to remember matmuls:

The main two rules for matrix multiplication to remember are:

The inner dimensions must match:
- (3, 2) @ (3, 2) won't work
- (2, 3) @ (3, 2) will work
- (3, 2) @ (2, 3) will work
The resulting matrix has the shape of the outer dimensions:
- (2, 3) @ (3, 2) -> (2, 2)
- (3, 2) @ (2, 3) -> (3, 3)

## torch.Norm

```python
x = torch.rand(3, 16, 16)
x.norm(dim=-1, keep_dims=True)
````

This calculates the L2 norm for each 16 length vector in the last dimension (aka (3 * 16) 48 vectors) and stores in a (3, 16, 1) tensor. If keepdims was False, you would get a (3, 16) but it's useful to have the same dimenision becuase you can do things like:

```python
x = torch.rand(3, 16, 16)
x /= x.norm(dim=-1, keep_dims=True)
```

which means that each element in the original x tensor is divided by their corresponding vector norm. The outcome of dividing a vector by its L2 norm is a unit vector - a vector whose L2 norm is exactly 1. Normalization in this way removes the influence of the vector's magnitude (length).





## Reshaping / Changing Dimensions

| Method | One-line description |
| ------ | -------------------- |
| torch.reshape(input, shape) | Reshapes input to shape (if compatible), can also use torch.Tensor.reshape(). |
| Tensor.view(shape) | Returns a view of the original tensor in a different shape but shares the same data as the original tensor. |
| torch.stack(tensors, dim=0) | Concatenates a sequence of tensors along a new dimension (dim), all tensors must be same size. |
| torch.squeeze(input) | Squeezes input to remove all the dimenions with value 1. |
| torch.unsqueeze(input, dim) | Returns input with a dimension value of 1 added at dim. |
| torch.permute(input, dims) | Returns a view of the original input with its dimensions permuted (rearranged) to dims. |

## Saving and Loading Models

The recommened way for saving and loading a model for **inference** is to save/load a model's `state_dict()`. You can do this by:

```python
torch.save(model.state_dict(), PATH)
```

It's common convention for PyTorch models to save/load models using the `.pt` or `.pth` file extension.

To load the model:

```python
torch.nn.Module.load_state_dict(torch.load(PATH))
```

## Torchinfo

```python
from torchinfo import summary

summary(model, input_size=<some input shape>)
```

This will show you how the input is transformed when passed through the model.


