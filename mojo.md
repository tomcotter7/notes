# Mojo

## Resources

- [Mojo Cheatsheet](https://github.com/czheo/mojo-cheatsheet/blob/main/README.md)
- [Learn Mojo](https://ruhati.net/mojo/)

## Ownership

[Full Details Here](https://www.youtube.com/watch?v=9ag0fPMmYPQ)

```mojo
fn b(borrowed x: String):
    # x is initialized
    # x is immutable

fn b(inout x: String):
    # x is initialized
    # x is mutable
    # x has to be initialized at the end of the function

fn b(owned x: String):
    # x is initialized
    # x is destroyed at the end of the function if it is not transferred.
```

```mojo

fn __moveinit__(inout self: Self, owned other: Self):
    # you "poach" the fields from owned into self.
    # other is destroyed at the end of the function.
```

## Mojo Benchmarks (vs. Python)

- Benchmarks revealed that Mojo has superior performance to Python in small, fixed-size matrix multiplications, attributing the speed to the high overhead of Python's numpy for such tasks.
