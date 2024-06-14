# Mojo

## Resources

- [Mojo Cheatsheet](https://github.com/czheo/mojo-cheatsheet/blob/main/README.md)
- [Learn Mojo](https://ruhati.net/mojo/)

## Ownership

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

