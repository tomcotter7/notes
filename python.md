# Python

## Pip

You can find the requirements for a package using the pypi json endpoint `https://pypi.org/pypi/<package>/<version>/json`.

## Importing Modules

A directory structure of the form:
```
- src/
-- __init__.py
-- package1/
--- foo.py
-- package2/
--- bar.py
-- main.py
```

If you are running src/main.py, the top level `__init__.py` is ignored, because we are already in `main.py`. Therefore, if you wanted to import foo into bar, you could just add `from package1.foo import baz` inside `bar.py`.
