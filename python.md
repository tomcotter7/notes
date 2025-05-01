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

## `pre-commit`

`pre-commit` is a useful way of setting up automated commands on every commit. Here is a nice, mininmal `.pre-commit-config.yaml`

```yaml
repos:
- repo: https://github.com/astral-sh/ruff-pre-commit
  rev: v0.4.10
  hooks:
    - id: ruff-format
- repo: https://github.com/astral-sh/ruff-pre-commit
  rev: v0.4.10
  hooks:
    - id: ruff
- repo: https://github.com/qoomon/git-conventional-commits
  rev: v2.7.2
  hooks:
    - id: conventional-commits

- repo: local
  hooks:
    - id: pip-compile
      name: uv pip compile
      entry: uv pip compile pyproject.toml -o requirements.txt
      language: system
      pass_filenames: false
      files: ^(pyproject.toml|\.pre-commit-config\.yaml)$
```

## `venv-pack`

[venv-pack](https://jcristharif.com/venv-pack/index.html) is a tool recommended by Spark to produce consistent production workspaces given a virtual environment.

You can simply create a tarball by running the following:

```bash
venv-pack -o my_env.tar.gz
```

## ipython / kernels

### Setting up a Kernel

`pip install ipykernel`
`python -m ipykernel install --user --name <name_of_kernel>`

You can now access that kernel. If you (and you should) installed the `ipykernel` inside a virtual env, then it has access to all the packages inside that virtual env.

### Removing Kernels

`jupyter kernelspec remove <name_of_kernel>`
