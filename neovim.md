# Neovim

## Find & Replace Commands

- `:%s/foo/bar/g` - Replace all occurrences of `foo` with `bar` in the entire file.
- `:s/foo/bar/g` - Replace all occurrences of `foo` with `bar` in the current line.
- `:10,20s/foo/bar/g` - Replace all occurrences of `foo` with `bar` between lines 10 and 20.

## Grep Commands

- :vimgrep /pattern/ file - Search for `pattern` in `file`.
    - use `copen` to open the quickfix window.

