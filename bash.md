# Bash

## diff

This one-liner gets all files in `dir1` but not in `dir2` and move them to `dir3`:

```bash
diff <(ls -1 dir1) <(ls -1 dir2) | grep "^<" | cut -c 3- | while read -r file; do mv "dir1/$file" dir3/; done
```

<(ls -1 dir1) lists files (one per line) and stores it a temporary file, so `diff` can do the comparision. The output of `diff` is a set of lines where < indicate files only in dir`, and > indicate files only in dir2. The grep filters away all files in dir2. The cut then removes the first two characters (`3-`), and then we exectute a while loop to iterate over the output of cut.
