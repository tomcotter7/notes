# Bash

## diff

Get all files in dir1 but not in dir2 and move them to dir3. `one-liner`:

```bash
diff <(ls -1 dir1) <(ls -1 dir2) | grep "^<" | cut -c 3- | while read -r file; do mv "dir1/$file" dir3/; done
```

## Test

hello
