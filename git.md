# Git

## Get the email of the author of the last commit

`git log --format="%ae" | head -1`

## Run a command after `git push`

git config alias.xpush `!git push $1 $2 && <your-command>`.

## Checkout to remote branch

`git fetch`
`git switch <remote-branch-name>`

## includeIf

```[includeIf "gitdir:~/work/"]
  path = ~/work/.gitconfig

[includeIf "gitdir:~/personal/"]
    path = ~/personal/.gitconfig
```

You can have different configurations for different directories.

## Git blame

See who changed a line in a file

`git blame -L 1,1 <file>` - would show who changed the first line of the file

`git blame -w` - ignore whitespace changes

## Git log

`git log -s` - allows you to do a regex search for a string to find any commits that contain changes to that string

## Git maintenance

run `git maintenance start` to run maintenance tasks in the background, should be run when you first create the repo.
