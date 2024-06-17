# Git

## Get the email of the author of the last commit

`git log --format="%ae" | head -1`

## Run a command after `git push`

git config alias.xpush `!git push $1 $2 && <your-command>`.

## Checkout to remote branch

`git fetch`
`git switch <remote-branch-name>`
