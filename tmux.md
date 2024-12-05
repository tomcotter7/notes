# TMUX

## Renumber windows

```
:move-window -r
```

## Move a pane to a new window

```
:break-pane -d
```

## View available sessions and switch to one

```
<C-b> s
```

## Rename current session

```
<C-b> $
```

## Create a new session

```
<C-b> :new
<C-b> s
```

## Kill a session

```
tmux kill-session -t <session_name>
```
