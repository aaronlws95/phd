# Snippets

## `nvidia-smi`

```
watch -n 0.5 nvidia-smi
```

## TMUX
* `tmux new-session -s aaron`
* `tmux attach -t aaron`
* `Ctrl+B D` go out
* `Ctrl+b "` split pane horizontally
* `Ctrl+b %` split pane vertically

## Jupyter Notebook on SSH
On remote:
```
jupyter notebook --no-browser --port=8080
```

On local:
```
ssh -N -L 8080:localhost:8080 aaron@172.21.148.106
Go to http://localhost:8080/
```

## Tensorboard on SSH
```
ssh -L 16006:127.0.0.1:6006 aaron@172.21.148.106
ssh -L 16006:127.0.0.1:6006 ntu-ws
tensorboard --logdir .
Go to http://127.0.0.1:16006
```

## SSH Shortcut

```
ssh-keygen -t rsa -b 2048
ssh-copy-id id@server
```

* Paste in .ssh/config
```
Host st-ws
    HostName 203.126.234.118
    User awslow
    Port 10087
```

## `git lg`

```
git config --global alias.lg "log --color --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit --"
```

## Iterating in Bash

```
for i in {0..100..10}; do echo $i; done
```