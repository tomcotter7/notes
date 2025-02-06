# Linux

## Find IP Addresses

`ifconfig | sed -En 's/127.0.0.1//;s/.*inet (addr:)?(([0-9]*\.){3}[0-9]*).*/\2/p'`

This finds all of your systems IP addressses.

## PM2

PM2 is an application manager that can help keep a process running 24/7. The init command is `pm2 start %command%`. The docs are [here](https://pm2.keymetrics.io/).
