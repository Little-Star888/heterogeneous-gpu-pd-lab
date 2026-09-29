#!/bin/sh
# Restart the xy PD front door inside the vLLM P21 container (idempotent).
pkill -f xy_pd_proxy >/dev/null 2>&1 || true
sleep 1
nohup python3 /srv/xy_pd_proxy.py --port 5701 > /srv/xy_pd_proxy.log 2>&1 &
sleep 3
echo "xy-restart: pids=$(pgrep -f xy_pd_proxy | tr '\n' ' ')"
tail -3 /srv/xy_pd_proxy.log 2>/dev/null || true
