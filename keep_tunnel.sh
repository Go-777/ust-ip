#!/bin/bash
# Keep SSH tunnel alive for tokenPlan API
# Remote forward: server's localhost:19443 -> modelservice.jdcloud.com:443
# Usage: nohup ./keep_tunnel.sh > /tmp/tunnel_keeper.log 2>&1 &

TUNNEL_LOG="/tmp/tunnel_keeper.log"

# Kill any existing tunnel processes (except ourselves)
kill_existing_tunnels() {
    pgrep -f "ssh.*-R 19443.*nb-m7wkdajepx" | while read pid; do
        if [ "$pid" != "$$" ]; then
            kill "$pid" 2>/dev/null
        fi
    done
}

while true; do
    echo "[$(date)] Checking tunnel connectivity..."
    
    # Test if tunnel is working by connecting from server side
    if ssh -o ConnectTimeout=10 nb-m7wkdajepx "curl -sk -o /dev/null -w '%{http_code}' -H 'Host: modelservice.jdcloud.com' https://localhost:19443/ 2>/dev/null" 2>/dev/null | grep -qE '^[2-5][0-9][0-9]$'; then
        echo "[$(date)] Tunnel is healthy (got HTTP response)"
        sleep 30
        continue
    fi
    
    echo "[$(date)] Tunnel is DOWN, restarting..."
    kill_existing_tunnels
    sleep 2
    
    # Start tunnel in foreground (blocks until disconnected)
    ssh -N -R 19443:modelservice.jdcloud.com:443 \
        -o ServerAliveInterval=10 \
        -o ServerAliveCountMax=3 \
        -o ExitOnForwardFailure=yes \
        -o TCPKeepAlive=yes \
        -o ConnectTimeout=30 \
        nb-m7wkdajepx
    
    echo "[$(date)] Tunnel died (exit=$?), restarting in 3s..."
    sleep 3
done