#!/usr/bin/env bash
# Local dev: a LiveKit server reachable from your phone on the LAN, plus the agent.
#   brew install livekit
#   python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
#   ./dev.sh
set -euo pipefail
cd "$(dirname "$0")"

LAN_IP="${LAN_IP:-$(ipconfig getifaddr en0 || ipconfig getifaddr en1)}"
echo "Mac LAN IP: $LAN_IP"
echo "In the iOS app, set Backend URL to: http://$LAN_IP:8765"
echo

# --dev uses API key "devkey" / secret "secret". Bind to all interfaces so the
# phone can reach it, and advertise the LAN IP for WebRTC media.
livekit-server --dev --bind 0.0.0.0 --node-ip "$LAN_IP" > livekit-server.log 2>&1 &
LK_PID=$!
trap 'kill $LK_PID 2>/dev/null' EXIT
sleep 1

PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python
"$PY" live_agent.py
