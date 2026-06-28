#!/usr/bin/env bash
# Stop backend + frontend started by startup.sh.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="$ROOT/.run"
BACK_PORT="${BACK_PORT:-8000}"
FRONT_PORT="${FRONT_PORT:-5173}"

ok()  { printf '\033[32m✓ %s\033[0m\n' "$*"; }
warn(){ printf '\033[33m• %s\033[0m\n' "$*"; }

# Kill a process tree by recorded PID, then clean up the pidfile.
stop_pid() { # name
  local f="$RUN/$1.pid"
  [ -f "$f" ] || { warn "$1: no pidfile"; return; }
  local pid; pid="$(cat "$f")"
  if kill -0 "$pid" 2>/dev/null; then
    pkill -TERM -P "$pid" 2>/dev/null || true   # children (uvicorn reloader, vite)
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 10); do kill -0 "$pid" 2>/dev/null || break; sleep 0.3; done
    kill -KILL "$pid" 2>/dev/null || true
    ok "$1 stopped (pid $pid)"
  else
    warn "$1: not running"
  fi
  rm -f "$f"
}

stop_pid backend
stop_pid frontend

# Safety net: reclaim the ports if anything is still bound (lsof is macOS/Linux default).
for port in "$BACK_PORT" "$FRONT_PORT"; do
  pids="$(lsof -ti tcp:"$port" 2>/dev/null || true)"
  if [ -n "$pids" ]; then
    warn "freeing port $port"
    # shellcheck disable=SC2086
    kill -KILL $pids 2>/dev/null || true
  fi
done

ok "all stopped"
