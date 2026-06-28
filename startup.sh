#!/usr/bin/env bash
# Start backend (FastAPI/uvicorn) + frontend (Vite) for local testing.
# Idempotent: sets up venv & deps if missing, refuses to double-start.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="$ROOT/.run"
VENV="$ROOT/.venv"
BACK_PORT="${BACK_PORT:-8000}"
FRONT_PORT="${FRONT_PORT:-5173}"
mkdir -p "$RUN"

# Load local secrets (ANTHROPIC_API_KEY, HF_API_TOKEN) so remote models light up.
# Vercel injects these from the dashboard in production; .env is local-only & gitignored.
if [ -f "$ROOT/.env" ]; then
  set -a; . "$ROOT/.env"; set +a
fi

log()  { printf '\033[36m▶ %s\033[0m\n' "$*"; }
ok()   { printf '\033[32m✓ %s\033[0m\n' "$*"; }
err()  { printf '\033[31m✗ %s\033[0m\n' "$*" >&2; }

need() { command -v "$1" >/dev/null 2>&1 || { err "missing required tool: $1"; exit 1; }; }
need python3; need node; need npm

# Refuse to start if a recorded PID is still alive.
running() { [ -f "$RUN/$1.pid" ] && kill -0 "$(cat "$RUN/$1.pid")" 2>/dev/null; }
if running backend || running frontend; then
  err "servers already running — run ./shutdown.sh first"; exit 1
fi

# --- Backend setup ---
if [ ! -d "$VENV" ]; then
  log "creating venv"
  python3 -m venv "$VENV"
fi
log "installing backend deps"
"$VENV/bin/pip" install -q -r "$ROOT/api/requirements.txt"

if [ ! -f "$ROOT/artifacts/logreg.pkl" ] || [ ! -f "$ROOT/artifacts/crf.pkl" ]; then
  log "training model pickles"
  "$VENV/bin/python" "$ROOT/scripts/train_models.py"
fi

# --- Frontend setup ---
if [ ! -d "$ROOT/frontend/node_modules" ]; then
  log "installing frontend deps"
  (cd "$ROOT/frontend" && npm install --silent)
fi

# --- Launch backend ---
log "starting backend on :$BACK_PORT"
(cd "$ROOT" && "$VENV/bin/uvicorn" api.index:app --port "$BACK_PORT" --reload \
  >"$RUN/backend.log" 2>&1 & echo $! >"$RUN/backend.pid")

# --- Launch frontend ---
log "starting frontend on :$FRONT_PORT"
(cd "$ROOT/frontend" && npm run dev -- --port "$FRONT_PORT" \
  >"$RUN/frontend.log" 2>&1 & echo $! >"$RUN/frontend.pid")

# --- Wait for health ---
wait_up() { # name url
  for _ in $(seq 1 40); do
    curl -fsS "$2" >/dev/null 2>&1 && { ok "$1 up"; return 0; }
    kill -0 "$(cat "$RUN/$1.pid")" 2>/dev/null || { err "$1 died — see $RUN/$1.log"; tail -n 20 "$RUN/$1.log" >&2; return 1; }
    sleep 0.5
  done
  err "$1 not responding — see $RUN/$1.log"; return 1
}

wait_up backend  "http://localhost:$BACK_PORT/api/models" || { "$ROOT/shutdown.sh"; exit 1; }
wait_up frontend "http://localhost:$FRONT_PORT"           || { "$ROOT/shutdown.sh"; exit 1; }

echo
ok "App ready:"
echo "   Frontend  http://localhost:$FRONT_PORT"
echo "   Backend   http://localhost:$BACK_PORT/api/models"
echo "   Logs      $RUN/{backend,frontend}.log"
echo "   Stop      ./shutdown.sh"
