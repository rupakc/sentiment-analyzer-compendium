#!/usr/bin/env bash
# ponytail: best-effort format, never block an edit
ruff format api scripts 2>/dev/null || true
(cd frontend 2>/dev/null && npx --no-install prettier -w 'src/**/*.{ts,tsx}' 2>/dev/null) || true
exit 0
