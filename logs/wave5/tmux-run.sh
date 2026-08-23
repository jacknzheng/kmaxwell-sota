#!/bin/bash
# Run a command in the detached `persist` tmux session so it survives
# Cursor / SSH disconnect. Usage:
#   logs/wave5/tmux-run.sh window-name -- command args...
set -euo pipefail
SESSION=persist
if [[ $# -lt 2 ]]; then
  echo "usage: $0 window-name -- command args..." >&2
  exit 2
fi
name=$1
shift
if [[ "${1:-}" == "--" ]]; then
  shift
fi
if [[ $# -lt 1 ]]; then
  echo "usage: $0 window-name -- command args..." >&2
  exit 2
fi
if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux new-session -d -s "$SESSION" -n status
fi
if tmux list-windows -t "$SESSION" -F '#{window_name}' | grep -Fxq "$name"; then
  echo "window $name already exists in session $SESSION" >&2
  exit 1
fi
tmux new-window -t "$SESSION" -n "$name" -- "$@"
echo "started in tmux $SESSION:$name  (attach: tmux attach -t $SESSION)"
