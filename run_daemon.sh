#!/usr/bin/env bash
# AL3X.NYC — 24/7 caffeinated daemon runner for macOS.
#
#   ./run_daemon.sh start   — launch in background, keep Mac awake, auto-restart on crash
#   ./run_daemon.sh stop    — graceful shutdown
#   ./run_daemon.sh restart — stop + start
#   ./run_daemon.sh status  — is it alive?
#   ./run_daemon.sh logs    — tail the live log
#
# Features:
#   • `caffeinate -dimsu` prevents display/idle/system sleep and keeps the
#     Mac acting as if user is active — survives closed lid on MacBooks
#     plugged into power (see Apple's pmset).
#   • Auto-restart on crash (max 1 restart per 10s to avoid thrash loops).
#   • Log rotation at 10 MB (keeps al3x.log + al3x.log.1).
#   • PID tracking for clean stop/restart.
#   • Exits cleanly if the user kills the supervisor.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv"
APP="$ROOT/app.py"
LOG="$ROOT/al3x.log"
PID="$ROOT/al3x.pid"
SUP_PID="$ROOT/al3x.supervisor.pid"

_activate() {
    if [[ -f "$VENV/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "$VENV/bin/activate"
    else
        echo "error: virtualenv not found at $VENV" >&2
        echo "run: python3 -m venv .venv && pip install -r requirements.txt" >&2
        exit 1
    fi
}

_rotate_log() {
    if [[ -f "$LOG" ]]; then
        local size
        size=$(stat -f%z "$LOG" 2>/dev/null || echo 0)
        # Rotate at 10 MB
        if (( size > 10485760 )); then
            mv -f "$LOG" "$LOG.1"
        fi
    fi
}

_is_running() {
    [[ -f "$1" ]] && kill -0 "$(cat "$1")" 2>/dev/null
}

start() {
    if _is_running "$SUP_PID"; then
        echo "AL3X.NYC supervisor already running (PID $(cat "$SUP_PID"))"
        exit 0
    fi

    _activate
    _rotate_log

    # Supervisor loop in a subshell so we can disown it.
    (
        echo "$BASHPID" > "$SUP_PID"
        trap 'rm -f "$SUP_PID" "$PID"; exit 0' TERM INT

        last_restart=0
        while true; do
            now=$(date +%s)
            # Thrash guard: if we just restarted <10s ago, wait longer
            if (( now - last_restart < 10 )); then
                sleep $(( 10 - (now - last_restart) ))
            fi
            last_restart=$(date +%s)

            echo "[$(date '+%Y-%m-%d %H:%M:%S')] starting AL3X.NYC" >> "$LOG"
            # caffeinate -dimsu: prevent display/idle/mask/system sleep,
            # act as if the user is active. Wraps python so caffeinate
            # exits when python does.
            caffeinate -dimsu python "$APP" >> "$LOG" 2>&1 &
            app_pid=$!
            echo "$app_pid" > "$PID"

            # Block on the child
            wait "$app_pid" || true
            exit_code=$?
            rm -f "$PID"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] app exited code=$exit_code — restarting" >> "$LOG"
        done
    ) &
    sup_pid=$!
    disown
    echo "AL3X.NYC started."
    echo "  Supervisor PID : $sup_pid"
    echo "  Log            : $LOG"
    echo "  Dashboard      : http://localhost:8090"
    echo
    echo "  Watch logs     : ./run_daemon.sh logs"
    echo "  Stop           : ./run_daemon.sh stop"
}

stop() {
    local stopped=0
    if _is_running "$SUP_PID"; then
        kill -TERM "$(cat "$SUP_PID")" 2>/dev/null || true
        stopped=1
    fi
    if _is_running "$PID"; then
        kill -TERM "$(cat "$PID")" 2>/dev/null || true
        stopped=1
    fi
    # Give it a moment, then force-kill any stragglers
    sleep 2
    if _is_running "$SUP_PID"; then
        kill -KILL "$(cat "$SUP_PID")" 2>/dev/null || true
    fi
    if _is_running "$PID"; then
        kill -KILL "$(cat "$PID")" 2>/dev/null || true
    fi
    rm -f "$SUP_PID" "$PID"
    if (( stopped == 0 )); then
        echo "AL3X.NYC was not running."
    else
        echo "AL3X.NYC stopped."
    fi
}

status() {
    local sup_alive=0 app_alive=0
    _is_running "$SUP_PID" && sup_alive=1
    _is_running "$PID" && app_alive=1

    if (( sup_alive && app_alive )); then
        echo "✓ AL3X.NYC is running"
        echo "  Supervisor PID : $(cat "$SUP_PID")"
        echo "  App PID        : $(cat "$PID")"
        echo "  Dashboard      : http://localhost:8090"
    elif (( sup_alive )); then
        echo "⚠ Supervisor alive but app is restarting"
        echo "  Supervisor PID : $(cat "$SUP_PID")"
    else
        echo "✗ AL3X.NYC is NOT running"
    fi
}

logs() {
    if [[ -f "$LOG" ]]; then
        tail -n 40 -f "$LOG"
    else
        echo "No log file at $LOG yet."
    fi
}

case "${1:-status}" in
    start)   start ;;
    stop)    stop ;;
    restart) stop; sleep 1; start ;;
    status)  status ;;
    logs)    logs ;;
    *)
        echo "usage: $0 {start|stop|restart|status|logs}" >&2
        exit 1
        ;;
esac
