#!/usr/bin/env bash
set -Eeuo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# FramePack launcher (WebUI + API) with terminal titles
#
# Usage examples:
#   ./framepack-launch.sh --webui --api
#   ./framepack-launch.sh --webui --webui-port 7863 --server 127.0.0.1
#   ./framepack-launch.sh --api --api-port 7000 --unload
#   ./framepack-launch.sh --write-autostart --webui --api --webui-port 7863 --api-port 7000
#
# Flags:
#   --webui             Launch demo_gradio.py (UI)
#   --api               Launch api.py (FastAPI service)
#   --webui-port <n>    Port for WebUI (default: 7863)
#   --api-port <n>      Port for API   (default: 7000)
#   --server <ip>       Bind address for WebUI (default: 127.0.0.1)
#   --unload            Pass --unload to API (free model after each request)
#   --write-autostart   Write/update ~/.config/autostart/framepack.desktop
#   --dry-run           Print the spawn commands without running them
#
# Window titles:
#   We set the window title using the xterm/OSC sequence, which is widely
#   supported (including GNOME Terminal). This is more reliable than legacy
#   --title flags across versions.
# ─────────────────────────────────────────────────────────────────────────────

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

want_webui=0
want_api=0
webui_port=7863
api_port=7000
server_ip=127.0.0.1
unload_flag=0
write_autostart=0
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --webui) want_webui=1; shift;;
    --api) want_api=1; shift;;
    --webui-port) webui_port="${2:?}"; shift 2;;
    --api-port) api_port="${2:?}"; shift 2;;
    --server) server_ip="${2:?}"; shift 2;;
    --unload) unload_flag=1; shift;;
    --write-autostart) write_autostart=1; shift;;
    --dry-run) dry_run=1; shift;;
    -h|--help)
      sed -n '1,160p' "$0"
      exit 0;;
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

if [[ $want_webui -eq 0 && $want_api -eq 0 && $write_autostart -eq 0 ]]; then
  echo "Nothing to do. Use --webui and/or --api (or --write-autostart)." >&2
  exit 2
fi

# ── Git update ────────────────────────────────────────────────────────────────
if git rev-parse --git-dir >/dev/null 2>&1; then
  echo "[INFO] Updating repo (git pull --ff-only)…"
  git fetch --all --prune || true
  if ! git pull --ff-only; then
    echo "[WARN] Non-fast-forward. Please resolve manually." >&2
  fi
else
  echo "[WARN] Not a git repo. Skipping update."
fi

# ── Python venv and dependencies ─────────────────────────────────────────────
if [[ ! -d ".venv" ]]; then
  echo "[INFO] Creating venv…"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip wheel setuptools

if [[ -f "requirements.txt" ]]; then
  echo "[INFO] Installing requirements…"
  python -m pip install -U -r requirements.txt
else
  echo "[WARN] requirements.txt not found, skipping."
fi

# ── Build command list (Title + Command) ─────────────────────────────────────
mkdir -p logs

entries=()  # each entry is "TITLE:::COMMAND" (we split via parameter expansion)

if [[ $want_webui -eq 1 ]]; then
  entries+=("FramePack WebUI:::PYTHONUNBUFFERED=1 python demo_gradio.py --server ${server_ip} --port ${webui_port}")
fi

if [[ $want_api -eq 1 ]]; then
  api_cmd="PYTHONUNBUFFERED=1 python api.py --api --port ${api_port}"
  [[ $unload_flag -eq 1 ]] && api_cmd+=" --unload"
  entries+=("FramePack API:::${api_cmd}")
fi

# Prefer gnome-terminal for visible windows; fallback to x-terminal-emulator or headless
term=""
if command -v gnome-terminal >/dev/null 2>&1; then
  term="gnome-terminal --window -- bash -lc"
elif command -v x-terminal-emulator >/dev/null 2>&1; then
  term="x-terminal-emulator -e bash -lc"
fi

if [[ $dry_run -eq 1 ]]; then
  echo "[DRY-RUN] Would start:"
  for entry in "${entries[@]}"; do
    title="${entry%%:::*}"       # split at first occurrence of ':::'
    command="${entry#*:::}"      # remainder after ':::'
    echo "  [$title] $command"
  done
  exit 0
fi

for entry in "${entries[@]}"; do
  title="${entry%%:::*}"         # safe split (no IFS) → avoids the '::cmd not found' bug
  command="${entry#*:::}"

  ts=$(date +%Y%m%d_%H%M%S)

  # Derive a log name from the first *.py token in the command
  script_base="process"
  for tok in $command; do
    if [[ "$tok" == *.py ]]; then
      base="${tok##*/}"
      script_base="${base%.py}"
      break
    fi
  done
  log="logs/${script_base}_${ts}.log"

  echo "[INFO] Starting [$title]: $command"

  if [[ -n "$term" ]]; then
    # Set the window title with the xterm/OSC sequence, then run the command.
    # Keep the terminal window open after exit to show logs.
    eval "$term 'echo -ne \"\033]0;${title}\007\"; $command | tee -a \"$log\"; echo; echo Press ENTER to close…; read -r _;'" &
  else
    # Headless fallback
    nohup bash -lc "echo -ne '\033]0;${title}\007'; $command >>\"$log\" 2>&1" >/dev/null 2>&1 &
  fi

  sleep 0.3
done

# ── Autostart .desktop (freedesktop) ─────────────────────────────────────────
if [[ $write_autostart -eq 1 ]]; then
  autostart_dir="${XDG_CONFIG_HOME:-$HOME/.config}/autostart"
  mkdir -p "$autostart_dir"
  desktop_file="${autostart_dir}/framepack.desktop"

  exec_args=()
  [[ $want_webui -eq 1 ]] && exec_args+=(--webui --webui-port "$webui_port" --server "$server_ip")
  [[ $want_api -eq 1 ]] && exec_args+=(--api --api-port "$api_port")
  [[ $unload_flag -eq 1 ]] && exec_args+=(--unload)

  # The script itself sets the window title via OSC, so autostart just re-runs it.
  exec_line="gnome-terminal --window -- bash -lc 'cd \"$HERE\" && ./$(basename "$0") ${exec_args[*]}; exec bash'"

  cat >"$desktop_file" <<EOF
[Desktop Entry]
Type=Application
Name=FramePack
Comment=Start FramePack WebUI/API in venv (visible terminal)
Exec=${exec_line}
Terminal=false
Path=${HERE}
X-GNOME-Autostart-enabled=true
EOF

  echo "[INFO] Wrote autostart file: $desktop_file"
  echo "      (Keys conform to the freedesktop.org Desktop Entry spec.)"
fi
