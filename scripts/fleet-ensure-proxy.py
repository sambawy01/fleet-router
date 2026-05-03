#!/usr/bin/env python3
"""Idempotent boot for the Fleet Router proxy.

Designed to be invoked from a Claude Code SessionStart hook. Concurrent
SessionStart hooks (e.g. two chats opening at once) coordinate via flock
so only one process actually launches `fleet --serve`.

Exit codes:
    0  proxy is healthy
    1  proxy failed to come up within deadline
    2  fleet binary not found

The proxy itself is detached via os.setsid so it survives this script
exiting and the hook closing its stdio.
"""
from __future__ import annotations

import errno
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path("/Users/bistrocloud/fleet-router")
VENV_FLEET = REPO_ROOT / "venv" / "bin" / "fleet"

PORT = int(os.environ.get("FLEET_PORT", "8765"))
API_KEY = os.environ.get("FLEET_API_KEY", "fleet-local")

TMPDIR = Path(os.environ.get("TMPDIR", "/tmp"))
PIDFILE = TMPDIR / "fleet-proxy.pid"
LOGFILE = TMPDIR / "fleet-proxy.log"
LOCKFILE = TMPDIR / "fleet-ensure-proxy.lock"

HEALTH_URL = f"http://127.0.0.1:{PORT}/healthz"
# Cold start can include sentence-transformers download/load; give it room.
BOOT_DEADLINE_S = int(os.environ.get("FLEET_BOOT_DEADLINE_S", "60"))


def log(msg: str) -> None:
    print(f"[fleet-ensure-proxy] {msg}", file=sys.stderr)


def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError as exc:
        return exc.errno == errno.EPERM
    return True


def read_pidfile() -> int | None:
    try:
        return int(PIDFILE.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def healthz_ok(timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=timeout) as resp:
            if resp.status != 200:
                return False
            payload = json.loads(resp.read().decode("utf-8"))
            return bool(payload.get("ok"))
    except (urllib.error.URLError, socket.timeout, OSError, ValueError):
        return False


def port_in_use() -> bool:
    """Detects an unrelated process holding the port — distinct from a
    healthy fleet proxy responding on /healthz."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        try:
            sock.connect(("127.0.0.1", PORT))
            return True
        except OSError:
            return False


def already_running() -> bool:
    pid = read_pidfile()
    if pid is None or not is_pid_alive(pid):
        return False
    return healthz_ok()


def spawn_proxy() -> int:
    if not VENV_FLEET.exists():
        log(f"fleet binary not found at {VENV_FLEET}")
        sys.exit(2)

    log_fh = open(LOGFILE, "ab", buffering=0)
    proc = subprocess.Popen(
        [str(VENV_FLEET), "--serve", "--port", str(PORT), "--api-key", API_KEY],
        stdout=log_fh,
        stderr=log_fh,
        stdin=subprocess.DEVNULL,
        # Detach so the proxy outlives this script and the hook's stdio.
        start_new_session=True,
        close_fds=True,
    )
    PIDFILE.write_text(f"{proc.pid}\n")
    return proc.pid


def wait_for_health(deadline_s: int) -> bool:
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        if healthz_ok():
            return True
        time.sleep(1.0)
    return False


def main() -> int:
    if already_running():
        log(f"proxy already healthy on port {PORT}")
        return 0

    LOCKFILE.touch(exist_ok=True)
    with open(LOCKFILE, "r+") as lock_fh:
        try:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            # Another SessionStart is already booting the proxy — wait for it.
            log("another instance is starting the proxy; waiting for health")
            if wait_for_health(BOOT_DEADLINE_S):
                return 0
            log(f"proxy did not become healthy within {BOOT_DEADLINE_S}s")
            return 1

        # We hold the exclusive lock — re-check under the lock and start.
        if already_running():
            log("proxy became healthy while acquiring lock")
            return 0

        # Stale pidfile or unrelated port holder?
        pid = read_pidfile()
        if pid is not None and is_pid_alive(pid):
            # Process is alive but /healthz is not OK — kill and respawn.
            log(f"pid {pid} alive but unhealthy; terminating")
            try:
                os.kill(pid, signal.SIGTERM)
                for _ in range(20):
                    if not is_pid_alive(pid):
                        break
                    time.sleep(0.25)
                if is_pid_alive(pid):
                    os.kill(pid, signal.SIGKILL)
            except OSError as exc:
                log(f"could not terminate pid {pid}: {exc}")

        if port_in_use():
            log(f"port {PORT} held by an unknown process; not auto-killing")
            return 1

        new_pid = spawn_proxy()
        log(f"spawned fleet --serve (pid {new_pid}); polling {HEALTH_URL}")

    if wait_for_health(BOOT_DEADLINE_S):
        log("proxy is healthy")
        return 0
    log(f"proxy did not become healthy within {BOOT_DEADLINE_S}s — see {LOGFILE}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
