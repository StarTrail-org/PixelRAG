#!/usr/bin/env python3
"""Cross-machine monitoring dashboard.

Reads shard claims from S3 to display global progress across all machines.

Usage:
    # One-shot status
    pixelrag-monitor --bucket my-bucket

    # Watch mode (refresh every 30s)
    pixelrag-monitor --bucket my-bucket --watch
"""

import argparse
import collections
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from .distributed import S3ShardCoordinator

# ── ANSI colors ──────────────────────────────────────────────────────
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
RESET = "\033[0m"

BAR_FILL = "\u2588"  # █
BAR_EMPTY = "\u2591"  # ░


def _no_color():
    global BOLD, DIM, GREEN, YELLOW, RED, CYAN, RESET
    BOLD = DIM = GREEN = YELLOW = RED = CYAN = RESET = ""


def _progress_bar(fraction: float, width: int = 40) -> str:
    filled = int(fraction * width)
    return f"{GREEN}{BAR_FILL * filled}{DIM}{BAR_EMPTY * (width - filled)}{RESET}"


def _format_duration(seconds: float) -> str:
    if seconds < 0:
        return "?"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    return f"{m}m"


def _format_rate(rate: float) -> str:
    if rate >= 1.0:
        return f"{rate:.1f}/s"
    if rate >= 1 / 60:
        return f"{rate * 60:.1f}/m"
    return f"{rate * 3600:.1f}/h"


def _shorten_machine(name: str, max_len: int = 24) -> str:
    if len(name) <= max_len:
        return name
    return name[: max_len - 1] + "\u2026"


def _extract_host(machine_id: str) -> str:
    """Extract physical hostname from machine_id (``hostname-PID``).

    Machine IDs look like ``ip-172-31-2-101-4188035`` (AWS) or
    ``129-80-136-19-14506`` (Lambda).  The PID is always the last
    ``-``-separated segment and is ≥5 digits on Linux.
    """
    import re

    # AWS: ip-X-X-X-X-PID
    m = re.match(r"(ip-\d+-\d+-\d+-\d+)-\d+$", machine_id)
    if m:
        return m.group(1)
    # Bare IP: X-X-X-X-PID
    m = re.match(r"(\d+-\d+-\d+-\d+)-\d+$", machine_id)
    if m:
        return m.group(1)
    return machine_id


# ── Tile validation (Gemini Vision via validate_tiles.py) ────────

_VALIDATE_SCRIPT = str(Path(__file__).resolve().parent / "validate_tiles.py")
_ENV_FILE = Path(__file__).resolve().parent.parent.parent.parent.parent / ".env"

# Load .env at import time — override=True so .env wins over shell env
# (fish shell may set a different GOOGLE_API_KEY by default)
try:
    from dotenv import load_dotenv

    load_dotenv(_ENV_FILE, override=True)
except ImportError:
    pass


def _validate_env() -> dict[str, str]:
    """Build env dict with GOOGLE_API_KEY for validate_tiles.py subprocess."""
    env = os.environ.copy()
    env.pop("GEMINI_API_KEY", None)  # avoid "both keys set" warning
    return env


def _parse_new_jsonl(results_file: Path, lines_before: int, state: dict, cycle: dict):
    """Parse newly appended lines from a JSONL results file into state/cycle."""
    if not results_file.exists():
        return
    with open(results_file) as f:
        for i, line in enumerate(f):
            if i < lines_before:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            issues = rec.get("issues", [])
            # Don't count API_ERROR as quality failures (transient network issues)
            if issues == ["API_ERROR"]:
                continue
            state["tiles_checked"] += 1
            cycle["tiles_checked"] += 1
            if not rec.get("pass", False):
                state["tiles_failed"] += 1
                cycle["tiles_failed"] += 1
                for issue in issues:
                    state["issues"][issue] += 1
                    cycle["issues"][issue] += 1


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for _ in f)


def _run_validate_tiles(
    output_dir: str,
    sample: int,
    shard_ids: list[int],
    state: dict,
    model: str | None = None,
) -> dict:
    """Run validate_tiles.py locally on given shards and parse JSONL results."""
    cycle = {"tiles_checked": 0, "tiles_failed": 0, "issues": collections.Counter()}
    if not shard_ids:
        return cycle

    results_file = state["results_file"]
    # --shard uses nargs="+", so place it before --concurrency which
    # terminates the greedy list (otherwise "local" gets consumed as a shard).
    cmd = [
        sys.executable,
        _VALIDATE_SCRIPT,
        "--sample",
        str(sample),
        "--shard",
        *(str(s) for s in shard_ids),
        "--concurrency",
        "10",
        "--seed",
        str(int(time.time())),
        "--resume",
        str(results_file),
    ]
    if model:
        cmd += ["--model", model]
    cmd += ["local", output_dir]

    try:
        lines_before = _count_lines(results_file)
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, env=_validate_env()
        )
        if proc.returncode not in (0, 1):
            return cycle
        _parse_new_jsonl(results_file, lines_before, state, cycle)
    except (subprocess.TimeoutExpired, Exception):
        pass

    return cycle


def _run_validate_tiles_s3(
    bucket: str,
    prefix: str,
    sample: int,
    shard_ids: list[int],
    state: dict,
    model: str | None = None,
) -> dict:
    """Run validate_tiles.py with S3 source on given shards and parse JSONL results."""
    cycle = {"tiles_checked": 0, "tiles_failed": 0, "issues": collections.Counter()}
    if not shard_ids:
        return cycle

    results_file = state["results_file_s3"]
    cmd = [
        sys.executable,
        _VALIDATE_SCRIPT,
        "--sample",
        str(sample),
        "--shard",
        *(str(s) for s in shard_ids),
        "--concurrency",
        "10",
        "--seed",
        str(int(time.time())),
        "--resume",
        str(results_file),
    ]
    if model:
        cmd += ["--model", model]
    cmd += ["s3", "--bucket", bucket, "--s3-prefix", prefix]

    try:
        lines_before = _count_lines(results_file)
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, env=_validate_env()
        )
        if proc.returncode not in (0, 1):
            return cycle
        _parse_new_jsonl(results_file, lines_before, state, cycle)
    except (subprocess.TimeoutExpired, Exception):
        pass

    return cycle


def _parse_ssh_spec(spec: str) -> tuple[str, str]:
    """Parse ``user@host:/project/dir`` → ``('user@host', '/project/dir')``.

    If no ``:path`` part, defaults to ``~/pixelrag-index``.
    """
    colon_idx = spec.find(":/")
    if colon_idx >= 0:
        return spec[:colon_idx], spec[colon_idx + 1 :]
    return spec, "~/pixelrag-index"


def _run_validate_tiles_ssh(
    ssh_spec: str,
    sample: int,
    shard_ids: list[int],
    state: dict,
    model: str | None = None,
) -> dict:
    """SSH to a remote machine, run validate_tiles.py on given shards, scp results back.

    ssh_spec format: ``user@host:/project/dir`` (project dir contains
    ``output_coordinated/`` and ``validation/``).
    """
    cycle = {"tiles_checked": 0, "tiles_failed": 0, "issues": collections.Counter()}
    if not shard_ids:
        return cycle

    ssh_host, project_dir = _parse_ssh_spec(ssh_spec)
    local_results = state["results_file_ssh"][ssh_spec]

    remote_output = f"{project_dir}/output_coordinated"
    remote_python = f"{project_dir}/validation/.venv/bin/python"
    remote_script = f"{project_dir}/validation/validate_tiles.py"
    remote_results = f"{project_dir}/validation/results/monitor_validation.jsonl"
    remote_results_dir = f"{project_dir}/validation/results"

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    seed = str(int(time.time()))
    shard_args = " ".join(str(s) for s in shard_ids)

    model_arg = f" --model {model}" if model else ""
    remote_cmd = (
        f"mkdir -p {remote_results_dir} && rm -f {remote_results} && "
        f"GOOGLE_API_KEY={api_key} {remote_python} {remote_script}"
        f" --sample {sample} --concurrency 10 --seed {seed}"
        f"{model_arg}"
        f" --shard {shard_args}"
        f" local {remote_output}"
    )

    ssh_cmd = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=10",
        ssh_host,
        remote_cmd,
    ]

    try:
        proc = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode not in (0, 1):
            return cycle

        # SCP the remote results file back
        lines_before = _count_lines(local_results)
        scp_cmd = [
            "scp",
            "-o",
            "StrictHostKeyChecking=no",
            "-q",
            f"{ssh_host}:{remote_results}",
            str(local_results),
        ]
        subprocess.run(scp_cmd, capture_output=True, timeout=30)
        _parse_new_jsonl(local_results, lines_before, state, cycle)
    except (subprocess.TimeoutExpired, Exception):
        pass

    return cycle


def render(
    coord: S3ShardCoordinator,
    prev_articles: int | None,
    prev_tiles: int | None,
    prev_time: float | None,
    verbose: bool = False,
    prev_machine_tiles: dict[str, int] | None = None,
    window_rates: dict[str, float] | None = None,
):
    """Fetch status and render one dashboard frame.

    Args:
        window_rates: Per-machine tiles/s from a 10-minute sliding window
            (computed in main() across refreshes).  Used for Rate column and
            global rate when available; falls back to session-based estimate.
    """
    now = time.time()
    status = coord.get_status()
    claims = status["claims"]
    total_articles = coord._manifest["total"]
    total_shards = status["total_shards"]
    articles_done = status["articles_done"]
    pct = articles_done / total_articles if total_articles else 0

    # ── per-machine aggregation ──────────────────────────────────────
    machines: dict[str, dict] = defaultdict(
        lambda: {
            "shards_done": 0,
            "shards_active": 0,
            "shards_stale": 0,
            "completed": 0,
            "failed": 0,
            "skipped": 0,
            "tiles": 0,
            "earliest_claim": float("inf"),
            "latest_heartbeat": 0,
            "current_shards": [],
            "in_flight": [],  # article IDs currently being processed
            "recent_errors": [],  # last N errors from active shard
            "disk_free_gb": None,  # latest disk_free_gb from heartbeat
        }
    )
    # Per-shard source info: {shard_id: {"machine": ..., "s3_sync": bool|None}}
    shard_source_info: dict[int, dict] = {}

    total_tiles = 0
    total_failed = 0
    for c in claims:
        host = _extract_host(c["machine"])
        m = machines[host]
        m["completed"] += c.get("completed", 0)
        m["failed"] += c.get("failed", 0)
        m["skipped"] += c.get("skipped", 0)
        total_failed += c.get("failed", 0)
        c_tiles = c.get("tiles", 0)
        m["tiles"] += c_tiles
        total_tiles += c_tiles
        claimed_at = c.get("claimed_at", 0)
        if claimed_at and claimed_at < m["earliest_claim"]:
            m["earliest_claim"] = claimed_at
        hb = c.get("heartbeat", 0)
        m["latest_heartbeat"] = max(m["latest_heartbeat"], hb)

        # Collect disk_free_gb (use the latest heartbeat value)
        if "disk_free_gb" in c:
            hb = c.get("heartbeat", 0)
            cur = m.get("_disk_hb", 0)
            if hb >= cur:
                m["disk_free_gb"] = c["disk_free_gb"]
                m["_disk_hb"] = hb

        # Track per-shard source info for S3 vs local validation routing
        sid = c.get("shard_id")
        if sid is not None:
            shard_source_info[sid] = {
                "machine": host,
                "s3_sync": c.get("s3_sync"),
            }

        if c["status"] == "completed":
            m["shards_done"] += 1
        elif c["status"] == "in_progress":
            age = now - c.get("heartbeat", 0)
            if age > coord.stale_timeout:
                m["shards_stale"] += 1
            else:
                m["shards_active"] += 1
                m["current_shards"].append(c.get("shard_id", "?"))
                # Carry in-flight articles and error info from active shard
                if c.get("in_flight"):
                    m["in_flight"].extend(c["in_flight"])
                if c.get("recent_errors"):
                    m["recent_errors"].extend(c["recent_errors"])

    # Per-machine tile snapshot (all machines) for instantaneous rate calc
    machine_tiles_snapshot: dict[str, int] = {}
    for name, m in machines.items():
        machine_tiles_snapshot[name] = m["tiles"]

    # ── compute rates (tiles/s) ─────────────────────────────────────
    # Priority:
    #   1. window_rates (10-min sliding window from main loop) — best
    #   2. session rate (latest PID's tiles/elapsed) — one-shot fallback

    # Fallback: latest worker session tiles/elapsed (for one-shot or first cycle)
    from collections import defaultdict as _dd

    _sessions: dict[str, dict[str, list]] = _dd(lambda: _dd(list))
    for c in claims:
        host = _extract_host(c["machine"])
        _sessions[host][c["machine"]].append(c)

    machine_session_rates: dict[str, float] = {}
    for host in machines:
        best_rate = 0.0
        best_hb = 0.0
        for mid, sess_claims in _sessions.get(host, {}).items():
            s_tiles = sum(c.get("tiles", 0) for c in sess_claims)
            s_earliest = min((c.get("claimed_at", 0) for c in sess_claims), default=0)
            s_latest = max(
                (c.get("completed_at", c.get("heartbeat", 0)) for c in sess_claims),
                default=0,
            )
            s_elapsed = s_latest - s_earliest
            if s_elapsed > 0 and s_tiles > 0:
                rate = s_tiles / s_elapsed
                if s_latest > best_hb:
                    best_hb = s_latest
                    best_rate = rate
        machine_session_rates[host] = best_rate

    # Final per-machine rate: prefer window rate, fall back to session rate
    machine_rates: dict[str, float] = {}
    for host in machines:
        wr = (window_rates or {}).get(host)
        if wr is not None and wr > 0:
            machine_rates[host] = wr
        else:
            machine_rates[host] = machine_session_rates.get(host, 0.0)

    # A machine is "alive" if its latest heartbeat is within 10 minutes,
    # even if it has 0 shards_active right now (claim_next() scanning gap).
    alive_threshold = 600  # 10 minutes
    alive_hosts = {
        n
        for n, m in machines.items()
        if m["latest_heartbeat"] > 0 and (now - m["latest_heartbeat"]) < alive_threshold
    }
    global_rate = sum(machine_rates[h] for h in alive_hosts) if alive_hosts else 0.0

    # ETA (based on tiles — estimate remaining tiles from avg tiles/article)
    remaining_articles = total_articles - articles_done
    if global_rate > 0:
        avg_tiles_per_art = total_tiles / articles_done if articles_done else 1.5
        eta = remaining_articles * avg_tiles_per_art / global_rate
    else:
        eta = -1

    # ── render ───────────────────────────────────────────────────────
    try:
        term_width = os.get_terminal_size().columns
    except (OSError, ValueError):
        term_width = 80
    w = min(term_width - 2, 86)
    bar_w = max(w - 30, 20)

    hline = "\u2500"  # ─
    line = hline * w
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("")
    lines.append(
        f"  {BOLD}Wiki-Screenshot Pipeline{RESET}{' ' * max(0, w - 50)}{DIM}{now_str}{RESET}"
    )
    lines.append(f"  {DIM}{line}{RESET}")

    # Progress bar
    bar = _progress_bar(pct, bar_w)
    lines.append(f"  {bar} {BOLD}{pct * 100:5.1f}%{RESET}")
    lines.append(
        f"  Articles  {BOLD}{articles_done:>12,}{RESET} / {total_articles:,}"
        f"    Shards  {BOLD}{status['completed']}{RESET}/{total_shards} done"
    )
    lines.append(f"  Tiles     {BOLD}{total_tiles:>12,}{RESET}")

    # Header rate: tiles/s from window or session fallback
    if global_rate > 0:
        rate_str = _format_rate(global_rate)
    else:
        rate_str = "paused"
    eta_str = (
        _format_duration(eta)
        if eta > 0
        else ("paused" if not alive_hosts else "calculating...")
    )
    total_processed = articles_done
    fail_pct = (total_failed / total_processed * 100) if total_processed > 0 else 0
    fail_color = RED if fail_pct > 5 else YELLOW if fail_pct > 1 else GREEN
    lines.append(
        f"  Rate      {BOLD}{rate_str:>12}{RESET}"
        f"    Fail  {fail_color}{fail_pct:.2f}%{RESET}"
        f"    ETA  {BOLD}{eta_str}{RESET}"
    )

    shard_detail = (
        f"{GREEN}{status['completed']} done{RESET}  "
        f"{CYAN}{status['in_progress']} active{RESET}"
    )
    if status["stale"] > 0:
        shard_detail += f"  {RED}{status['stale']} stale{RESET}"
    shard_detail += f"  {DIM}{status['unclaimed']} unclaimed{RESET}"
    lines.append(f"  Shards    {shard_detail}")

    lines.append(f"  {DIM}{line}{RESET}")

    # ── per-machine table ────────────────────────────────────────────
    name_w = 24
    header = (
        f"  {BOLD}{'Machine':<{name_w}}  {'Shards':>8}  {'Articles':>10}"
        f"  {'Tiles':>10}  {'Rate':>8}  {'Fail%':>6}  {'Disk':>7}  {'Current':>8}{RESET}"
    )
    lines.append(header)
    sep_w = name_w + 71
    lines.append(f"  {DIM}{hline * sep_w}{RESET}")

    # Sort: active machines first (by rate descending), then finished
    active_machines = {n: m for n, m in machines.items() if m["shards_active"] > 0}
    done_machines = {n: m for n, m in machines.items() if m["shards_active"] == 0}

    for name in sorted(active_machines, key=lambda n: machine_rates[n], reverse=True):
        m = active_machines[name]
        total_m = m["shards_done"] + m["shards_active"]
        arts = m["completed"] + m["failed"] + m["skipped"]
        rate = machine_rates[name]
        m_fail = (m["failed"] / arts * 100) if arts > 0 else 0
        fc = RED if m_fail > 5 else YELLOW if m_fail > 1 else ""
        fc_r = RESET if fc else ""
        cur = ",".join(str(s) for s in m["current_shards"][:3])
        if len(m["current_shards"]) > 3:
            cur += ".."

        stale_marker = f" {RED}!{RESET}" if m["shards_stale"] > 0 else ""
        disk_gb = m.get("disk_free_gb")
        if disk_gb is not None:
            disk_color = RED if disk_gb < 100 else YELLOW if disk_gb < 200 else ""
            disk_r = RESET if disk_color else ""
            disk_str = f"{disk_color}{disk_gb:>5.0f}G{disk_r}"
        else:
            disk_str = f"{DIM}     ?{RESET}"
        lines.append(
            f"  {CYAN}{_shorten_machine(name, name_w):<{name_w}}{RESET}"
            f"  {m['shards_done']:>3}/{total_m:<4}"
            f"  {arts:>10,}"
            f"  {m['tiles']:>10,}"
            f"  {_format_rate(rate):>8}"
            f"  {fc}{m_fail:>5.1f}%{fc_r}"
            f"  {disk_str:>7}"
            f"  {DIM}#{cur}{RESET}{stale_marker}"
        )

        # Verbose: in-flight articles
        if verbose and m["in_flight"]:
            for i, aid in enumerate(m["in_flight"][:5]):
                prefix = "\u2514" if i == min(len(m["in_flight"]), 5) - 1 else "\u251c"
                label = aid if len(aid) <= 40 else aid[:39] + "\u2026"
                lines.append(f"    {DIM}{prefix} {label}{RESET}")
            if len(m["in_flight"]) > 5:
                lines.append(
                    f"    {DIM}\u2514 ...and {len(m['in_flight']) - 5} more{RESET}"
                )

        # Verbose: recent errors
        if verbose and m["recent_errors"]:
            err_counts: dict[str, int] = {}
            for e in m["recent_errors"]:
                short = e[:50]
                err_counts[short] = err_counts.get(short, 0) + 1
            top_errs = sorted(err_counts.items(), key=lambda x: -x[1])[:3]
            err_parts = [f"{c}x {e}" for e, c in top_errs]
            lines.append(f"    {DIM}\u2514 errors: {'; '.join(err_parts)}{RESET}")

    for name in sorted(
        done_machines, key=lambda n: machines[n]["completed"], reverse=True
    ):
        m = done_machines[name]
        total_m = m["shards_done"]
        if total_m == 0:
            continue
        arts = m["completed"] + m["failed"] + m["skipped"]
        rate = machine_rates[name]
        m_fail = (m["failed"] / arts * 100) if arts > 0 else 0
        lines.append(
            f"  {DIM}{_shorten_machine(name, name_w):<{name_w}}"
            f"  {total_m:>3}/{total_m:<4}"
            f"  {arts:>10,}"
            f"  {m['tiles']:>10,}"
            f"  {_format_rate(rate):>8}"
            f"  {m_fail:>5.1f}%"
            f"          done{RESET}"
        )

    lines.append(f"  {DIM}{hline * sep_w}{RESET}")
    n_active = len(active_machines)
    n_total = len(
        [n for n, m in machines.items() if m["shards_done"] + m["shards_active"] > 0]
    )
    lines.append(
        f"  {BOLD}{n_active}{RESET} active / {n_total} total machines"
        f"      {BOLD}{_format_rate(global_rate)}{RESET} combined"
    )
    lines.append("")

    # Collect shard IDs for validation: active shards + recent completed shards
    # from alive machines (so validation runs even during claim gaps).
    active_shard_ids: list[int] = []
    for m in active_machines.values():
        for sid in m["current_shards"]:
            try:
                active_shard_ids.append(int(sid))
            except (ValueError, TypeError):
                pass
    # Also include recently completed shards from alive machines for validation.
    # Sample a small random subset to avoid passing hundreds of shards.
    import random as _rng

    validate_shard_ids: list[int] = list(active_shard_ids)
    if not validate_shard_ids:
        _completed_sids = []
        for c in claims:
            host = _extract_host(c["machine"])
            if host in alive_hosts and c["status"] == "completed":
                sid = c.get("shard_id")
                if sid is not None:
                    _completed_sids.append(sid)
        # Pick up to 5 random completed shards for validation
        if _completed_sids:
            validate_shard_ids = _rng.sample(
                _completed_sids, min(5, len(_completed_sids))
            )

    # Per-machine tile counts (only active machines, keyed by machine name)
    # Used by throughput alerting in main()
    machine_tiles: dict[str, int] = {}
    for name, m in active_machines.items():
        machine_tiles[name] = m["tiles"]

    # Per-machine disk info (all machines with disk_free_gb reported)
    machine_disk: dict[str, float | None] = {}
    for name, m in machines.items():
        if m["shards_active"] > 0 or m["shards_done"] > 0:
            machine_disk[name] = m.get("disk_free_gb")

    return (
        "\n".join(lines),
        articles_done,
        total_tiles,
        now,
        n_active,
        active_shard_ids,
        machine_tiles,
        machine_disk,
        shard_source_info,
        machine_tiles_snapshot,
        validate_shard_ids,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Cross-machine monitoring dashboard for pixelrag-index"
    )
    _DEFAULT_OUTPUT_DIR = "./index"

    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", default="kiwix")
    parser.add_argument("--watch", action="store_true", help="Refresh every 30s")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--no-color", action="store_true", help="Disable color output")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show in-flight articles and error distribution",
    )
    parser.add_argument(
        "--alert-min-tiles-per-min",
        type=int,
        default=1000,
        help=(
            "Minimum tiles/min per active machine. If total throughput stays "
            "below (N * active_machines) for 10 minutes, exit with code 1. "
            "Set to 0 to disable. Only works with --watch. (default: 1000)"
        ),
    )
    # Tile validation (Gemini Vision via validate_tiles.py)
    parser.add_argument(
        "--validate-output-dir",
        type=str,
        default=_DEFAULT_OUTPUT_DIR,
        help=f"Output directory for Gemini tile validation (default: {_DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--validate-sample",
        type=int,
        default=50,
        help="Number of tiles to sample per validation cycle (default: 50)",
    )
    parser.add_argument(
        "--validate-max-fail-pct",
        type=float,
        default=5.0,
        help="Exit(1) if Gemini fail percentage exceeds this (default: 10.0)",
    )
    parser.add_argument(
        "--validate-interval",
        type=int,
        default=60,
        help="Run validation every N watch cycles (default: 60, i.e. ~30min at 30s interval)",
    )
    parser.add_argument(
        "--validate-ssh",
        type=str,
        nargs="*",
        default=None,
        help=(
            "SSH specs for remote validation: user@host:/project/dir. "
            "If no :/path, defaults to ~/pixelrag-index. "
            "Runs validate_tiles.py on each host via SSH, scp results back. "
            "Pass with no args to disable."
        ),
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Disable Gemini tile validation (both local and SSH)",
    )

    parser.add_argument(
        "--alert-min-disk-gb",
        type=int,
        default=100,
        help=(
            "Minimum free disk space in GB on any active machine (from heartbeat data). "
            "If free space drops below this, exit with code 1. "
            "Set to 0 to disable. Only works with --watch. (default: 100)"
        ),
    )
    parser.add_argument(
        "--validate-model",
        type=str,
        default=None,
        help="Override Gemini model for validation (default: validate_tiles.py default)",
    )
    args = parser.parse_args()

    _DEFAULT_SSH_HOSTS: list[str] = []
    if args.no_validate:
        args.validate_output_dir = None
        args.validate_ssh = []
    elif args.validate_ssh is None:
        # Default: validate M2 via SSH
        args.validate_ssh = _DEFAULT_SSH_HOSTS
    # --validate-ssh with no args → empty list (disable SSH validation only)

    if args.no_color or not sys.stdout.isatty():
        _no_color()

    coord = S3ShardCoordinator(bucket=args.bucket, prefix=args.prefix)
    coord.load_manifest()

    prev_articles = None
    prev_tiles = None
    prev_time = None
    prev_machine_tiles: dict[str, int] | None = None

    # Tile validation state (Gemini Vision via validate_tiles.py)
    has_local = args.validate_output_dir is not None
    bool(args.validate_ssh)
    validate_enabled = args.watch and not args.no_validate
    results_dir = Path.cwd() / ".pixelrag-monitor" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    # Per-SSH-spec local copy of their results file
    ssh_results = {}
    for spec in args.validate_ssh or []:
        safe = (
            spec.replace("@", "_").replace(".", "-").replace("/", "_").replace(":", "_")
        )
        ssh_results[spec] = results_dir / f"monitor_validation_ssh_{safe}.jsonl"
    validate_state: dict = {
        "tiles_checked": 0,
        "tiles_failed": 0,
        "issues": collections.Counter(),
        "results_file": results_dir / "monitor_validation.jsonl",
        "results_file_ssh": ssh_results,
        "results_file_s3": results_dir / "monitor_validation_s3.jsonl",
    }
    watch_cycle = 0

    # Per-machine sliding windows for rate calculation + throughput alerting.
    # Tracks (timestamp, total_tiles) snapshots for ALL machines (not just active).
    throughput_alert_enabled = args.watch and args.alert_min_tiles_per_min > 0
    disk_alert_enabled = args.watch and args.alert_min_disk_gb > 0
    rate_window_sec = 600  # 10 minutes
    maxlen = (rate_window_sec // max(args.interval, 1)) + 2
    tile_history_per_machine: dict[str, collections.deque] = defaultdict(
        lambda: collections.deque(maxlen=maxlen)
    )
    window_rates: dict[str, float] = {}  # per-machine tiles/s from sliding window

    while True:
        (
            output,
            prev_articles,
            prev_tiles,
            prev_time,
            n_active_machines,
            active_shard_ids,
            machine_tiles,
            machine_disk,
            shard_source_info,
            machine_tiles_snapshot,
            validate_shard_ids,
        ) = render(
            coord,
            prev_articles,
            prev_tiles,
            prev_time,
            verbose=args.verbose,
            prev_machine_tiles=prev_machine_tiles,
            window_rates=window_rates,
        )
        prev_machine_tiles = machine_tiles_snapshot

        # ── Update sliding window for all machines ────────────────────
        now_ts = prev_time or time.time()
        for mname, mtiles in machine_tiles_snapshot.items():
            tile_history_per_machine[mname].append((now_ts, mtiles))
        # Compute per-machine window rates
        window_rates = {}
        for mname, dq in tile_history_per_machine.items():
            if len(dq) >= 2:
                oldest_time, oldest_tiles = dq[0]
                newest_time, newest_tiles = dq[-1]
                window_sec = newest_time - oldest_time
                if window_sec > 0:
                    window_rates[mname] = (newest_tiles - oldest_tiles) / window_sec

        if args.watch:
            # Clear screen and move cursor to top
            print("\033[2J\033[H", end="")

        print(output)

        # ── Disk space alerting (from heartbeat data) ────────────────
        if disk_alert_enabled and machine_disk:
            for mname, free_gb in machine_disk.items():
                if free_gb is None:
                    continue  # old worker, no disk_free_gb in heartbeat
                if free_gb < args.alert_min_disk_gb:
                    print(
                        f"\n{RED}{BOLD}ALERT:{RESET} disk space on {mname} "
                        f"is {free_gb:.1f} GB free < threshold {args.alert_min_disk_gb} GB. "
                        f"Exiting with code 1."
                    )
                    sys.exit(1)

        # ── Throughput alerting (per-machine, from window rates) ──────
        if throughput_alert_enabled and window_rates:
            threshold = args.alert_min_tiles_per_min
            for mname, rate_s in window_rates.items():
                # Only alert on machines that have enough history
                dq = tile_history_per_machine.get(mname)
                if not dq or len(dq) < 2:
                    continue
                window_sec = dq[-1][0] - dq[0][0]
                if window_sec < rate_window_sec:
                    continue  # not enough data yet
                tiles_per_min = rate_s * 60
                if (
                    mname in {n for n, m_tiles in machine_tiles.items()}
                    and tiles_per_min < threshold
                ):
                    print(
                        f"\n{RED}{BOLD}ALERT:{RESET} machine {mname}: "
                        f"throughput {tiles_per_min:.0f} tiles/min "
                        f"< threshold {threshold} tiles/min "
                        f"over {window_sec / 60:.1f} min window. Exiting with code 1."
                    )
                    sys.exit(1)

        # ── Tile validation (Gemini Vision) ──────────────────────────
        if validate_enabled and validate_shard_ids:
            watch_cycle += 1
            if watch_cycle % args.validate_interval == 0:
                cycle_total = {"tiles_checked": 0, "tiles_failed": 0}

                # Split shards into S3-synced vs local/SSH based on heartbeat data.
                s3_shard_ids = []
                local_shard_ids = []
                for sid in validate_shard_ids:
                    info = shard_source_info.get(sid, {})
                    if info.get("s3_sync") is True:
                        s3_shard_ids.append(sid)
                    else:
                        local_shard_ids.append(sid)

                # S3 validation for s3_sync=True shards
                if s3_shard_ids:
                    c = _run_validate_tiles_s3(
                        args.bucket,
                        args.prefix,
                        args.validate_sample,
                        s3_shard_ids,
                        validate_state,
                        model=args.validate_model,
                    )
                    cycle_total["tiles_checked"] += c["tiles_checked"]
                    cycle_total["tiles_failed"] += c["tiles_failed"]

                # Local validation (non-S3 active shards on this machine)
                if has_local and local_shard_ids:
                    c = _run_validate_tiles(
                        args.validate_output_dir,
                        args.validate_sample,
                        local_shard_ids,
                        validate_state,
                        model=args.validate_model,
                    )
                    cycle_total["tiles_checked"] += c["tiles_checked"]
                    cycle_total["tiles_failed"] += c["tiles_failed"]

                # SSH validation (non-S3 shards — remote only finds ones it has)
                if local_shard_ids:
                    for spec in args.validate_ssh or []:
                        c = _run_validate_tiles_ssh(
                            spec,
                            args.validate_sample,
                            local_shard_ids,
                            validate_state,
                            model=args.validate_model,
                        )
                        cycle_total["tiles_checked"] += c["tiles_checked"]
                        cycle_total["tiles_failed"] += c["tiles_failed"]

                st = validate_state
                fail_pct = (
                    (st["tiles_failed"] / st["tiles_checked"] * 100)
                    if st["tiles_checked"] > 0
                    else 0
                )
                fail_color = (
                    RED
                    if fail_pct > args.validate_max_fail_pct
                    else YELLOW
                    if fail_pct > 1
                    else GREEN
                )
                issue_str = (
                    ", ".join(f"{k}={v}" for k, v in st["issues"].most_common(5))
                    if st["issues"]
                    else "none"
                )
                sources = []
                if s3_shard_ids:
                    sources.append(f"s3({len(s3_shard_ids)})")
                if has_local and local_shard_ids:
                    sources.append("local")
                for spec in args.validate_ssh or []:
                    host_part, _ = _parse_ssh_spec(spec)
                    sources.append(
                        host_part.split("@")[-1] if "@" in host_part else host_part
                    )
                print(
                    f"  {DIM}Validation (Gemini, {'+'.join(sources)}):{RESET} "
                    f"{st['tiles_checked']} checked, "
                    f"{fail_color}{st['tiles_failed']} failed ({fail_pct:.1f}%){RESET}"
                    f"  {DIM}(+{cycle_total['tiles_checked']} this cycle){RESET}"
                    f"  issues: {issue_str}"
                )

                if st["tiles_checked"] >= 50 and fail_pct > args.validate_max_fail_pct:
                    print(
                        f"\n{RED}{BOLD}ALERT:{RESET} Gemini fail rate {fail_pct:.1f}% "
                        f"> threshold {args.validate_max_fail_pct}% "
                        f"({st['tiles_failed']}/{st['tiles_checked']} tiles). "
                        f"Exiting with code 1."
                    )
                    sys.exit(1)

        if not args.watch:
            break
        time.sleep(args.interval)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
