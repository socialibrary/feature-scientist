#!/usr/bin/env python3
"""Arena UI server (Days 7-8) — demo dashboard for the Feature Scientist.

Stdlib only: no pip dependencies. Reads the append-only experiment ledger at
results/experiments.jsonl and serves:

    /               the dashboard (src/arena_ui.html)
    /api/ledger     JSON list of ledger entries, each with a computed `category`
    /api/summary    baseline AUC, best delta, verdict counts, round list

Run:  python3 src/arena.py [--port 8765]
Then: open http://localhost:8765  (and run the scientist in another terminal:
      .venv/bin/python src/run_scientist.py --rounds 4)

The page polls /api/ledger every 3s, so the dashboard updates LIVE while the
agent loop runs. A missing or empty ledger is handled gracefully ("waiting
for first run").
"""

from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")
LEDGER_PATH = os.path.join(ROOT, "results", "experiments.jsonl")
BASELINE_PATH = os.path.join(ROOT, "results", "two_tower_baseline.json")
UI_PATH = os.path.join(BASE, "arena_ui.html")

FALLBACK_BASELINE_AUC = 0.7410


def load_ledger(path: str = LEDGER_PATH) -> list[dict]:
    """Parse the jsonl ledger; skip blank/torn lines so a live-writing
    run_scientist.py never breaks the API."""
    if not os.path.exists(path):
        return []
    entries: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # torn line from a concurrent writer; next poll picks it up
    return entries


def categorize(entry: dict) -> str:
    """accepted | rejected:leakage | rejected:cost | rejected:no-lift"""
    if entry.get("verdict") == "accepted":
        return "accepted"
    audit = entry.get("leakage_audit") or {}
    if not audit.get("passed", True):
        return "rejected:leakage"
    cost = entry.get("serving_cost")
    if cost is not None and not cost.get("passed", True):
        return "rejected:cost"
    return "rejected:no-lift"


def with_category(entries: list[dict]) -> list[dict]:
    return [{**e, "category": categorize(e)} for e in entries]


def load_baseline_auc(path: str = BASELINE_PATH) -> float:
    try:
        with open(path) as f:
            return float(json.load(f)["valid_auc"])
    except (OSError, ValueError, KeyError, TypeError):
        return FALLBACK_BASELINE_AUC


def build_summary(ledger_path: str = LEDGER_PATH,
                  baseline_path: str = BASELINE_PATH) -> dict:
    entries = with_category(load_ledger(ledger_path))
    baseline_auc = load_baseline_auc(baseline_path)
    counts = {"total": len(entries), "accepted": 0,
              "rejected:leakage": 0, "rejected:cost": 0, "rejected:no-lift": 0}
    best_delta = None
    best_feature = None
    rounds: list[int] = []
    last_ts = None
    for e in entries:
        counts[e["category"]] = counts.get(e["category"], 0) + 1
        r = e.get("round")
        if isinstance(r, int) and r not in rounds:
            rounds.append(r)
        if e.get("ts"):
            last_ts = e["ts"]
        m = e.get("metrics") or {}
        d = m.get("delta_auc")
        if isinstance(d, (int, float)):
            if best_delta is None or d > best_delta:
                best_delta = d
                best_feature = e.get("feature_name")
    return {
        "baseline_auc": baseline_auc,
        "best_delta_auc": best_delta,
        "best_feature": best_feature,
        "counts": counts,
        "rounds": sorted(rounds),
        "n_entries": len(entries),
        "last_ts": last_ts,
        "status": "live" if entries else "waiting",
    }


class ArenaHandler(BaseHTTPRequestHandler):
    server_version = "Arena/1.0"

    def _send_json(self, obj, code: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, code: int = 200) -> None:
        try:
            with open(UI_PATH, "rb") as f:
                body = f.read()
        except OSError:
            self._send_json({"error": "arena_ui.html not found"}, 500)
            return
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (http.server convention)
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send_html()
        elif path == "/api/ledger":
            self._send_json({"entries": with_category(load_ledger())})
        elif path == "/api/summary":
            self._send_json(build_summary())
        else:
            self._send_json({"error": "not found"}, 404)

    def log_message(self, fmt, *args):  # keep demo output clean
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Feature Scientist arena dashboard")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), ArenaHandler)
    print(f"Arena live at http://{args.host}:{args.port}  "
          f"(ledger: {LEDGER_PATH})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
