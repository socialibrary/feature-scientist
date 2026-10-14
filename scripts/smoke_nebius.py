#!/usr/bin/env python3
"""Smoke-test the Nebius Token Factory backend.

Needs NEBIUS_API_KEY in the environment. Makes one tiny JSON-mode call to
the configured Nemotron model and prints the round-trip latency.

    export NEBIUS_API_KEY=...
    python3 scripts/smoke_nebius.py [--model nvidia/nemotron-3-super-120b-a12b]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from llm_backend import NebiusBackend, LLMError  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    be = NebiusBackend(model=args.model)
    print(f"backend : {be.name}")
    print(f"base_url: {be.base_url}")
    print(f"model   : {be.model}")
    print(f"key     : {'set' if be.api_key else 'MISSING'}")
    if not be.configured:
        print("\nNot configured. Set NEBIUS_API_KEY first:")
        print("  1. https://tokenfactory.nebius.com -> create an API key")
        print("  2. claim $25 hackathon credits: code NEBIUS-DEVPOST-GLOBAL26")
        print("  3. export NEBIUS_API_KEY=<key>")
        return 2
    t0 = time.time()
    try:
        out = be.propose(
            'Reply with JSON only: {"ok": true, "who": "<model name>"}')
    except LLMError as e:
        print(f"\nLLM call failed: {e}")
        return 1
    dt = time.time() - t0
    print(f"\nround-trip: {dt:.1f}s")
    print(f"response  : {out[:300]}")
    print("\nOK - Nebius backend is live.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
