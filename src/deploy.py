#!/usr/bin/env python3
"""Deployment: upload trained models to Hugging Face Hub.

Usage:
    python3 src/deploy.py --model results/champion_model.pkl --name my-model

For now supports Hugging Face. Nebius/AWS coming.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def deploy_to_huggingface(model_path: str, repo_name: str,
                          token: str | None = None) -> str:
    """Upload a model to Hugging Face Hub. Returns the model URL.

    model_path: path to the pickled model (or directory with model files)
    repo_name: e.g. "username/model-name" or just "model-name"
    token: HF token (or set HF_TOKEN env var)

    NOTE: some sandboxed runtimes block authenticated requests to
    huggingface.co at the egress proxy (sentinel policy_denied). Run
    `python3 src/deploy.py --dry-run` first to check; if the policy
    blocks auth, run the upload from an unrestricted machine.
    """
    from huggingface_hub import HfApi, create_repo

    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "Hugging Face token required. Set HF_TOKEN env var or pass token=."
        )

    api = HfApi(token=token)

    # Create repo if it doesn't exist
    try:
        create_repo(repo_name, token=token, exist_ok=True)
    except Exception as e:
        _raise_policy_aware(e)
        print(f"  repo creation: {e}")

    # Upload the model file
    try:
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=os.path.basename(model_path),
            repo_id=repo_name,
            token=token,
        )
    except Exception as e:
        _raise_policy_aware(e)
        raise

    url = f"https://huggingface.co/{repo_name}"
    print(f"  deployed: {url}")
    return url


def _raise_policy_aware(exc: Exception) -> None:
    """Re-raise with a clear message when the egress policy blocks auth."""
    msg = str(exc)
    if "policy_denied" in msg or "sentinel" in msg.lower():
        raise RuntimeError(
            "Hugging Face upload blocked: this machine's egress proxy "
            "denies authenticated requests to huggingface.co "
            "(sentinel policy_denied on the Authorization header). "
            "The code is correct — run it from an unrestricted machine:\n"
            "  pip install huggingface_hub\n"
            "  HF_TOKEN=<your token> python3 src/deploy.py "
            "--model <champion.pkl> --name <repo>\n"
            "Or request an egress exception for huggingface.co auth."
        ) from exc


def check_hf_reachability() -> dict:
    """Diagnose HF connectivity from this machine. No credentials sent.

    Probes, with stdlib only:
      1. public read  GET /api/models            (expect 200)
      2. auth passthrough GET /api/whoami, no
         Authorization header                    (expect 401 from HF itself)
      3. auth probe  GET /api/whoami with a DUMMY
         bearer token                            (a fast policy_denied body
         means the egress proxy blocks the Authorization header)
    Returns a dict with the raw outcomes and a verdict.
    """
    import urllib.error
    import urllib.request

    def _get(path: str, token: str | None = None):
        req = urllib.request.Request(f"https://huggingface.co{path}")
        if token is not None:
            req.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.status, r.read(400).decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            return e.code, e.read(400).decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001 -- network may fail any way
            return None, f"{type(e).__name__}: {e}"

    public_status, _ = _get("/api/models?limit=1")
    noauth_status, _ = _get("/api/whoami")
    auth_status, auth_body = _get("/api/whoami", token="hf_dummy_reachability")
    policy_block = auth_body is not None and "policy_denied" in auth_body
    verdict = (
        "blocked: egress proxy denies authenticated HF requests "
        "(sentinel policy). Public reads work; uploads need an "
        "unrestricted machine or an egress exception."
        if policy_block else
        "reachable: authenticated requests are not proxy-blocked here."
        if auth_status == 401 else
        f"inconclusive (public={public_status}, noauth={noauth_status}, "
        f"auth={auth_status})"
    )
    return {
        "public_read_status": public_status,
        "whoami_no_auth_status": noauth_status,
        "whoami_dummy_auth_status": auth_status,
        "auth_policy_blocked": policy_block,
        "verdict": verdict,
    }


def save_champion_model(model, feature_names: list[str], path: str,
                        extra: dict | None = None):
    """Save a trained model with metadata for deployment.

    extra: optional dict merged into the payload (framework, target,
    metric, accepted_feature_code, ...). Everything needed to rebuild
    the inference feature matrix travels with the model.
    """
    import pickle
    payload = {
        "model": model,
        "feature_names": feature_names,
        "saved_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
    }
    payload.update(extra or {})
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"  saved champion to {path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="path to model file")
    ap.add_argument("--name", default=None, help="HF repo name")
    ap.add_argument("--token", default=None, help="HF token (or HF_TOKEN env)")
    ap.add_argument("--dry-run", action="store_true",
                    help="only diagnose HF connectivity; do not upload")
    args = ap.parse_args()

    if args.dry_run:
        diag = check_hf_reachability()
        for k, v in diag.items():
            print(f"  {k}: {v}")
        sys.exit(0)

    if not args.model or not args.name:
        ap.error("--model and --name are required (or use --dry-run)")

    url = deploy_to_huggingface(args.model, args.name, args.token)
    print(f"\nModel live at: {url}")
