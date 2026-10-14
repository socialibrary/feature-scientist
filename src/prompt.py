#!/usr/bin/env python3
"""Prompt-based entry point for Feature Scientist.

Usage:
    python3 src/prompt.py "Find features for spaceship-titanic"
    python3 src/prompt.py "Find features that predict Transported"

No flags, no YAML paths, no backend names. Just a prompt.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Map natural language to briefs
PROMPT_ROUTES = {
    "spaceship": ("briefs/spaceship_titanic.yaml", "st-domain"),
    "titanic": ("briefs/spaceship_titanic.yaml", "st-domain"),
    "transported": ("briefs/spaceship_titanic.yaml", "st-domain"),
    "house": ("briefs/house_price.yaml", "tabular-scripted"),
    "price": ("briefs/house_price.yaml", "tabular-scripted"),
    "movielens": ("briefs/movielens.yaml", "scripted"),
    "movie": ("briefs/movielens.yaml", "scripted"),
}


def route(prompt: str) -> tuple[str, str, str | None]:
    """Map a natural language prompt to (brief, backend, deploy_target)."""
    pl = prompt.lower()
    # Detect deployment intent
    deploy_target = None
    if "hugging face" in pl or "huggingface" in pl or " hf " in pl:
        deploy_target = "huggingface"
    elif "nebius" in pl:
        deploy_target = "nebius"
    
    for key, (brief, backend) in PROMPT_ROUTES.items():
        if key in pl:
            return brief, backend, deploy_target
    # Default: try to infer from prompt
    return "briefs/spaceship_titanic.yaml", "st-domain", deploy_target


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 src/prompt.py \"<your request>\"")
        print("Example: python3 src/prompt.py \"Find features for spaceship-titanic\"")
        sys.exit(1)

    prompt = " ".join(sys.argv[1:])
    brief, backend, deploy_target = route(prompt)

    print(f"Prompt: {prompt}")
    print(f"→ brief: {brief}, backend: {backend}")
    if deploy_target:
        print(f"→ deploy: {deploy_target}")
    print()

    # Delegate to brief.py
    argv = ["brief.py", brief, "--backend", backend, "--max-rounds", "2"]
    if deploy_target:
        argv += ["--deploy", deploy_target]
    sys.argv = argv
    from brief import main as brief_main
    brief_main()


if __name__ == "__main__":
    main()
