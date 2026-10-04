"""LLM backends for the agentic loop.

LLMBackend interface: propose / critique / revise, each taking a prompt
string and returning the model's raw text response.

  HTTPBackend     - OpenAI-compatible chat completions via stdlib urllib
                    (no new dependencies). Config from env:
                      FS_LLM_BASE_URL  e.g. https://api.openai.com/v1
                                       (or the Meta Model API endpoint)
                      FS_LLM_API_KEY
                      FS_LLM_MODEL     e.g. gpt-4o, llama-3.3-70b, ...
  ScriptedBackend - deterministic test double (no network). Canned responses
                    staged to demonstrate the agentic arcs end-to-end:
                    propose -> critic-reject -> revise -> accept, and
                    audit-reject -> revise (PIT fix).
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error


class LLMError(Exception):
    """Anything wrong talking to / parsing the model."""


class LLMNotConfiguredError(LLMError):
    """No usable backend (e.g. HTTPBackend without an API key)."""


class LLMBackend:
    """Interface: three roles, prompt in, raw text out."""

    name = "base"

    @property
    def configured(self) -> bool:
        return False

    def propose(self, prompt: str) -> str:
        raise NotImplementedError

    def critique(self, prompt: str) -> str:
        raise NotImplementedError

    def revise(self, prompt: str) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# HTTP: OpenAI-compatible chat completions, stdlib only.
# ---------------------------------------------------------------------------

class HTTPBackend(LLMBackend):
    name = "http"

    def __init__(self, base_url: str | None = None,
                 api_key: str | None = None,
                 model: str | None = None,
                 timeout: int = 180):
        self.base_url = (base_url or os.environ.get("FS_LLM_BASE_URL") or "").rstrip("/")
        self.api_key = api_key or os.environ.get("FS_LLM_API_KEY") or ""
        self.model = model or os.environ.get("FS_LLM_MODEL") or ""
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)

    def _require_configured(self):
        if not self.configured:
            missing = [k for k, v in {
                "FS_LLM_BASE_URL": self.base_url,
                "FS_LLM_API_KEY": "***" if self.api_key else "",
                "FS_LLM_MODEL": self.model,
            }.items() if not v]
            raise LLMNotConfiguredError(
                "HTTPBackend not configured; missing: " + ", ".join(missing) +
                ". Set FS_LLM_BASE_URL / FS_LLM_API_KEY / FS_LLM_MODEL, or run "
                "with --backend scripted for the offline test double."
            )

    def _chat(self, messages: list[dict], json_mode: bool = True) -> str:
        self._require_configured()
        url = self.base_url + "/chat/completions"
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }
        body = json.dumps({**payload,
                           "response_format": {"type": "json_object"}}).encode()
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            err_body = e.read().decode(errors="replace")[:400]
            # Some servers reject response_format: retry once without it.
            if json_mode and e.code == 400:
                body = json.dumps(payload).encode()
                req = urllib.request.Request(
                    url, data=body,
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {self.api_key}"})
                try:
                    with urllib.request.urlopen(req,
                                                timeout=self.timeout) as resp:
                        data = json.loads(resp.read().decode())
                except urllib.error.HTTPError as e2:
                    raise LLMError(
                        f"LLM HTTP {e2.code}: "
                        f"{e2.read().decode(errors='replace')[:400]}")
            else:
                raise LLMError(f"LLM HTTP {e.code}: {err_body}")
        except urllib.error.URLError as e:
            raise LLMError(f"LLM request failed: {e.reason}")
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"unexpected chat-completions shape: {str(data)[:300]}")

    def propose(self, prompt: str) -> str:
        return self._chat(
            [{"role": "system",
              "content": "You are a feature-discovery scientist. "
                         "Respond with JSON only."},
             {"role": "user", "content": prompt}],
            json_mode=True)

    def critique(self, prompt: str) -> str:
        return self._chat(
            [{"role": "system",
              "content": "You are a skeptical ML reviewer. "
                         "Respond with JSON only."},
             {"role": "user", "content": prompt}],
            json_mode=True)

    def revise(self, prompt: str) -> str:
        return self._chat(
            [{"role": "system",
              "content": "You are a feature-discovery scientist revising a "
                         "rejected feature. Respond with JSON only."},
             {"role": "user", "content": prompt}],
            json_mode=True)


# ---------------------------------------------------------------------------
# Scripted: deterministic test double. The default script stages the two
# demo arcs for a 1-round, 2-hypothesis run:
#   H1 (history entropy, history_scan): critic rejects on cost ->
#        revise -> movie_age_days (row_local) -> accept.
#   H2 (IMDb "quality prior" via imdb_map_avg, claims pit_correct):
#        critic passes with caution -> deterministic audit FAILs (B1) ->
#        revise -> release-year static feature (PIT fixed) -> no lift.
# Call order is fixed; the script documents the staging explicitly.
# ---------------------------------------------------------------------------

_H1_CODE = '''def compute(df, ctx):
    ue = ctx["user_events"]
    out = []
    for uid, ts in zip(df["user_id"].to_numpy(), df["timestamp"].to_numpy()):
        ev = ue.get(uid)
        if ev is None:
            out.append(0.0)
            continue
        r = ev[1][ev[0] <= ts]
        if len(r) == 0:
            out.append(0.0)
            continue
        p = np.bincount(r.astype(int), minlength=6)[1:].astype(float)
        p = p / p.sum()
        p = p[p > 0]
        out.append(float(-(p * np.log(p)).sum()))
    return pd.Series(out, index=df.index)
'''

_H1R_CODE = '''def compute(df, ctx):
    rel = df["movie_id"].map(ctx["movie_release_ts"])
    med = ctx["median_movie_age_days"]
    out = (df["timestamp"] - rel) / 86400.0
    return pd.Series(out.fillna(med).clip(lower=0), index=df.index)
'''

_H2_CODE = '''def compute(df, ctx):
    q = df["movie_id"].map(ctx["imdb_map_avg"])
    return pd.Series(q.fillna(ctx["global_mean"]), index=df.index)
'''

_H2R_CODE = '''def compute(df, ctx):
    rel = df["movie_id"].map(ctx["movie_release_ts"])
    yrs = pd.to_datetime(rel, unit="s", utc=True).dt.year
    return pd.Series(yrs.fillna(1995.0).astype(float), index=df.index)
'''

_DEFAULT_SCRIPT = {
    "propose": [json.dumps({
        "reasoning": (
            "Worst slices: cold/new movies and low-activity users — the model "
            "has no momentum or quality-prior signal where history is thin. "
            "Two ideas: (1) rater-behavior entropy from the user's own "
            "history — discriminating raters shift P(liked); it needs a full "
            "history scan at serve time, which may be too expensive, but the "
            "signal is worth testing. (2) an IMDb quality prior for cold "
            "movies; I will use the IMDb average mapped per movie and treat "
            "it as an as-of quality signal."
        ),
        "hypotheses": [
            {"name": "user_history_entropy_llm",
             "text": "Rater behavior matters: the entropy of a user's rating "
                     "history captures how discriminating they are, which "
                     "shifts P(liked).",
             "sources": ["ml1m_ratings"],
             "point_in_time": "timestamp",
             "temporal_scope": "pit_correct",
             "tower": "user",
             "serving": "history_scan",
             "serving_notes": "scans the user's full train-period history per request",
             "code": _H1_CODE},
            {"name": "imdb_quality_prior_llm",
             "text": "Cold movies have thin rating history; IMDb's average "
                     "rating encodes true quality better than our sparse "
                     "2001 data and should transfer as a quality prior.",
             "sources": ["imdb_enrichment"],
             "point_in_time": "timestamp",
             "temporal_scope": "pit_correct",
             "tower": "movie",
             "serving": "lookup",
             "serving_notes": "O(1) map from precomputed IMDb table",
             "code": _H2_CODE},
        ],
    })],
    # critique calls in ACTUAL pipeline order:
    #   review(H1) -> reject(cost); re-review(H1r) -> pass;
    #   review(H2) -> pass with caution (audit is the decider).
    "critique": [
        json.dumps({
            "verdict": "reject",
            "critique": (
                "Serving cost: history_scan = 25 latency units vs the 8.0 "
                "per-feature budget. The entropy signal may be real, but this "
                "feature can never ship; the cost gate will kill it before "
                "ablation. Propose a row_local or lookup alternative that "
                "captures a related signal."
            ),
            "suggested_fix": (
                "Replace with a time-invariant row-local feature, e.g. days "
                "since movie release from ctx['movie_release_ts'] "
                "(novelty decay)."
            ),
        }),
        json.dumps({
            "verdict": "pass",
            "critique": (
                "Row-local, time-invariant (release year only), 0.5 latency "
                "units. PIT-clean by construction. Weak signal expected, but "
                "cheap enough to test."
            ),
            "suggested_fix": "",
        }),
        json.dumps({
            "verdict": "pass",
            "critique": (
                "Code reads ctx['imdb_map_avg'], which the catalog flags as "
                "an ALL-TIME aggregate (votes through Oct 2026). The "
                "hypothesis claims as-of handling, but I see no as-of logic "
                "in the code — only a raw map. Borderline: I cannot prove "
                "the temporal question from the code alone, so I defer to "
                "the deterministic audit."
            ),
            "suggested_fix": "",
        }),
    ],
    # revise calls in order: H1 after critic-reject -> H1r;
    # H2 after audit-reject -> H2r (PIT fix: static release year).
    "revise": [
        json.dumps({
            "name": "movie_age_days_llm",
            "text": "Novelty decay: a movie's age at prediction time shifts "
                    "its like-rate. Time-invariant metadata, row-local, "
                    "cheap to serve.",
            "sources": ["imdb_enrichment", "ml1m_movies"],
            "point_in_time": "timestamp",
            "temporal_scope": "static",
            "tower": "movie",
            "serving": "row_local",
            "serving_notes": "timestamp minus release_ts; pure row function",
            "code": _H1R_CODE,
        }),
        json.dumps({
            "name": "movie_release_year_llm",
            "text": "Revised after the audit caught the all-time IMDb "
                    "aggregate: use only the time-invariant release year as "
                    "a static movie prior. PIT-clean by construction.",
            "sources": ["imdb_enrichment"],
            "point_in_time": "timestamp",
            "temporal_scope": "static",
            "tower": "movie",
            "serving": "lookup",
            "serving_notes": "O(1) map from release_ts table",
            "code": _H2R_CODE,
        }),
    ],
}


class ScriptedBackend(LLMBackend):
    """Deterministic canned backend for offline verification of the full
    agentic path (propose -> critic -> revise -> audit -> cost -> ablate).
    No network. Call order is fixed and documented in _DEFAULT_SCRIPT."""

    name = "scripted"

    def __init__(self, script: dict | None = None):
        self.script = script or _DEFAULT_SCRIPT
        self._counters = {"propose": 0, "critique": 0, "revise": 0}

    @property
    def configured(self) -> bool:
        return True

    def _next(self, role: str) -> str:
        items = self.script.get(role, [])
        if not items:
            raise LLMError(f"scripted backend: no canned '{role}' responses")
        i = min(self._counters[role], len(items) - 1)
        self._counters[role] += 1
        return items[i]

    def propose(self, prompt: str) -> str:
        return self._next("propose")

    def critique(self, prompt: str) -> str:
        return self._next("critique")

    def revise(self, prompt: str) -> str:
        return self._next("revise")
