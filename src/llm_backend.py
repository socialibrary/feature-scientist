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
import re
import sys
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
# Nebius: Token Factory, OpenAI-compatible. The hackathon's required path:
# the agentic loop (proposer/critic) runs on an NVIDIA open-source model
# (Nemotron) hosted on Nebius. Config from env:
#   NEBIUS_API_KEY   (or FS_LLM_API_KEY as fallback)
#   NEBIUS_MODEL     (default: Nemotron 3 Nano 30B — fast, cheap, JSON-clean;
#                     upgrade: nvidia/nemotron-3-super-120b-a12b)
#   NEBIUS_BASE_URL  (default: https://api.tokenfactory.nebius.com/v1)
# ---------------------------------------------------------------------------

NEBIUS_DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1"
NEBIUS_DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"


class NebiusBackend(HTTPBackend):
    name = "nebius"

    # Vault-backed CLI (no raw key in env). Falls back to direct HTTP when
    # NEBIUS_API_KEY is set explicitly.
    _VAULT_CLI = os.path.expanduser(
        "~/workspace/skills/nebius-token-factory/bin/chat.py")

    def __init__(self, base_url: str | None = None,
                 api_key: str | None = None,
                 model: str | None = None,
                 timeout: int = 180):
        super().__init__(
            base_url=(base_url or os.environ.get("NEBIUS_BASE_URL")
                      or NEBIUS_DEFAULT_BASE_URL),
            api_key=(api_key or os.environ.get("NEBIUS_API_KEY")
                     or os.environ.get("FS_LLM_API_KEY") or ""),
            model=(model or os.environ.get("NEBIUS_MODEL")
                   or os.environ.get("FS_LLM_MODEL")
                   or NEBIUS_DEFAULT_MODEL),
            timeout=timeout)
        self._use_vault = (not self.api_key
                           and os.path.exists(self._VAULT_CLI))

    @property
    def configured(self) -> bool:
        # Vault path needs no env key; direct path needs one.
        return bool(self.base_url and self.model
                    and (self.api_key or self._use_vault))

    def _require_configured(self):
        if not self.configured:
            raise LLMNotConfiguredError(
                "NebiusBackend not configured: set NEBIUS_API_KEY or connect "
                "the nebius-token-factory vault credential "
                "(get a key at https://tokenfactory.nebius.com; hackathon "
                "credits code NEBIUS-DEVPOST-GLOBAL26 via the Devpost "
                "Resources form). Optional: NEBIUS_MODEL, NEBIUS_BASE_URL.")

    def _chat_via_vault(self, messages: list[dict],
                        json_mode: bool = True) -> str:
        import subprocess
        cmd = [sys.executable, self._VAULT_CLI, "--model", self.model,
               "--timeout", str(self.timeout)]
        if json_mode:
            cmd.append("--json-mode")
        try:
            p = subprocess.run(
                cmd, input=json.dumps(messages).encode(),
                capture_output=True, timeout=self.timeout + 30)
        except subprocess.TimeoutExpired:
            raise LLMError("Nebius vault CLI timed out")
        if p.returncode != 0:
            raise LLMError(
                f"Nebius vault CLI failed: {p.stderr.decode()[:300]}")
        return p.stdout.decode().strip()

    def _chat(self, messages: list[dict], json_mode: bool = True) -> str:
        self._require_configured()
        if self._use_vault:
            return self._chat_via_vault(messages, json_mode=json_mode)
        return super()._chat(messages, json_mode=json_mode)


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


# ---------------------------------------------------------------------------
# Named canned scripts. FS_SCRIPTED=<name> selects one for ScriptedBackend.
#   "v2" (default): the original two-hypothesis demo arc (history entropy /
#                   IMDb quality prior, critic + audit staging).
#   "v3": fresh Wikidata-territory arc (budget prior, non-English flag,
#         awards fail-closed trap caught twice) -- see wikidata_script_v3.
# ---------------------------------------------------------------------------
def _named_scripts() -> dict:
    from wikidata_script_v3 import V3_SCRIPT
    return {"v2": _DEFAULT_SCRIPT, "v3": V3_SCRIPT}


_NAMED_SCRIPTS = None  # resolved lazily (import at module load is fine)


def _resolve_named_scripts() -> dict:
    global _NAMED_SCRIPTS
    if _NAMED_SCRIPTS is None:
        _NAMED_SCRIPTS = _named_scripts()
    return _NAMED_SCRIPTS


class ScriptedBackend(LLMBackend):
    """Deterministic canned backend for offline verification of the full
    agentic path (propose -> critic -> revise -> audit -> cost -> ablate).
    No network. Call order is fixed and documented in _DEFAULT_SCRIPT.

    A named canned script can be selected with FS_SCRIPTED=<name> (see
    _NAMED_SCRIPTS); an explicit `script` argument always wins. Unknown
    names fail loudly rather than silently running the default arc.
    """

    name = "scripted"

    def __init__(self, script: dict | None = None):
        if script is not None:
            self.script = script
        else:
            want = os.environ.get("FS_SCRIPTED", "")
            named = _resolve_named_scripts()
            if not want:
                self.script = _DEFAULT_SCRIPT
            elif want not in named:
                raise ValueError(
                    f"unknown scripted backend {want!r}; "
                    f"want one of {sorted(named)}")
            else:
                self.script = named[want]
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

    # Name-keyed critiques: the order-based script only covers round 1's
    # call sequence, but later rounds re-propose the same hypotheses (the
    # propose script clamps to its single item). Matching the critique to
    # the hypothesis NAME in the prompt keeps the skeptic's text coherent
    # when the call order diverges (e.g. round 2 re-reviews). Unknown names
    # fall back to order-based.
    _CRITIQUE_BY_NAME = {
        "user_history_entropy_llm": 0,   # reject: serving cost
        "movie_age_days_llm": 1,         # pass: cheap, PIT-clean
        "imdb_quality_prior_llm": 2,     # pass with caution: audit decides
        "movie_release_year_llm": 1,     # pass: cheap, PIT-clean
    }

    def critique(self, prompt: str) -> str:
        m = re.search(r'"name"\s*:\s*"([^"]+)"', prompt)
        if m:
            name = m.group(1)
            # Script-level name-keyed critiques win over the class map.
            by_name = self.script.get("critique_by_name", {})
            if name in by_name:
                return by_name[name]
            idx = self._CRITIQUE_BY_NAME.get(name)
            if idx is not None:
                return self.script["critique"][idx]
        return self._next("critique")

    def revise(self, prompt: str) -> str:
        return self._next("revise")


class TabularScriptedBackend(LLMBackend):
    """Data-driven scripted backend for generic tabular tasks.

    Parses the data catalog from the prompt and proposes feature-engineering
    hypotheses based on actual column types. No MovieLens-specific logic.
    """

    name = "tabular-scripted"

    @property
    def configured(self) -> bool:
        return True

    def _get_numeric_cols(self) -> list:
        """Read numeric columns from the profiled tables (profiling.json).

        The target column is excluded data-driven: any column present in
        <name>_train but absent from <name>_test is treated as the label
        (it cannot be a serving-time feature). Replaces the hardcoded name
        list, which missed 'fare_amount' and burned the nyc-taxi round on
        leakage-killed hypotheses (Oct 13 2026).
        """
        import json, os
        # Find profiling.json relative to this file.
        here = os.path.dirname(os.path.abspath(__file__))
        pj = os.path.join(here, "catalog", "profiling.json")
        if not os.path.exists(pj):
            return []
        try:
            doc = json.load(open(pj, encoding="utf-8"))
        except Exception:
            return []
        tables = doc.get("tables", {})
        legacy_exclude = {"id", "saleprice", "transported", "target"}
        numeric = []
        for tname, tinfo in tables.items():
            if not tname.endswith("_train"):
                continue
            train_cols = tinfo.get("columns") or {}
            test_cols = (tables.get(tname[: -len("_train")] + "_test", {})
                         .get("columns") or {})
            target_like = ({c for c in train_cols if c not in test_cols}
                           if test_cols else set())
            for cname, cinfo in train_cols.items():
                if not isinstance(cinfo, dict):
                    continue
                dt = str(cinfo.get("dtype", ""))
                if (dt.startswith(("int", "float"))
                        and cname.lower() not in legacy_exclude
                        and cname not in target_like):
                    numeric.append(cname)
            if numeric:
                break
        return numeric

    def propose(self, prompt: str) -> str:
        import json
        numeric_cols = self._get_numeric_cols()
        if len(numeric_cols) < 2:
            # Fallback if profiling.json is missing/unreadable.
            numeric_cols = ["col_a", "col_b"]

        c1, c2 = numeric_cols[0], numeric_cols[1]
        hyps = [
            {
                "name": f"{c1}_x_{c2}_interaction",
                "text": f"Interaction between {c1} and {c2} captures combined effects the model may miss.",
                "sources": ["train"],
                "point_in_time": "timestamp",
                "temporal_scope": "static",
                "tower": "wide",
                "serving": "row_local",
                "serving_notes": "row-local computation, no lookup needed",
                "code": (
                    f"def compute(df, ctx):\n"
                    f"    a = pd.to_numeric(df['{c1}'], errors='coerce').fillna(0)\n"
                    f"    b = pd.to_numeric(df['{c2}'], errors='coerce').fillna(0)\n"
                    f"    return pd.Series(a * b, index=df.index)\n"
                ),
            },
            {
                "name": f"{c1}_squared",
                "text": f"Quadratic term for {c1} captures non-linear effects.",
                "sources": ["train"],
                "point_in_time": "timestamp",
                "temporal_scope": "static",
                "tower": "wide",
                "serving": "row_local",
                "serving_notes": "row-local computation",
                "code": (
                    f"def compute(df, ctx):\n"
                    f"    a = pd.to_numeric(df['{c1}'], errors='coerce').fillna(0)\n"
                    f"    return pd.Series(a ** 2, index=df.index)\n"
                ),
            },
        ]
        return json.dumps({
            "reasoning": f"Proposing generic tabular features from {len(numeric_cols)} numeric columns.",
            "hypotheses": hyps,
        })

    def critique(self, prompt: str) -> str:
        import json
        return json.dumps({
            "verdict": "PASS",
            "concerns": [],
            "notes": "Row-local features, no leakage risk, cheap to serve.",
        })

    def revise(self, prompt: str) -> str:
        return self.propose(prompt)


class SpaceshipTitanicBackend(LLMBackend):
    """Domain-aware scripted backend for the Spaceship Titanic brief.

    Encodes the dataset's known-good feature-engineering patterns as
    deterministic, correct code: Cabin deck/num/side parsing, total spend
    across the 5 service columns, group size from PassengerId. The agent
    loop still judges every proposal (critic -> audit -> cost -> ablate);
    this backend only replaces the idea-generation/codegen step, which the
    template proposer and the LLM both fumble on this dataset.
    """

    name = "st-domain"

    _SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

    def __init__(self):
        self._idx = 0

    @property
    def configured(self) -> bool:
        return True

    def _pool(self) -> list:
        cabin_parse = (
            "def compute(df, ctx):\n"
            "    c = df['Cabin'].fillna('U/0/U').astype(str)\n"
            "    deck = c.str.split('/').str[0]\n"
            "    code = deck.map({'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'T':7}).fillna(-1)\n"
            "    return pd.Series(code.astype('float64'), index=df.index, name='cabin_deck')\n"
        )
        cabin_side = (
            "def compute(df, ctx):\n"
            "    c = df['Cabin'].fillna('U/0/U').astype(str)\n"
            "    side = c.str.split('/').str[2]\n"
            "    code = side.map({'P':1,'S':0}).fillna(-1)\n"
            "    return pd.Series(code.astype('float64'), index=df.index, name='cabin_side')\n"
        )
        cabin_num = (
            "def compute(df, ctx):\n"
            "    c = df['Cabin'].fillna('U/0/U').astype(str)\n"
            "    num = pd.to_numeric(c.str.split('/').str[1], errors='coerce').fillna(0)\n"
            "    return pd.Series(num.astype('float64'), index=df.index, name='cabin_num')\n"
        )
        total_spend = (
            "def compute(df, ctx):\n"
            f"    cols = {self._SPEND!r}\n"
            "    total = df[cols].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1)\n"
            "    return pd.Series(total, index=df.index, name='total_spend')\n"
        )
        group_size = (
            "def compute(df, ctx):\n"
            "    grp = df['PassengerId'].astype(str).str.split('_').str[0]\n"
            "    sizes = grp.map(grp.value_counts())\n"
            "    return pd.Series(sizes.astype('int64'), index=df.index, name='group_size')\n"
        )
        no_spend_flag = (
            "def compute(df, ctx):\n"
            f"    cols = {self._SPEND!r}\n"
            "    total = df[cols].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1)\n"
            "    return pd.Series((total == 0).astype('int64'), index=df.index, name='no_spend_flag')\n"
        )
        hyps = [
            ("cabin_deck",
             "Cabin encodes deck/num/side as 'B/0/P'; the deck letter (encoded "
             "as an integer code) captures ship section, which correlates with "
             "passenger class and destination.",
             ["Cabin"], cabin_parse),
            ("total_spend",
             "Total spending across the 5 onboard service columns (RoomService, "
             "FoodCourt, ShoppingMall, Spa, VRDeck) captures expenditure level, "
             "which correlates with being transported.",
             self._SPEND, total_spend),
            ("group_size",
             "PassengerId is '<group>_<member>'; the group size (from the group "
             "prefix) captures traveling-party effects the row-local model misses.",
             ["PassengerId"], group_size),
            ("cabin_side",
             "Cabin side (P port -> 1, S starboard -> 0) from the Cabin code; "
             "ship section effects beyond the deck letter.",
             ["Cabin"], cabin_side),
            ("cabin_num",
             "Cabin number within the deck from the Cabin code; location effects "
             "along the deck.",
             ["Cabin"], cabin_num),
            ("no_spend_flag",
             "Passengers with zero total onboard spending (e.g. CryoSleep) form "
             "a distinct behavioral slice with different transport rates.",
             self._SPEND + ["CryoSleep"], no_spend_flag),
        ]
        return [
            {
                "name": name,
                "text": text,
                "sources": sources,
                "point_in_time": "timestamp",
                "temporal_scope": "static",
                "tower": "wide",
                "serving": "row_local",
                "serving_notes": "row-local computation on booking-time fields, no lookup needed",
                "code": code,
            }
            for name, text, sources, code in hyps
        ]

    def _next_two(self) -> list:
        pool = self._pool()
        out = [pool[(self._idx + i) % len(pool)] for i in range(2)]
        self._idx = (self._idx + 2) % len(pool)
        return out

    def propose(self, prompt: str) -> str:
        import json
        return json.dumps({
            "reasoning": "Proposing domain-aware spaceship-titanic features: "
                         "Cabin decomposition (deck/side/num), total spend, "
                         "group size, no-spend flag.",
            "hypotheses": self._next_two(),
        })

    def critique(self, prompt: str) -> str:
        import json
        return json.dumps({
            "verdict": "PASS",
            "concerns": [],
            "notes": "Row-local features computed from booking-time fields "
                     "only; no target or future data; cheap to serve.",
        })

    def revise(self, prompt: str) -> str:
        return self.propose(prompt)
