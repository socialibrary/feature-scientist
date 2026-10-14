"""Offline Wikidata crawl for movie enrichment.

For each tconst in data/imdb/movie_enrichment.csv, queries the Wikidata
Query Service (IMDb ID = P345) for directors, principal cast, awards
received (P166), box office (P2142), budget (P2130), publication date
(P577), and original language (P364).

Politeness: chunked requests (default 100 tconsts), a delay between
requests (default 65s -- WDQS aggressively rate-limits during outages),
retries with exponential backoff on 429/5xx, a descriptive User-Agent,
raw JSON cached per chunk, and resume-friendly (skips tconsts already in
the output CSV).

Entity references are kept as Wikidata QIDs (not labels) -- for feature
engineering the identity is what matters. A second phase resolves labels
for directors/awards/languages into entity_labels.json for human-readable
demo output.

Usage:
    python3 src/crawl_wikidata.py [--chunk 100] [--delay 65] [--limit N]

Output:
    data/wikidata/movie_wikidata.csv  (tconst + enrichment columns)
    data/wikidata/raw/chunk_*.json     (cached raw SPARQL responses)
    data/wikidata/entity_labels.json   (QID -> English label, directors/
                                        awards/languages only)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import urllib.error

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENRICHMENT_CSV = os.path.join(BASE, "data", "imdb", "movie_enrichment.csv")
OUT_DIR = os.path.join(BASE, "data", "wikidata")
RAW_DIR = os.path.join(OUT_DIR, "raw")
OUT_CSV = os.path.join(OUT_DIR, "movie_wikidata.csv")
LABELS_JSON = os.path.join(OUT_DIR, "entity_labels.json")

SPARQL_URL = "https://query.wikidata.org/sparql"
USER_AGENT = ("FeatureScientist/0.1 (offline research crawl for hackathon "
              "demo; polite rate: 1 req/min)")

COLUMNS = ["tconst", "wikidata_qid", "directors", "cast", "awards",
           "box_office_amount", "budget_amount", "release_date",
           "original_language"]


def sparql(query: str, timeout: int = 120) -> dict:
    params = urllib.parse.urlencode({"query": query, "format": "json"})
    req = urllib.request.Request(
        SPARQL_URL + "?" + params,
        headers={"User-Agent": USER_AGENT,
                 "Accept": "application/sparql-results+json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def sparql_retry(query: str, max_retries: int = 12) -> dict:
    import socket
    wait = 75.0
    for attempt in range(max_retries):
        try:
            return sparql(query)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                print(f"    HTTP {e.code}, backing off {wait:.0f}s "
                      f"(attempt {attempt + 1}/{max_retries})", flush=True)
                time.sleep(wait)
                wait = min(wait * 1.5, 600.0)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, socket.timeout) as e:
            # raw network timeouts: same backoff treatment, then retry
            if attempt < max_retries - 1:
                print(f"    network timeout ({type(e).__name__}), backing off "
                      f"{wait:.0f}s (attempt {attempt + 1}/{max_retries})",
                      flush=True)
                time.sleep(wait)
                wait = min(wait * 1.5, 600.0)
                continue
            raise
    raise RuntimeError("unreachable")


CHUNK_QUERY = """
SELECT ?imdb ?qid
  (GROUP_CONCAT(DISTINCT ?dirqid; separator="|") AS ?directors)
  (GROUP_CONCAT(DISTINCT ?castqid; separator="|") AS ?cast)
  (GROUP_CONCAT(DISTINCT ?awardqid; separator="|") AS ?awards)
  (SAMPLE(?boxoffice) AS ?boxoffice)
  (SAMPLE(?budget) AS ?budget)
  (SAMPLE(?pubdate) AS ?pubdate)
  (SAMPLE(?langqid) AS ?lang)
WHERE {{
  VALUES ?imdb {{ {values} }}
  ?film wdt:P345 ?imdb.
  BIND(REPLACE(STR(?film), "http://www.wikidata.org/entity/", "") AS ?qid)
  OPTIONAL {{ ?film wdt:P57 ?dir.
              BIND(REPLACE(STR(?dir), "http://www.wikidata.org/entity/", "") AS ?dirqid) }}
  OPTIONAL {{ ?film wdt:P161 ?c.
              BIND(REPLACE(STR(?c), "http://www.wikidata.org/entity/", "") AS ?castqid) }}
  OPTIONAL {{ ?film wdt:P166 ?a.
              BIND(REPLACE(STR(?a), "http://www.wikidata.org/entity/", "") AS ?awardqid) }}
  OPTIONAL {{ ?film wdt:P2142 ?boxoffice. }}
  OPTIONAL {{ ?film wdt:P2130 ?budget. }}
  OPTIONAL {{ ?film wdt:P577 ?pubdate. }}
  OPTIONAL {{ ?film wdt:P364 ?l.
              BIND(REPLACE(STR(?l), "http://www.wikidata.org/entity/", "") AS ?langqid) }}
}}
GROUP BY ?imdb ?qid
"""

LABEL_QUERY = """
SELECT ?q ?qLabel WHERE {{
  VALUES ?q {{ {values} }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
"""


def load_tconsts() -> list[str]:
    with open(ENRICHMENT_CSV, newline="", encoding="utf-8") as f:
        return [r["tconst"] for r in csv.DictReader(f) if r.get("tconst")]


def load_done() -> set[str]:
    if not os.path.exists(OUT_CSV):
        return set()
    with open(OUT_CSV, newline="", encoding="utf-8") as f:
        return {r["tconst"] for r in csv.DictReader(f) if r.get("tconst")}


def parse_chunk(resp: dict, tconsts: list[str]) -> list[dict]:
    """Parse SPARQL bindings; tconsts with no Wikidata entity get an
    (empty) row so they are recorded as done and never re-queried."""
    rows = []
    seen = set()
    for b in resp["results"]["bindings"]:
        def g(k):
            return b.get(k, {}).get("value", "")
        t = g("imdb")
        seen.add(t)
        rows.append({
            "tconst": t,
            "wikidata_qid": g("qid"),
            "directors": g("directors"),
            "cast": g("cast"),
            "awards": g("awards"),
            "box_office_amount": g("boxoffice"),
            "budget_amount": g("budget"),
            "release_date": g("pubdate").split("T")[0] if g("pubdate") else "",
            "original_language": g("lang"),
        })
    empty = {c: "" for c in COLUMNS}
    for t in tconsts:
        if t not in seen:
            rows.append({**empty, "tconst": t})
    return rows


def append_rows(rows: list[dict]) -> None:
    # dedupe: never write a tconst that's already in the output
    seen = load_done()
    rows = [r for r in rows if r["tconst"] not in seen]
    if not rows:
        return
    new = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerows(rows)


def resolve_labels(qids: list[str], delay: float) -> dict[str, str]:
    """Batch-resolve QIDs to English labels; cached in entity_labels.json."""
    labels: dict[str, str] = {}
    if os.path.exists(LABELS_JSON):
        with open(LABELS_JSON, encoding="utf-8") as f:
            labels = json.load(f)
    todo = [q for q in dict.fromkeys(qids) if q and q not in labels]
    print(f"resolving labels for {len(todo)} entities "
          f"({len(labels)} cached)", flush=True)
    for i in range(0, len(todo), 400):
        chunk = todo[i:i + 400]
        vals = " ".join(f"wd:{q}" for q in chunk)
        resp = sparql_retry(LABEL_QUERY.format(values=vals))
        for b in resp["results"]["bindings"]:
            q = b["q"]["value"].rsplit("/", 1)[-1]
            labels[q] = b.get("qLabel", {}).get("value", q)
        with open(LABELS_JSON, "w", encoding="utf-8") as f:
            json.dump(labels, f, indent=1)
        print(f"  labeled {min(i + 400, len(todo))}/{len(todo)}", flush=True)
        time.sleep(delay)
    return labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--delay", type=float, default=75.0)
    ap.add_argument("--limit", type=int, default=0,
                    help="only crawl first N tconsts (0 = all)")
    ap.add_argument("--skip-labels", action="store_true")
    args = ap.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)
    tconsts = load_tconsts()
    if args.limit:
        tconsts = tconsts[:args.limit]
    done = load_done()
    todo = [t for t in tconsts if t not in done]
    print(f"{len(tconsts)} tconsts total, {len(done)} cached, "
          f"{len(todo)} to crawl", flush=True)

    n_chunks = (len(todo) + args.chunk - 1) // args.chunk
    import hashlib as _hl
    for ci in range(n_chunks):
        chunk = todo[ci * args.chunk:(ci + 1) * args.chunk]
        # content-addressed cache: stable across --chunk/--limit changes,
        # so a smoke-test cache can never collide with a full-run chunk.
        chash = _hl.sha1(",".join(chunk).encode()).hexdigest()[:10]
        raw_path = os.path.join(RAW_DIR, f"chunk_{chash}.json")
        if os.path.exists(raw_path):
            with open(raw_path, encoding="utf-8") as f:
                resp = json.load(f)
            print(f"[{ci + 1}/{n_chunks}] cache hit ({len(chunk)} tconsts)",
                  flush=True)
        else:
            vals = " ".join(f'"{t}"' for t in chunk)
            resp = sparql_retry(CHUNK_QUERY.format(values=vals))
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(resp, f)
            print(f"[{ci + 1}/{n_chunks}] crawled {len(chunk)} tconsts -> "
                  f"{len(resp['results']['bindings'])} films", flush=True)
            time.sleep(args.delay)
        rows = parse_chunk(resp, chunk)
        append_rows(rows)

    if not args.skip_labels:
        # labels only for directors/awards/languages (demo narrative needs
        # names here; cast stays QIDs)
        qids: list[str] = []
        with open(OUT_CSV, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                for col in ("directors", "awards", "original_language"):
                    qids.extend(x for x in r[col].split("|") if x)
        resolve_labels(qids, args.delay)

    # coverage report
    n, cov = 0, {c: 0 for c in COLUMNS if c not in ("tconst",)}
    with open(OUT_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            n += 1
            for c in cov:
                if r[c]:
                    cov[c] += 1
    print(f"\ncoverage over {n} movies:")
    for c, k in cov.items():
        print(f"  {c:18s} {k:5d}  ({100.0 * k / max(n, 1):5.1f}%)")


if __name__ == "__main__":
    main()
