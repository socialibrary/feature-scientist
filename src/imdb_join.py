"""Join MovieLens-1M movies to IMDb title.basics + title.ratings.

Outputs data/imdb/movie_enrichment.csv:
  movieId, tconst, primaryTitle, startYear, directors, runtimeMinutes,
  imdb_averageRating, imdb_numVotes

Join key: normalized title + release year.
"""
import gzip
import re
import unicodedata
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
IMDB = BASE / "data" / "imdb"
ML_MOVIES = BASE / "data" / "ml-1m" / "movies.dat"
OUT = IMDB / "movie_enrichment.csv"


def normalize_title(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    # Move trailing article to front: "Grumpier Old Men, The" -> "the grumpier old men"
    m = re.match(r"^(.*),\s*(the|a|an)$", s.strip())
    if m:
        s = f"{m.group(2)} {m.group(1)}"
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20",
}


def candidate_titles(title: str):
    """Yield normalized title candidates, best first.

    Handles MovieLens quirks: parenthetical alt-titles ("Seven (Se7en)"),
    articles ("Postino, Il (The Postman)"), number words ("Twelve Monkeys"
    vs IMDb's "12 Monkeys").
    """
    cands = []
    base = normalize_title(title)
    cands.append(base)
    # Strip parenthetical groups: "seven se7en" -> "seven"
    no_paren = normalize_title(re.sub(r"\([^)]*\)", " ", title))
    if no_paren != base:
        cands.append(no_paren)
    # Number words -> digits on both variants
    for c in list(cands):
        words = c.split()
        conv = " ".join(NUMBER_WORDS.get(w, w) for w in words)
        if conv != c and conv not in cands:
            cands.append(conv)
    return cands


def parse_ml_title(raw: str):
    """'Toy Story (1995)' -> ('Toy Story', 1995)."""
    m = re.match(r"^(.*)\s\((\d{4})\)\s*$", raw.strip())
    if m:
        return m.group(1).strip(), int(m.group(2))
    return raw.strip(), None


def main():
    print("Loading IMDb title.basics (movies only)...")
    basics = pd.read_csv(
        IMDB / "title.basics.tsv.gz",
        sep="\t",
        compression="gzip",
        # NOTE: current IMDb dumps moved directors/writers to title.crew.tsv.gz
        # (a third file). Per scope we only use the two files, so `directors`
        # is emitted empty; see docs/imdb_join.md for how to populate it.
        usecols=["tconst", "titleType", "primaryTitle", "startYear", "runtimeMinutes"],
        dtype={"tconst": str, "titleType": str, "primaryTitle": str},
        na_values=["\\N"],
        low_memory=False,
    )
    basics = basics[basics["titleType"] == "movie"].copy()
    basics["startYear"] = pd.to_numeric(basics["startYear"], errors="coerce")
    basics = basics.dropna(subset=["startYear", "primaryTitle"])
    basics["startYear"] = basics["startYear"].astype(int)
    basics["norm_title"] = basics["primaryTitle"].map(normalize_title)
    print(f"  {len(basics):,} movie titles")

    print("Loading IMDb title.ratings...")
    ratings = pd.read_csv(
        IMDB / "title.ratings.tsv.gz",
        sep="\t",
        compression="gzip",
        dtype={"tconst": str},
    )
    print(f"  {len(ratings):,} rated titles")

    imdb = basics.merge(ratings, on="tconst", how="left")
    # Deduplicate: same (norm_title, startYear) can map to several tconsts
    # (remakes/re-releases). Keep the one with most votes (most prominent).
    imdb = imdb.sort_values("numVotes", ascending=False).drop_duplicates(
        ["norm_title", "startYear"], keep="first"
    )
    print(f"  {len(imdb):,} unique (title, year) keys")

    print("Loading MovieLens movies.dat...")
    ml = pd.read_csv(
        ML_MOVIES, sep="::", engine="python", encoding="latin-1",
        names=["movieId", "title_raw", "genres"],
    )
    ml[["title", "year"]] = ml["title_raw"].apply(
        lambda r: pd.Series(parse_ml_title(r))
    )
    ml["year"] = pd.to_numeric(ml["year"], errors="coerce").astype("Int64")
    ml["title_cands"] = ml["title"].map(candidate_titles)
    print(f"  {len(ml):,} ml-1m movies")

    # Priority join: try candidate titles in order, keep first hit per movieId.
    imdb_lookup = imdb.set_index(["norm_title", "startYear"], drop=False)
    rows = []
    for _, r in ml.iterrows():
        hit = None
        for cand in r["title_cands"]:
            key = (cand, int(r["year"])) if pd.notna(r["year"]) else None
            if key and key in imdb_lookup.index:
                hit = imdb_lookup.loc[key]
                if isinstance(hit, pd.DataFrame):  # shouldn't happen post-dedup
                    hit = hit.iloc[0]
                break
        rows.append(hit)
    imdb_hits = pd.DataFrame(
        [h.to_dict() if h is not None else {} for h in rows], index=ml.index
    )
    joined = pd.concat([ml, imdb_hits], axis=1)
    hit = joined["tconst"].notna().sum()
    print(f"JOIN HIT RATE: {hit}/{len(ml)} = {hit/len(ml):.1%}")

    out = joined[[
        "movieId", "tconst", "primaryTitle", "startYear",
        "runtimeMinutes", "averageRating", "numVotes",
    ]].rename(columns={"averageRating": "imdb_averageRating", "numVotes": "imdb_numVotes"})
    out["directors"] = pd.NA  # needs title.crew.tsv.gz; see docs/imdb_join.md
    out["startYear"] = out["startYear"].astype("Int64")
    out = out[["movieId", "tconst", "primaryTitle", "startYear", "directors",
               "runtimeMinutes", "imdb_averageRating", "imdb_numVotes"]]
    out.to_csv(OUT, index=False)
    print(f"Wrote {OUT} ({len(out):,} rows)")

    print("\n--- 5 sample joined rows ---")
    print(out.dropna(subset=["tconst"]).head(5).to_string(index=False))
    print("\n--- 5 misses (no IMDb match) ---")
    misses = joined[joined["tconst"].isna()][["movieId", "title_raw"]].head(5)
    print(misses.to_string(index=False))


if __name__ == "__main__":
    main()
