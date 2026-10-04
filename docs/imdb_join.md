# IMDb external data source — join notes

## Sources

Downloaded 2026-10-03 into `data/imdb/` (re-runnable via `src/imdb_join.py`):

- https://datasets.imdbws.com/title.basics.tsv.gz — 644,860 `titleType='movie'` titles
  (tconst, primaryTitle, startYear, runtimeMinutes, genres)
- https://datasets.imdbws.com/title.ratings.tsv.gz — 1,716,471 rated titles
  (tconst, averageRating, numVotes)

IMDb publishes these as free bulk TSVs (no API key, no scraping). License: for
personal/non-commercial use; see https://developer.imdb.com/non-commercial-datasets/.

## Join method (`src/imdb_join.py`)

MovieLens-1M `movies.dat` titles look like `Toy Story (1995)`,
`Seven (Se7en) (1995)`, `Postino, Il (The Postman) (1994)`. IMDb's
`title.basics` has `primaryTitle` + `startYear` but different conventions
(`12 Monkeys`, `Seven`, `Il Postino`).

Pipeline:
1. Parse ML title → `(title, year)` via trailing `(YYYY)`.
2. Normalize: NFKD → ASCII, lowercase, move trailing articles to front
   (`"X, The"` → `"the x"`), strip punctuation.
3. Candidate keys per movie, tried in order:
   - normalized title + year
   - parenthetical alt-titles removed (`"seven (se7en)"` → `"seven"`) + year
   - number words → digits on either variant (`"twelve monkeys"` → `"12 monkeys"`)
4. IMDb side: filter `titleType='movie'`, dedupe `(norm_title, startYear)` keeping
   the tconst with the most votes (handles remakes/re-releases).
5. Left join → `data/imdb/movie_enrichment.csv` with one row per ML movie:
   `movieId, tconst, primaryTitle, startYear, directors, runtimeMinutes,
   imdb_averageRating, imdb_numVotes`.

## Hit rate

- **3,350 / 3,883 movies matched = 86.3%**
- **94.1% of the 1,000,209 ratings** belong to matched movies (misses are obscure
  titles few users rated — the modeling-relevant coverage is what matters).

Match examples (ML title → IMDb primaryTitle):

| MovieLens | IMDb | note |
|---|---|---|
| `Toy Story (1995)` | `Toy Story` (tt0114709) | direct |
| `Twelve Monkeys (1995)` | `12 Monkeys` (1995) | number-word fallback |
| `Seven (Se7en) (1995)` | `Seven` (1995) | parenthetical stripped |
| `Shanghai Triad (Yao a yao yao dao waipo qiao) (1995)` | `Shanghai Triad` (1995) | parenthetical stripped |
| `Lawnmower Man, The (1992)` | `The Lawnmower Man` (1992) | article moved |

Miss examples (no IMDb match — acceptable for v1):

| MovieLens | likely reason |
|---|---|
| `Persuasion (1995)` | TV movie in IMDb (`titleType='tvMovie'`, filtered out) |
| `Postino, Il (The Postman) (1994)` | IMDb primary title is `Il Postino`; double article form not covered |
| `Lawnmower Man 2: Beyond Cyberspace (1996)` | obscure sequel, no `movie`-type entry with that year |
| `In the Bleak Midwinter (1995)` | alternate-title-only match in IMDb |

## ⚠️ TEMPORAL CAVEAT — read before using as features

**`imdb_averageRating` / `imdb_numVotes` are all-time aggregates computed over
IMDb's entire history up to the download date (Oct 2026).** Using them raw as
features to "predict" a MovieLens rating from 2000–2001 is **temporal leakage**:
the feature contains votes cast *after* the prediction timestamp. A model using
raw `imdb_averageRating` will look great on the validation split and fail in
production. The Day 5–6 leakage checker must flag any feature built directly on
these columns.

### Point-in-time-correct pattern

The fix is to only use information that existed *before* the prediction
timestamp. Concrete example — **director's track record**:

```python
# WRONG (leaky): director's all-time average IMDb rating
feat = director_alltime_mean_imdb_rating[movie.director]

# RIGHT (point-in-time correct): for a rating made at time T of movie M
# directed by D, use the mean IMDb rating of D's movies released BEFORE T,
# where each of those movies' ratings is itself restricted to votes
# cast before T.
def director_track_record_at_T(director_D, T):
    prior = [m for m in movies_by_director[D] if m.release_date < T]
    return mean(imdb_rating_as_of(m, T) for m in prior)
```

Notes:
- `imdb_rating_as_of(m, T)` is not directly available from the bulk dump
  (IMDb only publishes the current aggregate). In practice: approximate with
  `numVotes`-weighted decay, restrict to movies released well before T
  (ratings stabilize), or treat the resulting feature as *leakage-suspicious*
  and require the agent to ablate it against a time-shifted validation.
- `runtimeMinutes`, `startYear`, and `primaryTitle`-derived features (e.g.
  title length) are **time-invariant** and safe to use raw.
- `directors` is currently empty in `movie_enrichment.csv`: current IMDb dumps
  moved director/writer credits to a third file,
  https://datasets.imdbws.com/title.crew.tsv.gz (`tconst, directors, writers`
  as nconst lists; join via `name.basics.tsv.gz` for names). Populate it with
  one extra download if the director-track-record feature is needed for the
  demo. The PIT-correct pattern above applies unchanged.
