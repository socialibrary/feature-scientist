"""Data loading for the Feature Scientist harness (MovieLens-1M).

Public dataset: https://grouplens.org/datasets/movielens/1m/
Local copy expected at <project>/data/ml-1m/ (override with ML1M_DIR env var).

File layouts (from the dataset README):
  ratings.dat : UserID::MovieID::Rating::Timestamp
  users.dat   : UserID::Gender::Age::Occupation::Zip-code
  movies.dat  : MovieID::Title::Genres

This module only *loads and joins* -- no modeling, no splitting. The label is
binary: liked = (rating >= 4), matching the ranking-style setup of the demo.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

DATA_DIR = Path(
    os.environ.get("ML1M_DIR", Path(__file__).resolve().parent.parent / "data" / "ml-1m")
)

RATINGS_COLS = ["user_id", "movie_id", "rating", "timestamp"]
USERS_COLS = ["user_id", "gender", "age", "occupation", "zip"]
MOVIES_COLS = ["movie_id", "title", "genres"]


def _read_dat(path: Path, names: list[str]) -> pd.DataFrame:
    # '::' is a multi-character separator -> requires the python engine.
    # latin-1 per the dataset README (movie titles contain non-ASCII bytes).
    return pd.read_csv(path, sep="::", engine="python", encoding="latin-1", names=names)


def load_ratings(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    df = _read_dat(data_dir / "ratings.dat", RATINGS_COLS)
    df["timestamp"] = df["timestamp"].astype("int64")
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df


def load_users(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return _read_dat(data_dir / "users.dat", USERS_COLS)


def load_movies(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return _read_dat(data_dir / "movies.dat", MOVIES_COLS)


def load_all(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Single joined dataframe: one row per rating, with user/movie side info.

    Adds:
      liked    - binary label, 1 when rating >= 4
      datetime - UTC timestamp of the rating (used for the temporal split)
    """
    ratings = load_ratings(data_dir)
    users = load_users(data_dir)
    movies = load_movies(data_dir)

    df = (
        ratings.merge(users, on="user_id", how="left", validate="many_to_one")
               .merge(movies, on="movie_id", how="left", validate="many_to_one")
    )
    df["liked"] = (df["rating"] >= 4).astype("int8")
    return df


if __name__ == "__main__":
    df = load_all()
    print(f"rows={len(df):,} users={df.user_id.nunique():,} "
          f"movies={df.movie_id.nunique():,}")
    print(f"time range: {df.datetime.min()} -> {df.datetime.max()}")
    print(f"positive rate (liked): {df.liked.mean():.3f}")
    print(df.head(3).to_string())
