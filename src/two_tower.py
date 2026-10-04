"""PyTorch two-tower baseline for MovieLens-1M (primary model).

User tower: user-side structured features -> MLP -> embedding (dim 48).
Movie tower: movie-side structured features -> MLP -> embedding (dim 48).
Sparse IDs: learned user/movie ID embeddings (dim 16, index 0 = unknown for
cold IDs unseen in train) concatenated onto the tower outputs -- on by
default, disable with --no-id-embeddings.
Score: dot(user_repr, movie_repr) + bias (+ optional learned wide term),
trained with BCEWithLogitsLoss. Same train-period-only feature discipline
as the LightGBM baseline: towers see only train-built aggregates; ID maps
are built from train only.

Candidate features from the science loop are routed by
provenance["tower"]:
  * "user"  -> appended to the user tower's input
  * "movie" -> appended to the movie tower's input
  * "wide"  -> interaction features (user x movie); enter as a learned
               additive term w*f + b instead of forcing them into a tower

Trains on 800k rows in ~1-2 min on CPU (4 epochs, batch 65536).
LightGBM remains available via run_scientist.py --model lgbm.
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from baseline import TARGET, build_train_only_stats, featurize
from data import load_all
from split import temporal_split

RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results",
    "two_tower_baseline.json",
)

USER_COLS = ["user_rating_count", "user_mean_rating", "gender_m", "age",
             "occupation"]
# movie cols = movie aggregates + genre one-hots (resolved dynamically)
MOVIE_AGG_COLS = ["movie_rating_count", "movie_mean_rating"]

EPOCHS = 4
BATCH_SIZE = 65536
EMB_DIM = 48
ID_EMB_DIM = 16
LR = 3e-3
SEED = 42


def _movie_cols(X: pd.DataFrame) -> list[str]:
    return MOVIE_AGG_COLS + [c for c in X.columns if c.startswith("genre_")]


def _mlp(d_in: int, emb_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_in, 128), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, emb_dim),
    )


class TwoTower(nn.Module):
    def __init__(self, d_user: int, d_movie: int, d_wide: int = 0,
                 emb_dim: int = EMB_DIM, n_users: int = 0, n_movies: int = 0,
                 id_emb_dim: int = ID_EMB_DIM, use_id_emb: bool = True):
        super().__init__()
        self.user_tower = _mlp(d_user, emb_dim)
        self.movie_tower = _mlp(d_movie, emb_dim)
        self.use_id_emb = use_id_emb and n_users > 0 and n_movies > 0
        if self.use_id_emb:
            # index 0 = unknown (cold IDs unseen in train).
            # Small init: default N(0,1) would dominate the dot product,
            # saturate the sigmoid, and stall learning.
            self.user_id_emb = nn.Embedding(n_users + 1, id_emb_dim)
            self.movie_id_emb = nn.Embedding(n_movies + 1, id_emb_dim)
            nn.init.normal_(self.user_id_emb.weight, std=0.01)
            nn.init.normal_(self.movie_id_emb.weight, std=0.01)
        self.wide = nn.Linear(d_wide, 1, bias=False) if d_wide > 0 else None
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, xu, xm, iu=None, im=None, xw=None):
        u = self.user_tower(xu)
        m = self.movie_tower(xm)
        if self.use_id_emb:
            u = torch.cat([u, self.user_id_emb(iu)], dim=1)
            m = torch.cat([m, self.movie_id_emb(im)], dim=1)
        logit = (u * m).sum(dim=1) + self.bias
        if xw is not None and self.wide is not None:
            logit = logit + self.wide(xw).squeeze(1)
        return logit


def _id_map(ids: pd.Series) -> dict:
    """Map raw IDs to 1-based embedding indices; 0 = unknown (cold ID)."""
    uniq = sorted(pd.unique(ids))
    return {v: i + 1 for i, v in enumerate(uniq)}


def prepare_tower_inputs(train: pd.DataFrame, valid: pd.DataFrame,
                         stats: dict) -> dict:
    """Featurize once; split into per-tower float32 matrices (unscaled).

    Also builds ID index arrays for the sparse ID embeddings: IDs are mapped
    from train only, so validation IDs unseen in train map to 0 (unknown).
    """
    Xt = featurize(train, stats)
    Xv = featurize(valid, stats)
    mcols = _movie_cols(Xt)
    user_map = _id_map(train["user_id"])
    movie_map = _id_map(train["movie_id"])
    prep = {
        "user_cols": USER_COLS,
        "movie_cols": mcols,
        "Xu_train": Xt[USER_COLS].to_numpy(np.float32),
        "Xm_train": Xt[mcols].to_numpy(np.float32),
        "iu_train": train["user_id"].map(user_map).fillna(0).to_numpy(np.int64),
        "im_train": train["movie_id"].map(movie_map).fillna(0).to_numpy(np.int64),
        "iu_valid": valid["user_id"].map(user_map).fillna(0).to_numpy(np.int64),
        "im_valid": valid["movie_id"].map(movie_map).fillna(0).to_numpy(np.int64),
        "n_users": len(user_map),
        "n_movies": len(movie_map),
        "Xu_valid": Xv[USER_COLS].to_numpy(np.float32),
        "Xm_valid": Xv[mcols].to_numpy(np.float32),
        "y_train": train[TARGET].to_numpy(np.float32),
        "y_valid": valid[TARGET].to_numpy(np.float32),
    }
    return prep


def _scale(train_arr: np.ndarray, valid_arr: np.ndarray):
    sc = StandardScaler()
    return sc.fit_transform(train_arr).astype(np.float32), \
        sc.transform(valid_arr).astype(np.float32), sc


def train_two_tower(prep: dict, extra: dict | None = None,
                    epochs: int = EPOCHS, batch_size: int = BATCH_SIZE,
                    emb_dim: int = EMB_DIM, lr: float = LR, seed: int = SEED,
                    use_id_emb: bool = True, id_emb_dim: int = ID_EMB_DIM,
                    verbose: bool = True) -> dict:
    """Train the two-tower model.

    extra: {tower: (train_col_2d, valid_col_2d)} with tower in
    {"user", "movie", "wide"}. Extra columns are standardized with their own
    train-fit scaler and appended to that tower's input (or the wide term).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xu_tr, Xu_va, _ = _scale(prep["Xu_train"], prep["Xu_valid"])
    Xm_tr, Xm_va, _ = _scale(prep["Xm_train"], prep["Xm_valid"])

    wide_tr = wide_va = None
    d_wide = 0
    if extra:
        w_tr_parts, w_va_parts = [], []
        for tower, (ctr, cva) in extra.items():
            ctr = np.asarray(ctr, dtype=np.float32).reshape(len(ctr), -1)
            cva = np.asarray(cva, dtype=np.float32).reshape(len(cva), -1)
            str_, sva, _ = _scale(ctr, cva)
            if tower == "user":
                Xu_tr = np.concatenate([Xu_tr, str_], axis=1)
                Xu_va = np.concatenate([Xu_va, sva], axis=1)
            elif tower == "movie":
                Xm_tr = np.concatenate([Xm_tr, str_], axis=1)
                Xm_va = np.concatenate([Xm_va, sva], axis=1)
            elif tower == "wide":
                w_tr_parts.append(str_)
                w_va_parts.append(sva)
            else:
                raise ValueError(f"unknown tower: {tower}")
        if w_tr_parts:
            wide_tr = np.concatenate(w_tr_parts, axis=1)
            wide_va = np.concatenate(w_va_parts, axis=1)
            d_wide = wide_tr.shape[1]

    device = torch.device("cpu")
    model = TwoTower(Xu_tr.shape[1], Xm_tr.shape[1], d_wide, emb_dim,
                     n_users=prep.get("n_users", 0),
                     n_movies=prep.get("n_movies", 0),
                     id_emb_dim=id_emb_dim,
                     use_id_emb=use_id_emb).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    tensors_tr = [torch.from_numpy(Xu_tr), torch.from_numpy(Xm_tr),
                  torch.from_numpy(prep["y_train"])]
    if model.use_id_emb:
        tensors_tr += [torch.from_numpy(prep["iu_train"]),
                       torch.from_numpy(prep["im_train"])]
    if wide_tr is not None:
        tensors_tr.append(torch.from_numpy(wide_tr))
    loader = DataLoader(TensorDataset(*tensors_tr), batch_size=batch_size,
                        shuffle=True)

    def _unpack(batch, is_train: bool):
        # batch layout: xu, xm, [y,] [iu, im,] [xw]
        xu, xm = batch[0], batch[1]
        idx = 3 if is_train else 2
        iu = im = xw = None
        if model.use_id_emb:
            iu, im = batch[idx], batch[idx + 1]
            idx += 2
        if len(batch) > idx:
            xw = batch[idx]
        y = batch[2] if is_train else None
        return xu, xm, iu, im, xw, y

    model.train()
    t0 = time.time()
    for ep in range(epochs):
        tot, nb = 0.0, 0
        for batch in loader:
            xu, xm, iu, im, xw, y = _unpack(batch, True)
            opt.zero_grad()
            loss = loss_fn(model(xu, xm, iu, im, xw), y)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        if verbose:
            print(f"  epoch {ep + 1}/{epochs} loss={tot / nb:.4f}")

    model.eval()
    with torch.no_grad():
        tensors_va = [torch.from_numpy(Xu_va), torch.from_numpy(Xm_va)]
        if model.use_id_emb:
            tensors_va += [torch.from_numpy(prep["iu_valid"]),
                           torch.from_numpy(prep["im_valid"])]
        if wide_va is not None:
            tensors_va.append(torch.from_numpy(wide_va))
        logits = []
        vloader = DataLoader(TensorDataset(*tensors_va), batch_size=batch_size)
        for batch in vloader:
            xu, xm, iu, im, xw, _ = _unpack(batch, False)
            logits.append(model(xu, xm, iu, im, xw))
        proba = torch.sigmoid(torch.cat(logits)).numpy()

    yv = prep["y_valid"]
    auc = float(roc_auc_score(yv, proba))
    ll = float(log_loss(yv, proba))
    if verbose:
        print(f"  valid AUC={auc:.4f} logloss={ll:.4f} "
              f"({time.time() - t0:.0f}s train+eval)")
    return {"model": model, "auc": auc, "log_loss": ll, "proba": proba,
            "d_user": Xu_tr.shape[1], "d_movie": Xm_tr.shape[1],
            "d_wide": d_wide}


def main() -> dict:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--id-embeddings", dest="id_emb", action="store_true",
                    default=True, help="learned user/movie ID embeddings "
                    "(default: on)")
    ap.add_argument("--no-id-embeddings", dest="id_emb", action="store_false",
                    help="disable ID embeddings (dense features only)")
    args = ap.parse_args()

    t0 = time.time()
    print("loading data + temporal split...")
    df = load_all()
    train, valid, cutoff_ts = temporal_split(df)
    print(f"train={len(train):,} valid={len(valid):,}")

    print("building train-only stats + tower inputs...")
    stats = build_train_only_stats(train)
    prep = prepare_tower_inputs(train, valid, stats)
    print(f"user tower in: {prep['Xu_train'].shape[1]} cols "
          f"({', '.join(prep['user_cols'])})")
    print(f"movie tower in: {prep['Xm_train'].shape[1]} cols "
          f"(aggregates + {len(prep['movie_cols']) - 2} genre one-hots)")
    print(f"id embeddings: {'on' if args.id_emb else 'off'} "
          f"({prep['n_users']} users, {prep['n_movies']} movies, "
          f"dim={ID_EMB_DIM})")

    print(f"training two-tower (epochs={EPOCHS}, batch={BATCH_SIZE}, "
          f"emb_dim={EMB_DIM})...")
    res = train_two_tower(prep, use_id_emb=args.id_emb)

    metrics = {
        "valid_auc": round(res["auc"], 4),
        "valid_log_loss": round(res["log_loss"], 4),
        "n_train": len(train),
        "n_valid": len(valid),
        "emb_dim": EMB_DIM,
        "id_emb_dim": ID_EMB_DIM if args.id_emb else 0,
        "id_embeddings": args.id_emb,
        "epochs": EPOCHS,
        "elapsed_s": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nvalidation AUC      = {metrics['valid_auc']:.4f}")
    print(f"validation log-loss = {metrics['valid_log_loss']:.4f}")
    print(f"metrics written to {RESULTS_PATH}")
    return metrics


if __name__ == "__main__":
    main()
