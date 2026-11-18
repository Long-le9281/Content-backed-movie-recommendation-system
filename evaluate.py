"""Held-out evaluation of the content recommender on MovieLens users.

For a random sample of MovieLens users, 20% of each user's ratings are hidden
at random. The recommender builds a profile from the remaining 80% and the
hidden movies the user rated 4.0 or higher are the ground truth. The same split
is scored for a popularity baseline and a random baseline.

Examples:
    python evaluate.py                          # 1000 users, content + baselines
    python evaluate.py --n-users 200            # quicker run
    python evaluate.py --reuse-split            # skip re-reading ratings.csv
    python evaluate.py --reuse-split --experiments   # add ablations / variants

Outputs (in --out-dir, default evaluation_results/):
    split.csv        the sampled users' ratings, each marked train or test
    popularity.csv   rating counts per movie, excluding the hidden test rows
    per_user.csv     every metric for every user, method and K
    summary.csv      means with 95% bootstrap confidence intervals, plus coverage
    metrics_vs_k.png recall/NDCG/hit rate against K (needs matplotlib)
    experiments_vs_k.png  the same for the --experiments variants
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from content_recommender import ContentRecommender
from recommender_cli import DEFAULT_DATA_PATH, DEFAULT_RATINGS_PATH, load_recommender


DEFAULT_TAGS_PATH = DEFAULT_RATINGS_PATH.parent / "tags.csv"
DEFAULT_OUT_DIR = Path(__file__).parent / "evaluation_results"
LIKED_THRESHOLD = 4.0
METRICS = ["hit_rate", "precision", "recall", "ndcg", "mrr", "diversity"]


# --------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------


def load_ratings(path: Path, catalog_ids: Iterable[int]) -> pd.DataFrame:
    """Read MovieLens ratings, keeping only movies that are in the catalog."""
    if not path.is_file():
        raise FileNotFoundError(f"MovieLens ratings file not found: {path}")
    ratings = pd.read_csv(
        path,
        usecols=["userId", "movieId", "rating"],
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
    )
    return ratings[ratings["movieId"].isin(set(catalog_ids))].reset_index(drop=True)


def sample_users(
    ratings: pd.DataFrame,
    n_users: int,
    min_ratings: int,
    min_liked: int,
    seed: int,
) -> np.ndarray:
    """Pick ``n_users`` users with enough ratings and liked movies to split."""
    stats = ratings.groupby("userId")["rating"].agg(
        n_ratings="size", n_liked=lambda r: int((r >= LIKED_THRESHOLD).sum())
    )
    eligible = stats.index[
        (stats["n_ratings"] >= min_ratings) & (stats["n_liked"] >= min_liked)
    ].to_numpy()
    if len(eligible) == 0:
        raise ValueError("No users meet the --min-ratings / --min-liked thresholds.")
    rng = np.random.default_rng(seed)
    size = min(n_users, len(eligible))
    return np.sort(rng.choice(eligible, size=size, replace=False))


def split_user_ratings(
    user_ratings: pd.DataFrame, test_frac: float, seed: int
) -> pd.DataFrame:
    """Randomly mark ``test_frac`` of one user's ratings as test (at least one).

    The random generator is seeded with (seed, userId), so a user's split
    does not depend on which other users were sampled.
    """
    user_id = int(user_ratings["userId"].iloc[0])
    rng = np.random.default_rng([seed, user_id])
    n = len(user_ratings)
    n_test = min(n - 1, max(1, round(test_frac * n)))
    test_positions = rng.choice(n, size=n_test, replace=False)
    split = np.full(n, "train", dtype=object)
    split[test_positions] = "test"
    return user_ratings.assign(split=split)


def build_split(
    ratings: pd.DataFrame, user_ids: np.ndarray, test_frac: float, seed: int
) -> pd.DataFrame:
    sampled = ratings[ratings["userId"].isin(set(user_ids))]
    parts = [
        split_user_ratings(group, test_frac, seed)
        for _, group in sampled.groupby("userId", sort=True)
    ]
    return pd.concat(parts, ignore_index=True)


def popularity_counts(ratings: pd.DataFrame, split: pd.DataFrame) -> pd.Series:
    """Ratings per movie, with the hidden test rows removed (no leakage)."""
    counts = ratings["movieId"].value_counts()
    test_counts = split.loc[split["split"] == "test", "movieId"].value_counts()
    counts = counts.sub(test_counts, fill_value=0).astype("int64")
    counts.index.name = "movieId"
    return counts.rename("count")


def rebuild_user_tags(tags_path: Path, exclude_users: Iterable[int]) -> pd.Series:
    """Rebuild ``userTags`` like the merge notebook, without the test users' tags.

    The catalog's tags were built from every MovieLens user, including tags the
    sampled users wrote on their own hidden movies. Dropping those users'
    tags keeps the hidden ratings from leaking into the item features.
    """
    tags = pd.read_csv(tags_path, usecols=["userId", "movieId", "tag"])
    tags = tags[~tags["userId"].isin(set(exclude_users))].dropna(subset=["tag"])
    return (
        tags.assign(tag=tags["tag"].astype(str).str.lower())
        .groupby("movieId")["tag"]
        .agg(lambda values: ", ".join(sorted(set(values))))
    )


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def ranking_metrics(recommended: Sequence, relevant: set, k: int) -> dict[str, float]:
    """Hit rate, precision, recall, NDCG and MRR of the top ``k`` items.

    NDCG uses binary relevance; the ideal ranking puts min(k, |relevant|)
    relevant movies at the top. MRR is 1/rank of the first hit, 0 if none.
    """
    if not relevant:
        raise ValueError("A user needs at least one relevant movie to be scored.")
    top = list(recommended)[:k]
    hits = [movie_id in relevant for movie_id in top]
    n_hits = sum(hits)
    dcg = sum(1.0 / math.log2(rank + 2) for rank, hit in enumerate(hits) if hit)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(min(k, len(relevant))))
    first_hit = next((rank for rank, hit in enumerate(hits) if hit), None)
    return {
        "hit_rate": float(n_hits > 0),
        "precision": n_hits / k,
        "recall": n_hits / len(relevant),
        "ndcg": dcg / idcg,
        "mrr": 0.0 if first_hit is None else 1.0 / (first_hit + 1),
    }


def intra_list_diversity(movie_ids: Sequence, genre_sets: dict) -> float:
    """Mean pairwise Jaccard distance between the recommended movies' genres."""
    sets = [genre_sets.get(movie_id, frozenset()) for movie_id in movie_ids]
    if len(sets) < 2:
        return float("nan")
    distances = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            similarity = len(sets[i] & sets[j]) / len(union) if union else 1.0
            distances.append(1.0 - similarity)
    return float(np.mean(distances))


def bootstrap_ci(
    values: np.ndarray, seed: int, n_resamples: int = 1000
) -> tuple[float, float]:
    """95% percentile bootstrap interval for the mean, resampling users."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n_resamples, len(values)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


# --------------------------------------------------------------------------
# Recommenders under test
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Variant:
    """One configuration of the content recommender."""

    name: str
    features: tuple[str, ...] = ("genres", "tags", "crew")
    weighting: str = "rating"  # "rating": weight = rating - 3; "plain": mean of liked
    rerank: bool = True  # blend IMDb averageRating into the score
    min_votes: int | None = None  # None: use --min-votes


DEFAULT_VARIANT = Variant("content")
EXPERIMENT_VARIANTS = [
    Variant("content_genres_only", features=("genres",)),
    Variant("content_genres_tags", features=("genres", "tags")),
    Variant("content_plain_mean", weighting="plain"),
    Variant("content_no_rerank", rerank=False),
    Variant("content_min_votes_10k", min_votes=10_000),
    Variant("content_min_votes_100k", min_votes=100_000),
]


def build_model(catalog: pd.DataFrame, variant: Variant, base_min_votes: int) -> ContentRecommender:
    data = catalog.copy()
    if "tags" not in variant.features:
        data["grouped_tags"] = ""
    if "crew" not in variant.features:
        data = data.drop(columns=["directors", "writers"], errors="ignore")
    if not variant.rerank:
        data = data.drop(columns=["averageRating"], errors="ignore")
    min_votes = base_min_votes if variant.min_votes is None else variant.min_votes
    model = ContentRecommender(data=data, min_votes=min_votes)
    model.load_data()
    model.fit_knn()
    return model


def recommend_content(
    model: ContentRecommender,
    train: pd.DataFrame,
    weighting: str,
    n: int,
) -> list[int]:
    """Top-n movie IDs from the training ratings; all training movies are excluded."""
    train_ids = train["movieId"].astype("int64").tolist()
    if weighting == "plain":
        liked = train.loc[train["rating"] >= LIKED_THRESHOLD, "movieId"]
        profile = model.build_profile_from_movie_ids(liked.astype("int64").tolist())
    else:
        profile = model.build_profile_from_movie_ids(
            train_ids, train["rating"].astype(float).tolist()
        )
    result = model._recommend_from_profile(profile, set(train_ids), n)
    assert model.movie_data is not None
    # The result keeps movie_data's positional index, which maps back to movieId.
    return model.movie_data.loc[result.index, "movieId"].astype("int64").tolist()


def recommend_popularity(ranked_ids: np.ndarray, train_ids: set, n: int) -> list[int]:
    out = []
    for movie_id in ranked_ids:
        if movie_id not in train_ids:
            out.append(int(movie_id))
            if len(out) == n:
                break
    return out


def recommend_random(
    eligible_ids: np.ndarray, train_ids: set, n: int, rng: np.random.Generator
) -> list[int]:
    size = min(len(eligible_ids), n + len(train_ids))
    candidates = rng.choice(eligible_ids, size=size, replace=False)
    return [int(m) for m in candidates if m not in train_ids][:n]


# --------------------------------------------------------------------------
# Evaluation loop
# --------------------------------------------------------------------------


def eligible_movie_ids(catalog: pd.DataFrame, min_votes: int) -> np.ndarray:
    votes = pd.to_numeric(catalog["numVotes"], errors="coerce")
    titled = catalog["primaryTitle"].notna() & catalog["primaryTitle"].astype(str).str.strip().ne("")
    return catalog.loc[votes.ge(min_votes) & titled, "movieId"].astype("int64").unique()


def genre_lookup(catalog: pd.DataFrame) -> dict[int, frozenset]:
    lookup = {}
    for movie_id, genres in zip(catalog["movieId"], catalog["genres_imdb"]):
        if isinstance(genres, str) and genres not in ("", "\\N"):
            lookup[int(movie_id)] = frozenset(g.strip() for g in genres.split(",") if g.strip())
    return lookup


def score_user(
    user_id: int, method: str, recommended: list[int], relevant: set, ks: list[int], genres: dict
) -> list[dict]:
    rows = []
    for k in ks:
        row = {"method": method, "userId": user_id, "k": k}
        row.update(ranking_metrics(recommended, relevant, k))
        row["diversity"] = intra_list_diversity(recommended[:k], genres)
        row["recommended"] = " ".join(str(m) for m in recommended[:k])
        rows.append(row)
    return rows


def evaluate(
    catalog: pd.DataFrame,
    split: pd.DataFrame,
    popularity: pd.Series,
    variants: list[Variant],
    ks: list[int],
    min_votes: int,
    seed: int,
    include_baselines: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Score every method on every sampled user. Returns per-user rows and info."""
    max_k = max(ks)
    eligible = eligible_movie_ids(catalog, min_votes)
    eligible_set = set(eligible.tolist())
    genres = genre_lookup(catalog)

    # Relevant = hidden, liked, and recommendable at the base --min-votes.
    # Variants with a stricter min_votes keep this same target, so their
    # score shows the real cost of hiding less-known movies.
    users = []
    skipped = {"no_relevant_test_movies": 0, "no_liked_train_movies": 0}
    for user_id, group in split.groupby("userId", sort=True):
        train = group[group["split"] == "train"]
        test = group[group["split"] == "test"]
        relevant = set(
            test.loc[test["rating"] >= LIKED_THRESHOLD, "movieId"].astype("int64")
        ) & eligible_set
        if not relevant:
            skipped["no_relevant_test_movies"] += 1
            continue
        if not (train["rating"] >= LIKED_THRESHOLD).any():
            skipped["no_liked_train_movies"] += 1
            continue
        users.append((int(user_id), train, relevant))
    if not users:
        raise ValueError("No sampled user has both liked training and relevant test movies.")

    rows: list[dict] = []
    if include_baselines:
        ranked = popularity[popularity.index.isin(eligible_set)].sort_values(
            ascending=False, kind="stable"
        ).index.to_numpy()
        rng = np.random.default_rng(seed)
        for user_id, train, relevant in users:
            train_ids = set(train["movieId"].astype("int64"))
            rows += score_user(user_id, "popularity", recommend_popularity(ranked, train_ids, max_k), relevant, ks, genres)
            rows += score_user(user_id, "random", recommend_random(eligible, train_ids, max_k, rng), relevant, ks, genres)

    for variant in variants:
        start = time.perf_counter()
        print(f"[{variant.name}] building TF-IDF features and KNN index...", flush=True)
        model = build_model(catalog, variant, min_votes)
        for i, (user_id, train, relevant) in enumerate(users, start=1):
            recommended = recommend_content(model, train, variant.weighting, max_k)
            rows += score_user(user_id, variant.name, recommended, relevant, ks, genres)
            if i % 100 == 0 or i == len(users):
                print(f"[{variant.name}] {i}/{len(users)} users", flush=True)
        print(f"[{variant.name}] done in {time.perf_counter() - start:.0f}s", flush=True)

    info = {
        "users_sampled": int(split["userId"].nunique()),
        "users_scored": len(users),
        "users_skipped": skipped,
        "eligible_movies": len(eligible),
        "mean_relevant_per_user": float(np.mean([len(r) for _, _, r in users])),
    }
    return pd.DataFrame(rows), info


def summarize(per_user: pd.DataFrame, eligible_count: int, seed: int) -> pd.DataFrame:
    """Mean and 95% bootstrap CI per method and K, plus catalog coverage."""
    rows = []
    for (method, k), group in per_user.groupby(["method", "k"], sort=False):
        row = {"method": method, "k": k, "users": len(group)}
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            low, high = bootstrap_ci(values, seed)
            row[metric] = float(np.nanmean(values))
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        recommended = set(" ".join(group["recommended"]).split())
        row["coverage"] = len(recommended) / eligible_count if eligible_count else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

# Categorical slots, light mode, from the dataviz reference palette (fixed order).
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def plot_metrics(summary: pd.DataFrame, path: Path, methods: list[str], title: str) -> bool:
    """Save recall/NDCG/hit rate vs K as small multiples. False if no matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    if len(methods) > len(SERIES_COLORS):
        raise ValueError(f"At most {len(SERIES_COLORS)} methods fit in one plot.")
    summary = summary[summary["method"].isin(methods)]
    panels = [("recall", "Recall@K"), ("ndcg", "NDCG@K"), ("hit_rate", "Hit rate@K")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True)
    for ax, (metric, title) in zip(axes, panels):
        for color, method in zip(SERIES_COLORS, methods):
            data = summary[summary["method"] == method].sort_values("k")
            ax.fill_between(data["k"], data[f"{metric}_ci_low"], data[f"{metric}_ci_high"], color=color, alpha=0.15, linewidth=0)
            ax.plot(data["k"], data[metric], color=color, linewidth=2, marker="o", markersize=5, label=method)
        ax.set_title(title, loc="left", fontsize=11, color="#222222")
        ax.set_xlabel("K (list length)", color="#555555")
        ax.set_xticks(sorted(summary["k"].unique()))
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
        ax.tick_params(colors="#555555")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#bbbbbb")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"{title} (shaded: 95% bootstrap CI)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def print_summary(summary: pd.DataFrame) -> None:
    table = summary[["method", "k", "hit_rate", "precision", "recall", "ndcg", "mrr", "diversity", "coverage"]]
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 140):
        for k, group in table.groupby("k"):
            print(f"\n=== K = {k} ===")
            print(group.drop(columns="k").sort_values("ndcg", ascending=False).to_string(index=False))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="movie_features.csv catalog.")
    parser.add_argument("--ratings-path", type=Path, default=DEFAULT_RATINGS_PATH, help="MovieLens ratings.csv.")
    parser.add_argument("--tags-path", type=Path, default=DEFAULT_TAGS_PATH, help="MovieLens tags.csv.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-users", type=int, default=1000)
    parser.add_argument("--min-ratings", type=int, default=20, help="Minimum ratings per sampled user.")
    parser.add_argument("--min-liked", type=int, default=10, help="Minimum ratings >= 4.0 per sampled user.")
    parser.add_argument("--test-frac", type=float, default=0.2, help="Share of each user's ratings hidden.")
    parser.add_argument("--k", type=int, nargs="+", default=[10, 20, 50], help="List lengths to score.")
    parser.add_argument("--min-votes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-split", action="store_true", help="Load split.csv/popularity.csv from --out-dir.")
    parser.add_argument("--experiments", action="store_true", help="Also run the ablation and variant configs.")
    parser.add_argument("--keep-test-user-tags", action="store_true", help="Use catalog tags as-is (leaks; for comparison).")
    args = parser.parse_args(argv)
    if args.n_users < 1:
        parser.error("--n-users must be at least 1.")
    if not 0 < args.test_frac < 1:
        parser.error("--test-frac must be between 0 and 1.")
    if min(args.k) < 1:
        parser.error("--k values must be at least 1.")
    args.k = sorted(set(args.k))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    split_path = args.out_dir / "split.csv"
    popularity_path = args.out_dir / "popularity.csv"

    catalog = load_recommender(args.data_path, args.min_votes).data
    assert catalog is not None
    catalog = catalog.dropna(subset=["movieId"]).copy()
    catalog["movieId"] = catalog["movieId"].astype("int64")

    if args.reuse_split:
        if not split_path.is_file() or not popularity_path.is_file():
            raise FileNotFoundError(f"--reuse-split needs {split_path} and {popularity_path}.")
        print(f"Reusing split from {split_path}")
        split = pd.read_csv(split_path)
        popularity = pd.read_csv(popularity_path, index_col="movieId")["count"]
    else:
        print(f"Reading {args.ratings_path} (32M rows, takes a minute)...", flush=True)
        ratings = load_ratings(args.ratings_path, catalog["movieId"])
        users = sample_users(ratings, args.n_users, args.min_ratings, args.min_liked, args.seed)
        print(f"Sampled {len(users)} users; splitting {1 - args.test_frac:.0%}/{args.test_frac:.0%}.", flush=True)
        split = build_split(ratings, users, args.test_frac, args.seed)
        popularity = popularity_counts(ratings, split)
        del ratings
        split.to_csv(split_path, index=False)
        popularity.to_csv(popularity_path)

    if not args.keep_test_user_tags:
        print("Rebuilding tags without the sampled users' tags...", flush=True)
        clean_tags = rebuild_user_tags(args.tags_path, split["userId"].unique())
        catalog["grouped_tags"] = catalog["movieId"].map(clean_tags).fillna("")

    variants = [DEFAULT_VARIANT] + (EXPERIMENT_VARIANTS if args.experiments else [])
    per_user, info = evaluate(catalog, split, popularity, variants, args.k, args.min_votes, args.seed)
    summary = summarize(per_user, info["eligible_movies"], args.seed)

    per_user.to_csv(args.out_dir / "per_user.csv", index=False)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    run_info = {**info, **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}
    (args.out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))

    print(f"\nScored {info['users_scored']} of {info['users_sampled']} users "
          f"(skipped: {info['users_skipped']}); "
          f"{info['mean_relevant_per_user']:.1f} relevant hidden movies per user on average.")
    print_summary(summary)
    plots = [("metrics_vs_k.png", [DEFAULT_VARIANT.name, "popularity", "random"], "Content model vs baselines")]
    if args.experiments:
        plots.append(("experiments_vs_k.png", [v.name for v in variants], "Content model variants"))
    for filename, methods, title in plots:
        if not plot_metrics(summary, args.out_dir / filename, methods, title):
            print("\nmatplotlib is not installed; skipped the plots (pip install matplotlib).")
            break
        print(f"Plot saved to {args.out_dir / filename}")
    print(f"Results written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
