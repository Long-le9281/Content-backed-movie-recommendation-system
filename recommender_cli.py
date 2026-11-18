"""Run content-based movie recommendations with the TF-IDF/KNN model.

Examples:
    python recommender_cli.py --user-id 123
    python recommender_cli.py --like "The Shining" --like "The Lighthouse" --like "American Psycho"
    python recommender_cli.py --ratings-file test_user_profile.csv --show-profile
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from content_recommender import ContentRecommender


DEFAULT_DATA_PATH = Path(__file__).parent / "recommender_data" / "movie_features.csv"
DEFAULT_RATINGS_PATH = (
    Path(__file__).parent / "recommender_data" / "movielens" / "ml-32m" / "ratings.csv"
)
RATINGS_CHUNK_SIZE = 2_000_000


def load_recommender(data_path: Path, min_votes: int) -> ContentRecommender:
    """Load an interaction dataset or a standalone movie-feature catalog."""
    data = pd.read_csv(data_path)
    column_aliases = {
        "imdbGenres": "genres_imdb",
        "userTags": "grouped_tags",
        "title": "movieLensTitle",
    }
    rename_columns = {
        source: target
        for source, target in column_aliases.items()
        if source in data.columns and target not in data.columns
    }
    if rename_columns:
        data = data.rename(columns=rename_columns)
    return ContentRecommender(data=data, min_votes=min_votes)


def load_ratings_file(path: Path) -> tuple[list, list[float] | None]:
    """Read movie IDs (and optional ratings) for a new user's profile.

    The CSV needs a ``movieId`` column. If it also has a ``rating`` column,
    the profile is rating-weighted; otherwise every movie counts equally.
    """
    ratings = pd.read_csv(path)
    if "movieId" not in ratings.columns:
        raise ValueError(f"Ratings file {path} must have a movieId column.")
    ratings["movieId"] = pd.to_numeric(ratings["movieId"], errors="coerce")
    columns = ["movieId"]
    if "rating" in ratings.columns:
        ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
        columns.append("rating")
    ratings = ratings.dropna(subset=columns)
    if ratings.empty:
        raise ValueError(f"Ratings file {path} contains no valid rows.")
    movie_ids = ratings["movieId"].astype("int64").tolist()
    if "rating" not in ratings.columns:
        return movie_ids, None
    return movie_ids, ratings["rating"].tolist()


def load_user_ratings(
    path: Path,
    user_id: str,
    chunk_size: int = RATINGS_CHUNK_SIZE,
    assume_sorted: bool = True,
) -> tuple[list[int], list[float]]:
    """Read one user's ratings from a MovieLens ``ratings.csv`` in chunks.

    Only that user's rows are kept in memory. MovieLens documents that
    ratings.csv is ordered by userId, so by default the scan stops at the
    first chunk that ends past the user. Pass ``assume_sorted=False`` for a
    file without that guarantee; the whole file is then scanned.
    """
    if not path.is_file():
        raise FileNotFoundError(f"MovieLens ratings file not found: {path}")
    try:
        target = int(user_id)
    except ValueError as error:
        raise ValueError(f"User ID must be an integer, got {user_id!r}.") from error

    matches = []
    with pd.read_csv(
        path, usecols=["userId", "movieId", "rating"], chunksize=chunk_size
    ) as chunks:
        for chunk in chunks:
            matches.append(chunk[chunk["userId"] == target])
            if assume_sorted and chunk["userId"].iloc[-1] > target:
                break

    user_ratings = pd.concat(matches, ignore_index=True)
    if user_ratings.empty:
        raise ValueError(f"User {target} was not found in {path}.")
    return (
        user_ratings["movieId"].astype("int64").tolist(),
        user_ratings["rating"].astype(float).tolist(),
    )


def print_profile_terms(terms: pd.DataFrame) -> None:
    """Print the strongest features in the user profile vector."""
    print("Top profile features:")
    print(terms.to_string(index=False, formatters={"weight": "{:.4f}".format}))
    print()


def print_recommendations(recommendations: pd.DataFrame) -> None:
    """Print model recommendations as a readable terminal table."""
    if recommendations.empty:
        print("No unseen movies meet the IMDb vote threshold.")
        return
    print(
        recommendations.rename(
            columns={
                "primaryTitle": "Recommended Movie",
                "genres_imdb": "IMDb Genres",
                "Similarity_Score": "Cosine Similarity",
            }
        ).to_string(index=False, formatters={"Cosine Similarity": "{:.4f}".format})
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_path",
        nargs="?",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Movie catalog or merged rating-level CSV (default: {DEFAULT_DATA_PATH}).",
    )
    parser.add_argument(
        "--user-id",
        help="Recommend for an existing MovieLens user from their rating history.",
    )
    parser.add_argument(
        "--movielens-ratings",
        type=Path,
        default=DEFAULT_RATINGS_PATH,
        metavar="CSV",
        help=f"MovieLens ratings.csv used by --user-id (default: {DEFAULT_RATINGS_PATH}).",
    )
    parser.add_argument(
        "--like",
        action="append",
        default=[],
        metavar="TITLE",
        help="Liked movie title; repeat to build a multi-movie profile.",
    )
    parser.add_argument(
        "--ratings-file",
        type=Path,
        metavar="CSV",
        help="CSV with movieId (and optional rating) columns for a new user.",
    )
    parser.add_argument(
        "--show-profile",
        action="store_true",
        help="Print the highest-weighted features of the profile vector.",
    )
    parser.add_argument(
        "-n", "--top-n", type=int, default=10, help="Maximum recommendations."
    )
    parser.add_argument(
        "--min-votes",
        type=int,
        default=100,
        help="Exclude movies with fewer IMDb votes (default: 100).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run recommendations and return a process exit code."""
    args = parse_args(argv)
    sources = [bool(args.user_id), bool(args.like), args.ratings_file is not None]
    if sum(sources) != 1:
        print(
            "Error: provide exactly one recommendation source: "
            "--user-id, --like, or --ratings-file."
        )
        return 2
    if args.top_n < 1:
        print("Error: --top-n must be at least 1.")
        return 2
    if args.min_votes < 0:
        print("Error: --min-votes cannot be negative.")
        return 2

    try:
        recommender = load_recommender(args.data_path, args.min_votes)
        if args.like:
            build_profile = lambda: recommender.build_profile_from_titles(args.like)
            recommend = lambda: recommender.get_top_n_recommendations_for_titles(
                args.like, n=args.top_n
            )
        elif args.ratings_file is not None:
            movie_ids, ratings = load_ratings_file(args.ratings_file)
            build_profile = lambda: recommender.build_profile_from_movie_ids(
                movie_ids, ratings
            )
            recommend = lambda: recommender.get_top_n_recommendations_for_movie_ids(
                movie_ids, ratings, n=args.top_n
            )
        elif ContentRecommender.RATING_COLUMNS.issubset(recommender.data.columns):
            # Small merged files that already contain ratings still work.
            build_profile = lambda: recommender.generate_user_profile(args.user_id)
            recommend = lambda: recommender.get_top_n_recommendations(
                args.user_id, n=args.top_n
            )
        else:
            movie_ids, ratings = load_user_ratings(args.movielens_ratings, args.user_id)
            build_profile = lambda: recommender.build_profile_from_movie_ids(
                movie_ids, ratings, ignore_missing=True
            )
            recommend = lambda: recommender.get_top_n_recommendations_for_movie_ids(
                movie_ids, ratings, n=args.top_n, ignore_missing=True
            )
        recommendations = recommend()
        if args.show_profile:
            print_profile_terms(recommender.top_profile_terms(build_profile()))
        print_recommendations(recommendations)
    except (OSError, ValueError) as error:
        print(f"Error: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
