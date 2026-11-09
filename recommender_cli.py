"""Run content-based movie recommendations with the TF-IDF/KNN model.

Examples:
    python recommender_cli.py recommender_data/interactions.csv --user-id 12345
    python recommender_cli.py recommender_data/movie_features.csv --like "The Shining" --like "The Lighthouse" --like "American Psycho"
    python recommender_cli.py recommender_data/interactions.csv --like "The Shining" --like "The Lighthouse" --like "American Psycho"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from content_recommender import ContentRecommender


DEFAULT_DATA_PATH = Path(__file__).parent / "recommender_data" / "movie_features.csv"


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
        help="Recommend using an existing user's ratings (use this or --like).",
    )
    parser.add_argument(
        "--like",
        action="append",
        default=[],
        metavar="TITLE",
        help="Liked movie title; repeat to build a multi-movie profile.",
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
    if bool(args.user_id) == bool(args.like):
        print("Error: provide exactly one recommendation source: --user-id or --like.")
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
            recommendations = recommender.get_top_n_recommendations_for_titles(
                args.like, n=args.top_n
            )
        else:
            recommendations = recommender.get_top_n_recommendations(
                args.user_id, n=args.top_n
            )
        print_recommendations(recommendations)
    except (OSError, ValueError) as error:
        print(f"Error: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
