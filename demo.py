"""Try the recommender on preset taste profiles without retyping CLI commands.

The model is built once and reused for every query, so running several presets
or comparisons is much faster than calling recommender_cli.py repeatedly.

From a notebook or the Python shell:
    from demo import *
    model = load_model()
    run_preset(model, "mixed_taste")
    run_titles(model, ["Alien", "Hereditary"], expected_genre="Horror")
    
"""


'''
python demo.py                       # every preset, then a summary table
python demo.py horror_fan ghibli     # just these presets
python demo.py --list                # show the presets
python demo.py horror_fan --compare  # same profile at min_votes 100 / 10k / 100k
python demo.py --user-id 1           # a MovieLens user's history
'''

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Sequence

import pandas as pd

from content_recommender import ContentRecommender
from recommender_cli import (
    DEFAULT_DATA_PATH,
    DEFAULT_RATINGS_PATH,
    load_recommender,
    load_user_ratings,
)


# Each preset is a set of liked titles plus the IMDb genre we would expect most
# recommendations to share. ``None`` means there is no single expected genre.
PRESETS: dict[str, dict] = {
    "horror_fan": {
        "titles": ["The Shining", "Hereditary", "The Exorcist"],
        "genre": "Horror",
    },
    "nolan_fan": {
        "titles": ["Inception", "Interstellar", "The Prestige", "Memento"],
        "genre": "Sci-Fi",
    },
    "rom_com": {
        "titles": [
            "Notting Hill",
            "When Harry Met Sally...",
            "Pretty Woman",
            "10 Things I Hate About You",
        ],
        "genre": "Romance",
    },
    "ghibli": {
        "titles": ["Spirited Away", "My Neighbor Totoro", "Princess Mononoke"],
        "genre": "Animation",
    },
    "tarantino": {
        "titles": ["Pulp Fiction", "Reservoir Dogs", "Kill Bill: Vol. 1"],
        "genre": "Crime",
    },
    "sci_fi_classics": {
        "titles": ["The Matrix", "Blade Runner", "Alien"],
        "genre": "Sci-Fi",
    },
    # Two unrelated tastes: shows how averaging can land between them.
    "mixed_taste": {
        "titles": ["The Shining", "Hereditary", "Notting Hill", "Pretty Woman"],
        "genre": None,
    },
}


def load_model(
    data_path: Path = DEFAULT_DATA_PATH, min_votes: int = 100
) -> ContentRecommender:
    """Load the catalog and build TF-IDF vectors and the KNN index once."""
    start = time.perf_counter()
    model = load_recommender(data_path, min_votes)
    model.fit_knn()
    assert model.movie_data is not None and model.vectorizer is not None
    print(
        f"Model ready in {time.perf_counter() - start:.1f}s: "
        f"{len(model.movie_data):,} movies, "
        f"{len(model.vectorizer.vocabulary_):,} features, min_votes={min_votes}"
    )
    return model


def genre_hit_rate(recommendations: pd.DataFrame, genre: str) -> float:
    """Share of recommendations whose IMDb genres include ``genre``."""
    if recommendations.empty:
        return float("nan")
    hits = recommendations["genres_imdb"].fillna("").map(
        lambda genres: genre in [part.strip() for part in str(genres).split(",")]
    )
    return float(hits.mean())


def genre_breakdown(recommendations: pd.DataFrame) -> pd.Series:
    """Count how often each IMDb genre appears in the recommendations."""
    genres = recommendations["genres_imdb"].fillna("").str.split(",").explode()
    return genres[genres.str.strip() != ""].str.strip().value_counts()


def _print_result(
    model: ContentRecommender,
    profile,
    recommendations: pd.DataFrame,
    expected_genre: str | None,
    show_terms: int,
) -> None:
    if show_terms:
        terms = model.top_profile_terms(profile, k=show_terms)
        print("Top profile features: " + ", ".join(
            f"{term} ({weight:.3f})" for term, weight in zip(terms["term"], terms["weight"])
        ))
    if recommendations.empty:
        print("No unseen movies meet the IMDb vote threshold.")
        return
    print(
        recommendations.rename(
            columns={
                "primaryTitle": "Recommended Movie",
                "genres_imdb": "IMDb Genres",
                "Similarity_Score": "Similarity",
            }
        ).to_string(index=False, formatters={"Similarity": "{:.4f}".format})
    )
    if expected_genre:
        rate = genre_hit_rate(recommendations, expected_genre)
        print(f"{expected_genre} hit rate: {rate:.0%}")
    else:
        top = genre_breakdown(recommendations).head(5)
        print("Genre mix: " + ", ".join(f"{genre} {count}" for genre, count in top.items()))


def run_titles(
    model: ContentRecommender,
    titles: list[str],
    n: int = 10,
    expected_genre: str | None = None,
    show_terms: int = 8,
) -> pd.DataFrame:
    """Recommend from liked titles, print the result, and return it."""
    profile = model.build_profile_from_titles(titles)
    recommendations = model.get_top_n_recommendations_for_titles(titles, n=n)
    print(f"Liked: {', '.join(titles)}")
    _print_result(model, profile, recommendations, expected_genre, show_terms)
    return recommendations


def run_preset(model: ContentRecommender, name: str, n: int = 10) -> pd.DataFrame:
    """Run one entry from ``PRESETS``."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset {name!r}. Choose from: {', '.join(PRESETS)}")
    preset = PRESETS[name]
    print(f"\n=== {name} ===")
    return run_titles(model, preset["titles"], n=n, expected_genre=preset["genre"])


def run_user(
    model: ContentRecommender,
    user_id: int | str,
    n: int = 10,
    ratings_path: Path = DEFAULT_RATINGS_PATH,
    show_terms: int = 8,
) -> pd.DataFrame:
    """Recommend for a MovieLens user from their rating history."""
    movie_ids, ratings = load_user_ratings(ratings_path, str(user_id))
    liked = sum(rating >= 4.0 for rating in ratings)
    print(f"\n=== MovieLens user {user_id} ===")
    print(f"{len(ratings)} ratings, {liked} of them 4.0 or higher")
    profile = model.build_profile_from_movie_ids(movie_ids, ratings, ignore_missing=True)
    recommendations = model.get_top_n_recommendations_for_movie_ids(
        movie_ids, ratings, n=n, ignore_missing=True
    )
    _print_result(model, profile, recommendations, None, show_terms)
    return recommendations


def compare(
    model: ContentRecommender,
    titles: list[str],
    min_votes_values: Sequence[int] = (100, 10_000, 100_000),
    n: int = 10,
) -> pd.DataFrame:
    """Show the same profile's top-n titles side by side for each min_votes.

    Changing ``min_votes`` only changes which movies may be recommended, so
    the model does not need rebuilding.
    """
    original = model.min_votes
    columns = {}
    try:
        for value in min_votes_values:
            model.min_votes = value
            result = model.get_top_n_recommendations_for_titles(titles, n=n)
            columns[f"min_votes={value:,}"] = result["primaryTitle"].reset_index(drop=True)
    finally:
        model.min_votes = original
    table = pd.DataFrame(columns).fillna("")
    table.index = range(1, len(table) + 1)
    print(f"Liked: {', '.join(titles)}")
    print(table.to_string())
    return table


def summary(model: ContentRecommender, names: Sequence[str], n: int = 10) -> pd.DataFrame:
    """One row per preset: expected-genre hit rate and mean similarity."""
    rows = []
    for name in names:
        preset = PRESETS[name]
        result = model.get_top_n_recommendations_for_titles(preset["titles"], n=n)
        rows.append(
            {
                "preset": name,
                "expected genre": preset["genre"] or "-",
                "hit rate": (
                    f"{genre_hit_rate(result, preset['genre']):.0%}"
                    if preset["genre"]
                    else "-"
                ),
                "mean similarity": f"{result['Similarity_Score'].mean():.4f}",
            }
        )
    table = pd.DataFrame(rows)
    print("\n=== Summary ===")
    print(table.to_string(index=False))
    return table


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("presets", nargs="*", help="Preset names (default: all).")
    parser.add_argument("--list", action="store_true", help="List presets and exit.")
    parser.add_argument("--user-id", help="Also run a MovieLens user's history.")
    parser.add_argument("-n", "--top-n", type=int, default=10, help="Recommendations per run.")
    parser.add_argument("--min-votes", type=int, default=100, help="IMDb vote cutoff.")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare the presets' recommendations across min_votes values.",
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    args = parser.parse_args(argv)

    if args.list:
        for name, preset in PRESETS.items():
            print(f"{name:<16} {preset['genre'] or '-':<10} {', '.join(preset['titles'])}")
        return 0
    unknown = [name for name in args.presets if name not in PRESETS]
    if unknown:
        print(f"Error: unknown preset(s): {', '.join(unknown)}. Use --list to see them.")
        return 2
    if args.top_n < 1 or args.min_votes < 0:
        print("Error: -n must be at least 1 and --min-votes cannot be negative.")
        return 2

    names = args.presets or ([] if args.user_id else list(PRESETS))
    try:
        model = load_model(args.data_path, args.min_votes)
        for name in names:
            if args.compare:
                print(f"\n=== {name}: min_votes comparison ===")
                compare(model, PRESETS[name]["titles"], n=args.top_n)
            else:
                run_preset(model, name, n=args.top_n)
        if args.user_id:
            run_user(model, args.user_id, n=args.top_n)
        if len(names) > 1 and not args.compare:
            summary(model, names, n=args.top_n)
    except (OSError, ValueError) as error:
        print(f"Error: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
