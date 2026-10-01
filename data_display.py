"""Display small samples from the recommender datasets.

Examples:
    python data_display.py
    python data_display.py --dataset movielens --rows 5
    python data_display.py --dataset imdb --rows 3
    python data_display.py --build-interactions
"""

import argparse
import csv
import gzip
from pathlib import Path
from typing import Iterable, TextIO

import pandas as pd


MOVIELENS_FILES = (
    "movies.csv",
    "ratings.csv",
    "tags.csv",
    "links.csv",
)
IMDB_FILES = (
    "title.basics.tsv.gz",
    "title.crew.tsv.gz",
    "title.ratings.tsv.gz",
)
FILE_DESCRIPTIONS = {
    "movies.csv": "Movie titles and genres",
    "ratings.csv": "User ratings and timestamps",
    "tags.csv": "User-generated movie tags",
    "links.csv": "MovieLens, IMDb, and TMDb IDs",
    "title.basics.tsv.gz": "IMDb titles, years, runtimes, and genres",
    "title.crew.tsv.gz": "IMDb directors and writers",
    "title.ratings.tsv.gz": "IMDb ratings and vote counts",
}
DATASET_DESCRIPTIONS = {
    "movielens": "MovieLens user activity and movie metadata",
    "imdb": "IMDb movie metadata and ratings",
}


def build_interaction_dataframe(data_dir: Path) -> pd.DataFrame:
    """Build a rating-level dataframe enriched with IMDb and tag features."""
    movielens_dir = data_dir / "movielens" / "ml-32m"
    imdb_basics_path = data_dir / "imdb" / "title.basics.tsv.gz"

    required_paths = (
        movielens_dir / "movies.csv",
        movielens_dir / "links.csv",
        movielens_dir / "ratings.csv",
        movielens_dir / "tags.csv",
        imdb_basics_path,
    )
    missing_paths = [path for path in required_paths if not path.is_file()]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Required dataset file(s) not found: {missing}")

    movies = pd.read_csv(movielens_dir / "movies.csv")
    links = pd.read_csv(movielens_dir / "links.csv", dtype={"imdbId": "string"})
    ratings = pd.read_csv(movielens_dir / "ratings.csv")
    tags = pd.read_csv(movielens_dir / "tags.csv")

    imdb_basics = pd.read_csv(
        imdb_basics_path,
        sep="\t",
        compression="gzip",
        usecols=["tconst", "primaryTitle", "startYear", "runtimeMinutes", "genres"],
        dtype="string",
        na_values=[r"\N"],
    ).rename(columns={"genres": "imdbGenres"})

    # MovieLens stores IMDb IDs without the IMDb prefix and leading zeroes.
    imdb_ids = pd.to_numeric(links["imdbId"], errors="coerce")
    links["tconst"] = imdb_ids.astype("Int64").map(
        lambda value: f"tt{value:07d}" if pd.notna(value) else pd.NA
    )

    item_features = (
        movies.rename(columns={"genres": "movieLensGenres"})
        .merge(links[["movieId", "tconst"]], on="movieId", how="inner", validate="one_to_one")
        .merge(imdb_basics, on="tconst", how="inner", validate="many_to_one")
    )

    grouped_tags = (
        tags.dropna(subset=["tag"])
        .groupby("movieId", as_index=False)["tag"]
        .agg(lambda values: ", ".join(str(value) for value in values))
        .rename(columns={"tag": "userTags"})
    )
    item_features = item_features.merge(grouped_tags, on="movieId", how="left")
    item_features["userTags"] = item_features["userTags"].fillna("")

    return ratings.merge(
        item_features,
        on="movieId",
        how="inner",
        validate="many_to_one",
    )


def open_data_file(path: Path) -> TextIO:
    """Open a plain-text or gzip-compressed data file as UTF-8 text."""
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8", newline="")
    return path.open(mode="r", encoding="utf-8", newline="")


def sample_rows(path: Path, row_limit: int) -> tuple[list[str], list[list[str]]]:
    """Read the header and a small number of rows without loading a file."""
    delimiter = "\t" if path.name.endswith(".tsv.gz") else ","
    with open_data_file(path) as data_file:
        reader = csv.reader(data_file, delimiter=delimiter)
        header = next(reader, [])
        rows = [row for _, row in zip(range(row_limit), reader)]
    return header, rows


def shorten(value: str, max_width: int = 32) -> str:
    """Keep wide fields from making the sample difficult to scan."""
    if len(value) <= max_width:
        return value
    return value[: max_width - 3] + "..."


def display_table(header: list[str], rows: list[list[str]]) -> None:
    """Print rows as an aligned table with bounded column widths."""
    display_rows = [[shorten(value) for value in row] for row in rows]
    widths = [len(column) for column in header]
    for row in display_rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    heading = "  #  " + " | ".join(
        column.ljust(width) for column, width in zip(header, widths)
    )
    divider = "  " + "-" * (len(heading) - 2)
    print(heading)
    print(divider)
    for row_number, row in enumerate(display_rows, start=1):
        values = " | ".join(value.ljust(width) for value, width in zip(row, widths))
        print(f"  {row_number:<2} {values}")


def display_file(path: Path, row_limit: int) -> None:
    """Print a file's metadata and sample rows."""
    if not path.exists():
        print(f"\n{path.name}")
        print("  File not found")
        return

    size_mb = path.stat().st_size / 1_048_576
    description = FILE_DESCRIPTIONS.get(path.name, "Dataset file")
    print(f"\n{path.name} - {description} ({size_mb:.1f} MB)")
    header, rows = sample_rows(path, row_limit)
    if not header:
        print("  Empty file")
        return

    print(f"  Columns ({len(header)}): {', '.join(header)}")
    if rows:
        display_table(header, rows)
    else:
        print("  No data rows found")


def files_for_dataset(data_dir: Path, dataset: str) -> Iterable[Path]:
    if dataset in ("all", "movielens"):
        yield from (data_dir / "movielens" / "ml-32m" / name for name in MOVIELENS_FILES)
    if dataset in ("all", "imdb"):
        yield from (data_dir / "imdb" / name for name in IMDB_FILES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "recommender_data",
        help="Root directory containing the movielens and imdb folders.",
    )
    parser.add_argument(
        "--dataset",
        choices=("all", "movielens", "imdb"),
        default="all",
        help="Dataset to display (default: all).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=5,
        help="Number of sample rows per file (default: 5).",
    )
    parser.add_argument(
        "--build-interactions",
        action="store_true",
        help="Build and export the rating-level MovieLens/IMDb interaction dataframe.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV path (default: <data-dir>/interactions.csv).",
    )
    args = parser.parse_args()

    if args.rows < 1:
        parser.error("--rows must be at least 1")
    if not args.data_dir.is_dir():
        parser.error(f"data directory does not exist: {args.data_dir}")

    if args.build_interactions:
        output_path = args.output or args.data_dir / "interactions.csv"
        interactions = build_interaction_dataframe(args.data_dir)
        interactions.to_csv(output_path, index=False)

        memory_mb = interactions.memory_usage(deep=True).sum() / 1_048_576
        print(f"Saved interaction dataframe to: {output_path.resolve()}")
        print(f"Final dataframe shape: {interactions.shape}")
        print(f"Memory usage: {memory_mb:.2f} MB")
        print("Head:")
        print(interactions.head())
        return

    print(f"Data directory: {args.data_dir.resolve()}")
    selected_datasets = ("movielens", "imdb") if args.dataset == "all" else (args.dataset,)
    for dataset in selected_datasets:
        print(f"\n{'=' * 72}\n{dataset.upper()} - {DATASET_DESCRIPTIONS[dataset]}\n{'=' * 72}")
        dataset_dir = args.data_dir / ("movielens/ml-32m" if dataset == "movielens" else "imdb")
        print(f"Location: {dataset_dir}")
        for path in files_for_dataset(args.data_dir, dataset):
            display_file(path, args.rows)


if __name__ == "__main__":
    main()