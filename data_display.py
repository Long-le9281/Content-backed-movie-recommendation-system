"""Display small samples from the recommender datasets.

Examples:
    python data_display.py
    python data_display.py --dataset movielens --rows 5
    python data_display.py --dataset imdb --rows 3
"""

import argparse
import csv
import gzip
from pathlib import Path
from typing import Iterable, TextIO


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
    args = parser.parse_args()

    if args.rows < 1:
        parser.error("--rows must be at least 1")
    if not args.data_dir.is_dir():
        parser.error(f"data directory does not exist: {args.data_dir}")

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