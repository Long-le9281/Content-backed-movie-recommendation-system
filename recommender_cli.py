"""Generate a mock profile and query placeholder movie recommendations.

Examples:
    python recommender_cli.py --generate-mock-user
    python recommender_cli.py --user-id 12345 --top-n 10
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_USER_ID = 12345
DEFAULT_PROFILE_PATH = Path("test_user_profile.csv")


@dataclass(frozen=True)
class Recommendation:
    """Represent one placeholder movie recommendation."""

    title: str
    genres: str
    similarity: float


MOCK_MOVIES: tuple[tuple[int, float], ...] = (
    (1, 4.5),      # Toy Story
    (296, 5.0),    # Pulp Fiction
    (318, 5.0),    # The Shawshank Redemption
    (527, 4.5),    # Schindler's List
    (593, 4.0),    # The Silence of the Lambs
    (858, 4.5),    # The Godfather
    (1196, 4.0),   # Star Wars: Episode V
    (1221, 5.0),   # The Godfather: Part II
    (2571, 4.5),   # The Matrix
    (2762, 4.0),   # The Sixth Sense
    (2959, 4.5),   # Fight Club
    (4306, 3.5),   # Shrek
)

PLACEHOLDER_RECOMMENDATIONS: tuple[Recommendation, ...] = (
    Recommendation("Goodfellas", "Crime, Drama", 0.9412),
    Recommendation("The Dark Knight", "Action, Crime, Drama", 0.9237),
    Recommendation("Inception", "Action, Sci-Fi, Thriller", 0.9124),
    Recommendation("Forrest Gump", "Drama, Romance", 0.8876),
    Recommendation("The Lord of the Rings", "Adventure, Fantasy", 0.8741),
    Recommendation("The Prestige", "Drama, Mystery, Sci-Fi", 0.8619),
    Recommendation("Gladiator", "Action, Adventure, Drama", 0.8495),
    Recommendation("The Departed", "Crime, Drama, Thriller", 0.8382),
    Recommendation("Jurassic Park", "Action, Adventure, Sci-Fi", 0.8248),
    Recommendation("Back to the Future", "Adventure, Comedy, Sci-Fi", 0.8136),
    Recommendation("Alien", "Horror, Sci-Fi", 0.8027),
    Recommendation("The Green Mile", "Crime, Drama, Fantasy", 0.7914),
)


def generate_mock_user_data(
    output_path: str | Path = DEFAULT_PROFILE_PATH,
    user_id: int = DEFAULT_USER_ID,
) -> Path:
    """Create and save a realistic MovieLens-style test user profile.

    Args:
        output_path: Destination CSV path.
        user_id: User ID assigned to every generated rating.

    Returns:
        The resolved path of the generated CSV file.

    Raises:
        OSError: If the destination cannot be created or written.
        ValueError: If ``user_id`` is not a positive integer.
    """
    if user_id < 1:
        raise ValueError("user_id must be a positive integer.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as profile_file:
        writer = csv.writer(profile_file)
        writer.writerow(["userId", "movieId", "rating"])
        writer.writerows((user_id, movie_id, rating) for movie_id, rating in MOCK_MOVIES)
    return path.resolve()


class ContentRecommender:
    """Placeholder recommender that reads user ratings and returns dummy results.

    Replace ``get_top_n_recommendations`` with the TF-IDF/KNN implementation
    from ``content_recommender.py`` when the production item catalogue is
    available.
    """

    REQUIRED_COLUMNS = {"userId", "movieId", "rating"}

    def __init__(self, profile_path: str | Path) -> None:
        """Initialize the recommender and load the profile CSV.

        Args:
            profile_path: CSV containing ``userId``, ``movieId``, and ``rating``.

        Raises:
            FileNotFoundError: If the profile CSV does not exist.
            ValueError: If the CSV schema or data is invalid.
        """
        self.profile_path = Path(profile_path)
        self.user_ratings = self._load_profile()

    def _load_profile(self) -> list[dict[str, str]]:
        """Load and validate the profile rows."""
        if not self.profile_path.is_file():
            raise FileNotFoundError(f"Profile CSV not found: {self.profile_path}")

        with self.profile_path.open(newline="", encoding="utf-8") as profile_file:
            reader = csv.DictReader(profile_file)
            if reader.fieldnames is None:
                raise ValueError("Profile CSV has no header.")
            missing = self.REQUIRED_COLUMNS.difference(reader.fieldnames)
            if missing:
                raise ValueError(
                    "Profile CSV is missing column(s): " + ", ".join(sorted(missing))
                )
            rows = list(reader)

        if not rows:
            raise ValueError("Profile CSV contains no ratings.")
        for row_number, row in enumerate(rows, start=2):
            try:
                int(row["userId"])
                int(row["movieId"])
                rating = float(row["rating"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid rating data on CSV line {row_number}.") from exc
            if not 0.0 <= rating <= 5.0:
                raise ValueError(f"Rating on CSV line {row_number} must be between 0 and 5.")
        return rows

    def get_top_n_recommendations(
        self, user_id: int, top_n: int = 10
    ) -> list[Recommendation]:
        """Return placeholder recommendations for a user.

        Args:
            user_id: User whose profile should be queried.
            top_n: Maximum number of recommendations to return.

        Returns:
            Dummy recommendations ordered by descending similarity.

        Raises:
            ValueError: If the user is absent or ``top_n`` is invalid.
        """
        if top_n < 1:
            raise ValueError("top_n must be at least 1.")
        if not any(int(row["userId"]) == user_id for row in self.user_ratings):
            raise ValueError(f"User {user_id} was not found in {self.profile_path}.")
        return list(PLACEHOLDER_RECOMMENDATIONS[:top_n])


def print_recommendations(recommendations: Sequence[Recommendation]) -> None:
    """Print recommendations as a clean fixed-width terminal table."""
    headers = (
        "Recommended Movie Title",
        "IMDb Genres",
        "Cosine Similarity Score",
    )
    rows = [
        (item.title, item.genres, f"{item.similarity:.4f}")
        for item in recommendations
    ]
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    separator = "-+-".join("-" * width for width in widths)
    print(" | ".join(header.ljust(width) for header, width in zip(headers, widths)))
    print(separator)
    for row in rows:
        print(" | ".join(value.ljust(width) for value, width in zip(row, widths)))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", type=int, default=DEFAULT_USER_ID)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument(
        "--generate-mock-user",
        action="store_true",
        help="Generate the mock profile before querying recommendations.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    args = parse_args(argv)
    try:
        if args.generate_mock_user or not args.profile.exists():
            generated_path = generate_mock_user_data(args.profile, args.user_id)
            if args.generate_mock_user:
                print(f"Generated mock profile: {generated_path}")
        recommender = ContentRecommender(args.profile)
        recommendations = recommender.get_top_n_recommendations(
            args.user_id, args.top_n
        )
        print_recommendations(recommendations)
    except (OSError, ValueError) as error:
        print(f"Error: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
