"""Content-based movie recommendations using TF-IDF features and KNN."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


class ContentRecommender:
    """Recommend unseen movies from a user's highly rated movie content."""

    REQUIRED_COLUMNS = {
        "movieId",
        "primaryTitle",
        "genres_imdb",
        "grouped_tags",
        "numVotes",
    }
    RATING_COLUMNS = {"userId", "rating"}

    def __init__(
        self,
        data_path: str | Path | None = None,
        data: pd.DataFrame | None = None,
        min_votes: int = 100,
    ) -> None:
        if data_path is None and data is None:
            raise ValueError("Provide either data_path or data.")
        if data_path is not None and data is not None:
            raise ValueError("Provide data_path or data, not both.")
        if min_votes < 0:
            raise ValueError("min_votes cannot be negative.")

        self.data_path = Path(data_path) if data_path is not None else None
        self.data = data.copy() if data is not None else None
        self.min_votes = min_votes
        self.movie_data: pd.DataFrame | None = None
        self.movie_feature_matrix: csr_matrix | None = None
        self.vectorizer: TfidfVectorizer | None = None
        self.knn_model: NearestNeighbors | None = None

    def load_data(self, data_path: str | Path | None = None) -> pd.DataFrame:
        """Load and validate the rating-level merged dataset."""
        path = Path(data_path) if data_path is not None else self.data_path
        if self.data is None:
            if path is None:
                raise ValueError("No dataset path was provided.")
            if not path.is_file():
                raise FileNotFoundError(f"Dataset not found: {path}")
            self.data = pd.read_csv(path)

        missing = self.REQUIRED_COLUMNS.difference(self.data.columns)
        if missing:
            columns = ", ".join(sorted(missing))
            raise ValueError(f"Dataset is missing required column(s): {columns}")
        present_rating_columns = self.RATING_COLUMNS.intersection(self.data.columns)
        if present_rating_columns and present_rating_columns != self.RATING_COLUMNS:
            missing_ratings = self.RATING_COLUMNS.difference(self.data.columns)
            columns = ", ".join(sorted(missing_ratings))
            raise ValueError(
                f"Dataset must contain both rating columns or neither; missing: {columns}"
            )
        if self.data.empty:
            raise ValueError("Dataset contains no rows.")

        self.data = self.data.copy()
        self.data["numVotes"] = pd.to_numeric(self.data["numVotes"], errors="coerce")
        valid_columns = ["movieId"]
        if present_rating_columns:
            self.data["rating"] = pd.to_numeric(self.data["rating"], errors="coerce")
            valid_columns.extend(self.RATING_COLUMNS)
        self.data = self.data.dropna(subset=valid_columns)
        if self.data.empty:
            raise ValueError("Dataset contains no valid movie rows.")
        return self.data

    def _coerce_user_id(self, user_id: Any) -> Any:
        """Match CLI string IDs to numeric IDs inferred by pandas."""
        assert self.data is not None
        user_dtype = self.data["userId"].dtype
        if pd.api.types.is_numeric_dtype(user_dtype):
            try:
                return pd.to_numeric(user_id)
            except (TypeError, ValueError):
                return user_id
        return user_id

    def build_item_embeddings(self) -> csr_matrix:
        """Build one TF-IDF vector per movie from genres, tags, and crew.

        For movie *i*, the document is the normalized concatenation of its
        genres, tags, directors, and writers. The resulting row vector is x_i, and
        ``movie_feature_matrix`` contains all x_i.
        """
        if self.data is None:
            self.load_data()
        assert self.data is not None

        movie_columns = [
            "movieId",
            "primaryTitle",
            "genres_imdb",
            "grouped_tags",
            "numVotes",
        ]
        movie_columns.extend(
            column
            for column in ("directors", "writers", "startYear", "movieLensTitle")
            if column in self.data.columns
        )
        self.movie_data = self.data[movie_columns].copy()
        text_columns = ["genres_imdb", "grouped_tags"]
        for column in ("directors", "writers"):
            if column in self.movie_data.columns:
                text_columns.append(column)
        self.movie_data[text_columns] = self.movie_data[text_columns].fillna("")
        director_tokens = ""
        if "directors" in self.movie_data.columns:
            director_tokens = self.movie_data["directors"].map(
                lambda value: " ".join(
                    f"director_{identifier}"
                    for identifier in str(value).split(",")
                    if identifier
                )
            )
        writer_tokens = ""
        if "writers" in self.movie_data.columns:
            writer_tokens = self.movie_data["writers"].map(
                lambda value: " ".join(
                    f"writer_{identifier}"
                    for identifier in str(value).split(",")
                    if identifier
                )
            )
        self.movie_data["text"] = (
            self.movie_data["genres_imdb"].astype(str).str.replace(",", " ", regex=False)
            + " "
            + self.movie_data["grouped_tags"].astype(str).str.replace(",", " ", regex=False)
            + " "
            + director_tokens
            + " "
            + writer_tokens
        ).str.strip()
        self.movie_data = self.movie_data.drop_duplicates("movieId").reset_index(drop=True)

        self.vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[\w-]+\b")
        self.movie_feature_matrix = self.vectorizer.fit_transform(self.movie_data["text"])
        return self.movie_feature_matrix

    def fit_knn(self) -> NearestNeighbors:
        """Fit cosine-distance KNN on the item feature matrix."""
        if self.movie_feature_matrix is None:
            self.build_item_embeddings()
        assert self.movie_feature_matrix is not None
        self.knn_model = NearestNeighbors(metric="cosine")
        self.knn_model.fit(self.movie_feature_matrix)
        return self.knn_model

    def generate_user_profile(self, user_id: Any) -> csr_matrix:
        """Return the rating-weighted mean vector for a user's ratings >= 4.0.

        With item vectors x_i and ratings r_i, the profile is
        p = sum((r_i - 3) * x_i) / sum(r_i - 3), for r_i >= 4.
        Subtracting 3 makes a 4-star rating contribute weight 1 and a
        5-star rating contribute weight 2.
        """
        if self.data is None:
            self.load_data()
        if not self.RATING_COLUMNS.issubset(self.data.columns):
            raise ValueError(
                "User-based recommendations require a dataset with userId and rating columns."
            )
        if self.movie_data is None or self.movie_feature_matrix is None:
            self.build_item_embeddings()
        assert self.data is not None
        assert self.movie_data is not None
        assert self.movie_feature_matrix is not None

        user_id = self._coerce_user_id(user_id)
        user_ratings = self.data[self.data["userId"] == user_id]
        if user_ratings.empty:
            raise ValueError(f"User {user_id!r} was not found in the dataset.")
        liked = user_ratings[user_ratings["rating"] >= 4.0]
        if liked.empty:
            raise ValueError(f"User {user_id!r} has no ratings of 4.0 or higher.")

        movie_positions = self.movie_data.reset_index().set_index("movieId")["index"]
        liked = liked[liked["movieId"].isin(movie_positions.index)]
        if liked.empty:
            raise ValueError(f"User {user_id!r} has no rated movies with content features.")

        positions = movie_positions.loc[liked["movieId"]].to_numpy()
        weights = liked["rating"].to_numpy(dtype=float) - 3.0
        return self._build_profile(positions, weights)

    def _ensure_embeddings(self) -> None:
        """Load data and build item vectors if that has not happened yet."""
        if self.data is None:
            self.load_data()
        if self.movie_data is None or self.movie_feature_matrix is None:
            self.build_item_embeddings()

    def _build_profile(
        self, positions: Any, weights: Any | None = None
    ) -> csr_matrix:
        """Average item vectors x_i at ``positions`` into one profile vector.

        Without weights this is the plain mean p = (1/k) * sum(x_i). With
        weights w_i it is the weighted mean p = sum(w_i * x_i) / sum(w_i).
        """
        assert self.movie_feature_matrix is not None
        positions = list(positions)
        if not positions:
            raise ValueError("A profile needs at least one movie.")
        rows = self.movie_feature_matrix[positions]
        if weights is None:
            return csr_matrix(rows.mean(axis=0))
        weights = pd.Series(weights, dtype=float).to_numpy().reshape(-1, 1)
        if len(weights) != len(positions):
            raise ValueError("Provide exactly one weight per movie.")
        if weights.sum() <= 0:
            raise ValueError("Profile weights must sum to a positive value.")
        return csr_matrix(rows.multiply(weights).sum(axis=0) / weights.sum())

    def build_profile_from_titles(self, titles: list[str]) -> csr_matrix:
        """Return the plain mean vector of the liked titles (duplicates once)."""
        self._ensure_embeddings()
        return self._build_profile(self._resolve_title_positions(titles))

    def _resolve_title_positions(self, titles: list[str]) -> list[int]:
        if not titles:
            raise ValueError("Provide at least one movie title.")
        positions = [self._find_title_position(title) for title in titles]
        return list(dict.fromkeys(positions))

    def build_profile_from_movie_ids(
        self, movie_ids: list[Any], ratings: list[float] | None = None
    ) -> csr_matrix:
        """Build a profile from movie IDs, optionally weighted by ratings.

        Without ratings, every movie counts equally (plain mean). With
        ratings, the same rule as ``generate_user_profile`` applies: only
        ratings >= 4.0 are used, with weight ``rating - 3``. If an ID appears
        more than once, its last rating is used.
        """
        self._ensure_embeddings()
        assert self.movie_data is not None
        if not movie_ids:
            raise ValueError("Provide at least one movie ID.")
        if ratings is not None and len(ratings) != len(movie_ids):
            raise ValueError("Provide exactly one rating per movie ID.")

        movie_positions = self.movie_data.reset_index().set_index("movieId")["index"]
        missing = [movie_id for movie_id in movie_ids if movie_id not in movie_positions.index]
        if missing:
            shown = ", ".join(str(movie_id) for movie_id in missing[:10])
            raise ValueError(f"Movie ID(s) not found in the dataset: {shown}")

        if ratings is None:
            unique_ids = list(dict.fromkeys(movie_ids))
            return self._build_profile(movie_positions.loc[unique_ids].to_numpy())

        rating_by_id = dict(zip(movie_ids, (float(rating) for rating in ratings)))
        liked = {movie_id: rating for movie_id, rating in rating_by_id.items() if rating >= 4.0}
        if not liked:
            raise ValueError("No ratings of 4.0 or higher were provided.")
        positions = movie_positions.loc[list(liked)].to_numpy()
        weights = [rating - 3.0 for rating in liked.values()]
        return self._build_profile(positions, weights)

    def top_profile_terms(self, profile: csr_matrix, k: int = 10) -> pd.DataFrame:
        """Return the ``k`` highest-weighted TF-IDF terms in a profile."""
        if k < 1:
            raise ValueError("k must be at least 1.")
        assert self.vectorizer is not None
        terms = self.vectorizer.get_feature_names_out()
        dense = profile.toarray().ravel()
        order = dense.argsort()[::-1][:k]
        order = [index for index in order if dense[index] > 0]
        return pd.DataFrame({"term": terms[order], "weight": dense[order]})

    def get_top_n_recommendations_for_titles(
        self, titles: list[str], n: int = 10
    ) -> pd.DataFrame:
        """Recommend movies from a set of liked titles, each treated as 5 stars."""
        if n < 1:
            raise ValueError("n must be at least 1.")
        self._ensure_embeddings()
        assert self.movie_data is not None

        positions = self._resolve_title_positions(titles)
        profile = self._build_profile(positions)
        seen_movie_ids = set(self.movie_data.iloc[positions]["movieId"])
        return self._recommend_from_profile(profile, seen_movie_ids, n)

    def get_top_n_recommendations_for_movie_ids(
        self,
        movie_ids: list[Any],
        ratings: list[float] | None = None,
        n: int = 10,
    ) -> pd.DataFrame:
        """Recommend from movie IDs; every listed movie counts as seen."""
        if n < 1:
            raise ValueError("n must be at least 1.")
        profile = self.build_profile_from_movie_ids(movie_ids, ratings)
        return self._recommend_from_profile(profile, set(movie_ids), n)

    def _find_title_position(self, title: str) -> int:
        """Resolve a title, using an optional year or IMDb votes to break ties."""
        assert self.movie_data is not None
        if not isinstance(title, str) or not title.strip():
            raise ValueError("Movie titles must be non-empty strings.")

        requested_title = title.strip()
        year_match = re.fullmatch(r"(.+?)\s+\((\d{4})\)", requested_title)
        normalized_title = (
            year_match.group(1) if year_match else requested_title
        ).casefold()
        matches = self.movie_data[
            self.movie_data["primaryTitle"].astype(str).str.casefold()
            == normalized_title
        ]
        if year_match and "startYear" not in self.movie_data.columns:
            raise ValueError(
                f"Cannot resolve release year for {title!r}: startYear is missing."
            )
        if year_match:
            matches = matches[
                pd.to_numeric(matches["startYear"], errors="coerce")
                == int(year_match.group(2))
            ]
        if matches.empty:
            raise ValueError(f"Movie title {title!r} was not found in the dataset.")
        if len(matches) > 1:
            matches = matches.sort_values("numVotes", ascending=False, na_position="last")
        return int(matches.index[0])

    def get_top_n_recommendations(
        self, user_id: Any, n: int = 10
    ) -> pd.DataFrame:
        """Return the nearest unseen movies and cosine similarity scores."""
        if n < 1:
            raise ValueError("n must be at least 1.")
        if self.knn_model is None:
            self.fit_knn()
        assert self.knn_model is not None
        assert self.movie_data is not None

        profile = self.generate_user_profile(user_id)
        assert self.data is not None
        user_id = self._coerce_user_id(user_id)
        seen_movie_ids = set(self.data.loc[self.data["userId"] == user_id, "movieId"])
        return self._recommend_from_profile(profile, seen_movie_ids, n)

    def _recommend_from_profile(
        self, profile: csr_matrix, seen_movie_ids: set[Any], n: int
    ) -> pd.DataFrame:
        if self.knn_model is None:
            self.fit_knn()
        assert self.knn_model is not None
        assert self.movie_data is not None

        if profile.nnz == 0:
            raise ValueError("The liked movies have no usable content features.")
        eligible = self.movie_data["numVotes"].ge(self.min_votes).fillna(False)
        available = self.movie_data.loc[
            eligible & ~self.movie_data["movieId"].isin(seen_movie_ids)
        ]
        if available.empty:
            return pd.DataFrame(
                columns=["primaryTitle", "genres_imdb", "Similarity_Score"]
            )

        excluded_count = len(self.movie_data) - len(available)
        neighbor_count = min(len(self.movie_data), n + excluded_count)
        distances, indices = self.knn_model.kneighbors(
            profile, n_neighbors=neighbor_count
        )
        result = self.movie_data.iloc[indices[0]].copy()
        result["Similarity_Score"] = 1.0 - distances[0]
        result = result[
            result["numVotes"].ge(self.min_votes).fillna(False)
            & ~result["movieId"].isin(seen_movie_ids)
        ]
        return result[["primaryTitle", "genres_imdb", "Similarity_Score"]].head(n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", type=Path, help="Merged rating-level CSV path.")
    parser.add_argument("user_id", help="User ID to recommend movies for.")
    parser.add_argument("-n", type=int, default=10, help="Number of recommendations.")
    parser.add_argument(
        "--min-votes",
        type=int,
        default=100,
        help="Exclude movies with fewer IMDb votes (default: 100).",
    )
    args = parser.parse_args()
    if args.n < 1:
        parser.error("-n must be at least 1.")
    if args.min_votes < 0:
        parser.error("--min-votes cannot be negative.")

    recommender = ContentRecommender(
        data_path=args.data_path, min_votes=args.min_votes
    )
    recommender.load_data()
    recommendations = recommender.get_top_n_recommendations(args.user_id, n=args.n)
    print(recommendations.to_string(index=False))


if __name__ == "__main__":
    main()
