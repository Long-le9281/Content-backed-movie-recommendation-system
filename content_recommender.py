"""Content-based movie recommendations using TF-IDF features and KNN."""

from __future__ import annotations

import argparse
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
        "userId",
        "rating",
        "primaryTitle",
        "genres_imdb",
        "grouped_tags",
    }

    def __init__(
        self,
        data_path: str | Path | None = None,
        data: pd.DataFrame | None = None,
    ) -> None:
        if data_path is None and data is None:
            raise ValueError("Provide either data_path or data.")
        if data_path is not None and data is not None:
            raise ValueError("Provide data_path or data, not both.")

        self.data_path = Path(data_path) if data_path is not None else None
        self.data = data.copy() if data is not None else None
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
        if self.data.empty:
            raise ValueError("Dataset contains no rows.")

        self.data = self.data.copy()
        self.data["rating"] = pd.to_numeric(self.data["rating"], errors="coerce")
        self.data = self.data.dropna(subset=["movieId", "userId", "rating"])
        if self.data.empty:
            raise ValueError("Dataset contains no valid rating rows.")
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
        """Build one TF-IDF vector per movie from genres and tags.

        For movie *i*, the document is the normalized concatenation of its
        genres and tags. The resulting row vector is x_i, and
        ``movie_feature_matrix`` contains all x_i.
        """
        if self.data is None:
            self.load_data()
        assert self.data is not None

        movie_columns = ["movieId", "primaryTitle", "genres_imdb", "grouped_tags"]
        self.movie_data = self.data[movie_columns].copy()
        self.movie_data["genres_imdb"] = self.movie_data["genres_imdb"].fillna("")
        self.movie_data["grouped_tags"] = self.movie_data["grouped_tags"].fillna("")
        self.movie_data["text"] = (
            self.movie_data["genres_imdb"].astype(str).str.replace(",", " ", regex=False)
            + " "
            + self.movie_data["grouped_tags"].astype(str).str.replace(",", " ", regex=False)
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
        weights = (liked["rating"].to_numpy(dtype=float) - 3.0).reshape(-1, 1)
        profile = self.movie_feature_matrix[positions].multiply(weights).sum(axis=0)
        return csr_matrix(profile / weights.sum())

    def get_top_n_recommendations(
        self, user_id: Any, n: int = 10
    ) -> pd.DataFrame:
        """Return the nearest unseen movies and cosine similarity scores."""
        if n < 1:
            raise ValueError("n must be at least 1.")
        if self.movie_data is not None and n > len(self.movie_data):
            raise ValueError("n cannot exceed the number of movies.")
        if self.knn_model is None:
            self.fit_knn()
        assert self.knn_model is not None
        assert self.movie_data is not None

        profile = self.generate_user_profile(user_id)
        distances, indices = self.knn_model.kneighbors(
            profile, n_neighbors=n
        )
        user_id = self._coerce_user_id(user_id)
        seen_movie_ids = set(self.data.loc[self.data["userId"] == user_id, "movieId"])
        result = self.movie_data.iloc[indices[0]].copy()
        result["Similarity_Score"] = 1.0 - distances[0]
        result = result[~result["movieId"].isin(seen_movie_ids)]
        return result[["primaryTitle", "genres_imdb", "Similarity_Score"]].head(n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", type=Path, help="Merged rating-level CSV path.")
    parser.add_argument("user_id", help="User ID to recommend movies for.")
    parser.add_argument("-n", type=int, default=10, help="Number of recommendations.")
    args = parser.parse_args()

    recommender = ContentRecommender(data_path=args.data_path)
    recommender.load_data()
    recommendations = recommender.get_top_n_recommendations(args.user_id, n=args.n)
    print(recommendations.to_string(index=False))


if __name__ == "__main__":
    main()
