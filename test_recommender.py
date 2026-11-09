"""Focused tests for interaction building and content-based recommendations."""

from __future__ import annotations

import gzip
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

from content_recommender import ContentRecommender
from data_display import build_interaction_dataframe
from recommender_cli import main as recommender_cli_main


class ContentRecommenderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = pd.DataFrame(
            [
                (1, 7, 5.0, "The Shining", "Horror,Drama", "hotel,psychological", 10000, "nm_director_a", "nm_writer_a", "1980"),
                (2, 7, 5.0, "The Lighthouse", "Drama,Horror", "lighthouse,psychological", 300000, "nm_director_b", "nm_writer_b", "2019"),
                (3, 7, 5.0, "The Lighthouse", "Drama", "silent,comedy", 150, "nm_director_c", "nm_writer_c", "1937"),
                (4, 7, 5.0, "American Psycho", "Horror,Crime", "identity,psychological", 90000, "nm_director_a", "nm_writer_d", "2000"),
                (5, 8, 0.0, "Candidate", "Horror,Drama", "psychological", 500, "nm_director_a", "nm_writer_a", "2020"),
                (6, 8, 0.0, "Low Vote Candidate", "Horror,Drama", "psychological", 20, "nm_director_a", "nm_writer_a", "2021"),
                (7, 8, 0.0, "Other Candidate", "Drama,Crime", "identity", 200, "nm_director_d", "nm_writer_d", "2022"),
            ],
            columns=[
                "movieId",
                "userId",
                "rating",
                "primaryTitle",
                "genres_imdb",
                "grouped_tags",
                "numVotes",
                "directors",
                "writers",
                "startYear",
            ],
        )

    def test_features_include_directors_and_writers(self) -> None:
        recommender = ContentRecommender(data=self.data)
        recommender.build_item_embeddings()

        assert recommender.vectorizer is not None
        self.assertIn("director_nm_director_a", recommender.vectorizer.vocabulary_)
        self.assertIn("writer_nm_writer_d", recommender.vectorizer.vocabulary_)

    def test_filters_low_vote_items_and_returns_requested_unseen_count(self) -> None:
        recommender = ContentRecommender(data=self.data, min_votes=100)

        recommendations = recommender.get_top_n_recommendations(7, n=2)

        self.assertEqual(len(recommendations), 2)
        self.assertNotIn("Low Vote Candidate", recommendations["primaryTitle"].tolist())
        self.assertNotIn("The Shining", recommendations["primaryTitle"].tolist())

    def test_title_profile_uses_two_liked_titles_and_excludes_them(self) -> None:
        recommender = ContentRecommender(data=self.data, min_votes=100)

        recommendations = recommender.get_top_n_recommendations_for_titles(
            ["The Shining", "American Psycho"], n=2
        )

        self.assertEqual(len(recommendations), 2)
        self.assertIn("Candidate", recommendations["primaryTitle"].tolist())
        self.assertNotIn("Low Vote Candidate", recommendations["primaryTitle"].tolist())
        self.assertTrue(
            {"The Shining", "American Psycho"}.isdisjoint(
                recommendations["primaryTitle"].tolist()
            )
        )

    def test_duplicate_title_defaults_to_highest_vote_match(self) -> None:
        recommender = ContentRecommender(data=self.data, min_votes=100)
        recommender.build_item_embeddings()

        selected_movie = recommender.movie_data.iloc[
            recommender._find_title_position("The Lighthouse")
        ]

        self.assertEqual(selected_movie["movieId"], 2)

    def test_year_qualifies_duplicate_title(self) -> None:
        recommender = ContentRecommender(data=self.data, min_votes=100)
        recommender.build_item_embeddings()

        self.assertEqual(recommender._find_title_position("The Lighthouse (1937)"), 2)

    def test_missing_vote_counts_are_not_recommended(self) -> None:
        data = self.data.copy()
        data["numVotes"] = pd.array(data["numVotes"], dtype="Int64")
        data.loc[data["movieId"] == 7, "numVotes"] = pd.NA
        recommender = ContentRecommender(data=data, min_votes=100)

        recommendations = recommender.get_top_n_recommendations(7, n=5)

        self.assertNotIn("Other Candidate", recommendations["primaryTitle"].tolist())

    def test_empty_seed_features_are_reported(self) -> None:
        data = self.data.copy()
        data.loc[data["movieId"] == 1, ["genres_imdb", "grouped_tags", "directors", "writers"]] = ""
        recommender = ContentRecommender(data=data)

        with self.assertRaisesRegex(ValueError, "no usable content features"):
            recommender.get_top_n_recommendations_for_titles(["The Shining"])

    def test_cli_accepts_catalog_without_ratings_for_title_queries(self) -> None:
        catalog = self.data.drop(columns=["userId", "rating"]).rename(
            columns={"genres_imdb": "imdbGenres", "grouped_tags": "userTags"}
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "movie_features.csv"
            catalog.to_csv(path, index=False)
            output = io.StringIO()

            with redirect_stdout(output):
                result = recommender_cli_main(
                    [
                        str(path),
                        "--like",
                        "The Shining",
                        "--like",
                        "American Psycho",
                        "-n",
                        "2",
                    ]
                )

        self.assertEqual(result, 0)
        self.assertIn("Recommended Movie", output.getvalue())
        self.assertIn("Candidate", output.getvalue())


class InteractionBuilderTests(unittest.TestCase):
    def test_joins_creator_vote_and_normalized_tag_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            movielens_dir = data_dir / "movielens" / "ml-32m"
            imdb_dir = data_dir / "imdb"
            movielens_dir.mkdir(parents=True)
            imdb_dir.mkdir()
            pd.DataFrame(
                [(10, "Sample Movie (2020)", "Drama|Thriller")],
                columns=["movieId", "title", "genres"],
            ).to_csv(movielens_dir / "movies.csv", index=False)
            pd.DataFrame(
                [(10, "1", 100)],
                columns=["movieId", "imdbId", "tmdbId"],
            ).to_csv(movielens_dir / "links.csv", index=False)
            pd.DataFrame(
                [(7, 10, 4.5, 123)],
                columns=["userId", "movieId", "rating", "timestamp"],
            ).to_csv(movielens_dir / "ratings.csv", index=False)
            pd.DataFrame(
                [(7, 10, "mystery", 123)],
                columns=["userId", "movieId", "tag", "timestamp"],
            ).to_csv(movielens_dir / "tags.csv", index=False)

            self._write_tsv_gzip(
                imdb_dir / "title.basics.tsv.gz",
                "tconst\ttitleType\tprimaryTitle\tstartYear\truntimeMinutes\tgenres\n"
                "tt0000001\tmovie\tSample Movie\t2020\t100\tDrama,Thriller\n",
            )
            self._write_tsv_gzip(
                imdb_dir / "title.crew.tsv.gz",
                "tconst\tdirectors\twriters\n"
                "tt0000001\tnm0000001\tnm0000002\n",
            )
            self._write_tsv_gzip(
                imdb_dir / "title.ratings.tsv.gz",
                "tconst\taverageRating\tnumVotes\n"
                "tt0000001\t7.5\t2500\n",
            )

            interactions = build_interaction_dataframe(data_dir)

        row = interactions.iloc[0]
        self.assertEqual(row["genres_imdb"], "Drama,Thriller")
        self.assertEqual(row["grouped_tags"], "mystery")
        self.assertEqual(row["directors"], "nm0000001")
        self.assertEqual(row["writers"], "nm0000002")
        self.assertEqual(row["numVotes"], 2500)

    @staticmethod
    def _write_tsv_gzip(path: Path, contents: str) -> None:
        with gzip.open(path, "wt", encoding="utf-8") as output_file:
            output_file.write(contents)


if __name__ == "__main__":
    unittest.main()
