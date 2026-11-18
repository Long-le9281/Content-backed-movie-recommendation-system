"""Tests for the held-out evaluation helpers in evaluate.py."""

from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from evaluate import (
    Variant,
    build_model,
    build_split,
    evaluate,
    intra_list_diversity,
    popularity_counts,
    ranking_metrics,
    recommend_content,
    recommend_popularity,
    sample_users,
)


def make_catalog() -> pd.DataFrame:
    rows = [
        (1, "Horror A", "Horror,Drama", "scary,psychological", 1000, 7.5),
        (2, "Horror B", "Horror", "scary,ghost", 2000, 7.0),
        (3, "Horror C", "Horror,Thriller", "psychological,ghost", 3000, 6.5),
        (4, "Romance A", "Romance,Comedy", "love,wedding", 4000, 6.0),
        (5, "Romance B", "Romance", "love,paris", 5000, 7.2),
        (6, "Obscure Horror", "Horror", "scary", 10, 5.0),
    ]
    return pd.DataFrame(
        rows,
        columns=["movieId", "primaryTitle", "genres_imdb", "grouped_tags", "numVotes", "averageRating"],
    )


class MetricTests(unittest.TestCase):
    def test_ranking_metrics_hand_computed(self) -> None:
        metrics = ranking_metrics([10, 20, 30, 40], relevant={20, 40, 99}, k=4)
        self.assertEqual(metrics["hit_rate"], 1.0)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 2 / 3)
        self.assertAlmostEqual(metrics["mrr"], 0.5)
        dcg = 1 / math.log2(3) + 1 / math.log2(5)
        idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
        self.assertAlmostEqual(metrics["ndcg"], dcg / idcg)

    def test_ranking_metrics_only_counts_top_k(self) -> None:
        metrics = ranking_metrics([1, 2, 3], relevant={3}, k=2)
        self.assertEqual(metrics["hit_rate"], 0.0)
        self.assertEqual(metrics["ndcg"], 0.0)
        self.assertEqual(metrics["mrr"], 0.0)

    def test_ranking_metrics_rejects_empty_relevant(self) -> None:
        with self.assertRaises(ValueError):
            ranking_metrics([1], relevant=set(), k=1)

    def test_intra_list_diversity(self) -> None:
        genres = {1: frozenset({"Horror"}), 2: frozenset({"Horror"}), 3: frozenset({"Romance"})}
        self.assertEqual(intra_list_diversity([1, 2], genres), 0.0)
        self.assertAlmostEqual(intra_list_diversity([1, 2, 3], genres), 2 / 3)


class SplitTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        rows = []
        for user_id in range(1, 6):
            for movie_id in range(1, 26):
                rows.append((user_id, movie_id, float(rng.choice([2.0, 4.0, 5.0]))))
        self.ratings = pd.DataFrame(rows, columns=["userId", "movieId", "rating"])

    def test_split_is_20_percent_and_reproducible(self) -> None:
        first = build_split(self.ratings, np.array([1, 2]), test_frac=0.2, seed=7)
        second = build_split(self.ratings, np.array([2, 1, 3]), test_frac=0.2, seed=7)
        for user_id in (1, 2):
            user = first[first["userId"] == user_id]
            self.assertEqual((user["split"] == "test").sum(), 5)
            pd.testing.assert_frame_equal(
                user.reset_index(drop=True),
                second[second["userId"] == user_id].reset_index(drop=True),
            )

    def test_sample_users_respects_thresholds(self) -> None:
        small = self.ratings[~((self.ratings["userId"] == 3) & (self.ratings["movieId"] > 5))]
        users = sample_users(small, n_users=10, min_ratings=20, min_liked=1, seed=0)
        self.assertNotIn(3, users)
        self.assertEqual(len(users), 4)

    def test_popularity_excludes_test_rows(self) -> None:
        split = build_split(self.ratings, np.array([1]), test_frac=0.2, seed=7)
        counts = popularity_counts(self.ratings, split)
        test_movie = int(split.loc[split["split"] == "test", "movieId"].iloc[0])
        self.assertEqual(counts[test_movie], 4)

    def test_recommend_popularity_skips_training_movies(self) -> None:
        self.assertEqual(recommend_popularity(np.array([5, 4, 3, 2]), {4}, 2), [5, 3])


class ContentEvaluationTests(unittest.TestCase):
    def test_hidden_movies_can_be_recommended(self) -> None:
        catalog = make_catalog()
        model = build_model(catalog, Variant("content"), base_min_votes=100)
        train = pd.DataFrame({"movieId": [1, 2, 4], "rating": [5.0, 5.0, 1.0]})
        recommended = recommend_content(model, train, "rating", n=3)
        self.assertEqual(recommended[0], 3)  # the hidden horror film ranks first
        self.assertNotIn(1, recommended)  # training movies are excluded
        self.assertNotIn(6, recommended)  # below min_votes

    def test_evaluate_end_to_end(self) -> None:
        catalog = make_catalog()
        split = pd.DataFrame(
            {
                "userId": [1, 1, 1, 1, 2, 2],
                "movieId": [1, 2, 4, 3, 4, 1],
                "rating": [5.0, 4.0, 1.0, 5.0, 5.0, 2.0],
                "split": ["train", "train", "train", "test", "train", "test"],
            }
        )
        popularity = pd.Series({1: 9, 2: 8, 3: 7, 4: 6, 5: 5}, name="count")
        per_user, info = evaluate(catalog, split, popularity, [Variant("content")], [1, 2], 100, 0)
        self.assertEqual(info["users_scored"], 1)  # user 2 has no liked hidden movie
        self.assertEqual(info["users_skipped"]["no_relevant_test_movies"], 1)
        content_k1 = per_user[(per_user["method"] == "content") & (per_user["k"] == 1)]
        self.assertEqual(content_k1["hit_rate"].iloc[0], 1.0)
        self.assertEqual(set(per_user["method"]), {"content", "popularity", "random"})


if __name__ == "__main__":
    unittest.main()
