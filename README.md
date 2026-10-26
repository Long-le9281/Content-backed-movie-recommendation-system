# Content-backed-movie-recommendation-system

To set up the project: 
run ./requirements.txt in terminal to get dataset

Meeting 2 (23-Sep-2026)
How do you validate accuracy?
no real users so there's a difficulty in testing
the recommendation system can be wrong,
there is no ground truth
-- look into how to validate a recommendation system

Meeting 3 (30-Sep-2026)
Start with K-NN, vary K, re-run, check how the set of movies are classified....
check other classification approach
visualize 

What are the mathematical tools?

## Content-based recommender

`content_recommender.py` builds a TF-IDF vector `x_i` for each movie by
concatenating its `genres_imdb` and `grouped_tags` text. For a user with
highly rated movies (`rating >= 4.0`), the rating-weighted profile is:

```text
p = sum((r_i - 3) * x_i) / sum(r_i - 3)
```

The recommender fits `sklearn.neighbors.NearestNeighbors` with
`metric="cosine"`. Its similarity score is `1 - distance`, and movies already
rated by the user are removed from the returned results.

Run the CLI with a merged rating-level CSV:

```text
python content_recommender.py path\to\merged.csv 123 -n 10
```

The CSV must contain `movieId`, `userId`, `rating`, `primaryTitle`,
`genres_imdb`, and `grouped_tags`.
