# Content-backed-movie-recommendation-system

Install the Python dependencies with `python -m pip install pandas scipy
scikit-learn`. The MovieLens and IMDb files should be under `recommender_data/`;
the build command below joins them into `recommender_data/interactions.csv`.

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

`data_display.py --build-interactions` joins MovieLens ratings and tags with
IMDb title, genre, director, writer, and rating/vote-count metadata.
`content_recommender.py` builds a TF-IDF vector `x_i` for each movie from its
genres, user tags, directors, and writers. By default, movies with fewer than
100 IMDb votes are not recommended. For a user with highly rated movies
(`rating >= 4.0`), the model combines each movie's TF-IDF feature vector into
one user profile:

```text
weight = rating - 3
user profile = weighted average of liked movies' feature vectors
```

Only movies rated 4.0 or higher are included. This gives a 4-star movie weight
1 and a 5-star movie weight 2.

The recommender fits `sklearn.neighbors.NearestNeighbors` with
`metric="cosine"`. Its similarity score is `1 - distance`, and movies already
seen by the user are removed before selecting the top results.

Build the merged dataset and recommend from a MovieLens user profile:

```text
python data_display.py --build-interactions
python recommender_cli.py recommender_data\interactions.csv --user-id 123 -n 10
```

For a quick title-based test, use the existing movie-feature catalog directly;
this does not require building the 32-million-row interactions file. The
recommendation table is printed in the terminal:

```text
python recommender_cli.py --like "The Shining" --like "American Psycho" -n 10
```

The `--like` titles are treated as equally preferred. This matches the
title-input use case and does not ask the user to enter ratings. The
rating-weighted profile formula above applies only to `--user-id`: it reads
existing ratings for that MovieLens user from `interactions.csv`. It does not
represent ratings provided by the person currently running the recommender.
Use `--user-id` only when you have a user ID with rating history.

For a new user who is not in MovieLens, pass a CSV with a `movieId` column and
an optional `rating` column. Without ratings, the profile is the plain average
of the listed movies' vectors. With ratings, the `rating - 3` weighting above
is used. Every listed movie is excluded from the results. Add `--show-profile`
to print the highest-weighted features of the profile vector:

```text
python recommender_cli.py --ratings-file test_user_profile.csv --show-profile
```

`python -m unittest -v` runs assertions and reports pass/fail; it does not
print the recommendations table.

For duplicate titles, the highest-vote IMDb match is selected by default; add
the release year in parentheses to choose a specific release. Use `--min-votes`
to change the vote-count cutoff.
The merged CSV includes `movieId`, `userId`, `rating`, `primaryTitle`,
`genres_imdb`, `grouped_tags`, and `numVotes`; `directors` and `writers` are
joined as additional text features.
