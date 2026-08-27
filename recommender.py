import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv("movies.csv")


# ==========================================
# PARSE DATA
# ==========================================

def parse_space_separated_list(text_str):

    if pd.isna(text_str) or not isinstance(text_str, str):
        return []

    return [
        item.strip()
        for item in text_str.split(" ")
        if item.strip()
    ]


df["genres_list"] = df["genres"].apply(
    parse_space_separated_list
)

df["keywords_list"] = df["keywords"].apply(
    parse_space_separated_list
)


# ==========================================
# DATE + FINANCIAL FEATURES
# ==========================================

df["release_date"] = pd.to_datetime(
    df["release_date"],
    errors="coerce"
)

df["release_year"] = df["release_date"].dt.year

df["profit"] = (
    df["revenue"] -
    df["budget"]
)

df["roi_percent"] = np.where(
    df["budget"] > 0,
    (df["profit"] / df["budget"]) * 100,
    0
)


# ==========================================
# TEXT FEATURES
# ==========================================

df["genres_text"] = df["genres_list"].apply(
    lambda x: " ".join(x)
    if isinstance(x, list)
    else ""
)

df["keywords_text"] = df["keywords_list"].apply(
    lambda x: " ".join(x)
    if isinstance(x, list)
    else ""
)


df["overview_text"] = df["overview"].fillna("")


df["movie_soup"] = (
    df["genres_text"] + " " +
    df["keywords_text"] + " " +
    df["overview_text"]
)


# ==========================================
# TF-IDF
# ==========================================

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=10000,
    ngram_range=(1, 2)
)

tfidf_matrix = tfidf.fit_transform(
    df["movie_soup"]
)


# ==========================================
# COSINE SIMILARITY
# ==========================================

cosine_sim = cosine_similarity(
    tfidf_matrix
)


# ==========================================
# MOVIE INDEX
# ==========================================

indices = pd.Series(
    df.index,
    index=df["title"].str.lower()
).drop_duplicates()


# ==========================================
# CONTENT RECOMMENDER
# ==========================================

def recommend_movies(title, n=10):

    title_key = title.lower()

    if title_key not in indices:
        return pd.DataFrame()

    idx = indices[title_key]

    similarity_scores = list(
        enumerate(cosine_sim[idx])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    similarity_scores = similarity_scores[1:n+1]

    movie_indices = [
        item[0]
        for item in similarity_scores
    ]

    recommendations = df.iloc[
        movie_indices
    ][
        [
            "title",
            "genres_text",
            "vote_average",
            "popularity"
        ]
    ].copy()

    recommendations["similarity_score"] = [
        round(item[1], 3)
        for item in similarity_scores
    ]

    return recommendations


# ==========================================
# HYBRID RECOMMENDER
# ==========================================

def hybrid_recommend_movies(title, n=10):

    title_key = title.lower()

    if title_key not in indices:
        return pd.DataFrame()

    idx = indices[title_key]

    similarity = cosine_sim[idx]

    popularity = df["popularity"].fillna(0)

    popularity_norm = (
        (popularity - popularity.min()) /
        (
            popularity.max() -
            popularity.min()
        )
    )

    rating = df["vote_average"].fillna(
        df["vote_average"].median()
    )

    rating_norm = rating / 10

    hybrid_score = (
        0.70 * similarity +
        0.15 * popularity_norm +
        0.15 * rating_norm
    )

    hybrid_score[idx] = -1

    top_indices = np.argsort(
        hybrid_score
    )[::-1][:n]

    result = df.iloc[
        top_indices
    ][
        [
            "title",
            "genres_text",
            "vote_average",
            "popularity"
        ]
    ].copy()

    result["similarity"] = similarity[
        top_indices
    ]

    result["hybrid_score"] = hybrid_score[
        top_indices
    ]

    return result.reset_index(drop=True)


# ==========================================
# GENRE RECOMMENDER
# ==========================================

def recommend_by_genre(genre, n=10):

    genre_movies = df[
        df["genres_list"].apply(
            lambda genres:
            genre.lower()
            in [
                g.lower()
                for g in genres
            ]
        )
    ].copy()

    genre_movies = genre_movies.sort_values(
        by=[
            "vote_average",
            "popularity"
        ],
        ascending=False
    )

    return genre_movies[
        [
            "title",
            "genres_text",
            "vote_average",
            "popularity"
        ]
    ].head(n)