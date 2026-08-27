import streamlit as st
import pandas as pd

from recommender import (
    df,
    recommend_movies,
    hybrid_recommend_movies,
)

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide"
)

# -----------------------------
# CUSTOM HTML / CSS
# -----------------------------

st.html("""
<style>

.main-title {
    font-size: 52px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 35px;
}

.movie-card {
    padding: 20px;
    border-radius: 15px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-bottom: 15px;
}

</style>
""")

# -----------------------------
# HEADER
# -----------------------------

st.markdown(
    '<div class="main-title">🎬 CineMatch</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent Movie Recommendation Engine'
    '</div>',
    unsafe_allow_html=True
)

# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.header("Recommendation Settings")

mode = st.sidebar.radio(
    "Recommendation method",
    [
        "Hybrid",
        "Content-Based"
    ]
)

number = st.sidebar.slider(
    "Number of recommendations",
    3,
    15,
    6
)

# -----------------------------
# MOVIE SELECTOR
# -----------------------------

titles = sorted(
    df["title"]
    .dropna()
    .unique()
)

selected_movie = st.selectbox(
    "🎥 What movie do you like?",
    titles
)

# -----------------------------
# RECOMMEND BUTTON
# -----------------------------

if st.button(
    "✨ RECOMMEND MOVIES",
    use_container_width=True
):

    if mode == "Hybrid":

        results = hybrid_recommend_movies(
            selected_movie,
            number
        )

    else:

        results = recommend_movies(
            selected_movie,
            number
        )

    if results.empty:

        st.error(
            "No recommendations found."
        )

    else:

        st.markdown(
            "## 🍿 Recommended for You"
        )

        columns = st.columns(3)

        for i, (_, movie) in enumerate(
            results.iterrows()
        ):

            with columns[i % 3]:

                st.markdown(
                    '<div class="movie-card">',
                    unsafe_allow_html=True
                )

                st.subheader(
                    f"🎬 {movie['title']}"
                )

                st.write(
                    f"⭐ Rating: "
                    f"{movie['vote_average']}"
                )

                st.write(
                    f"🎭 {movie['genres_text']}"
                )

                if "similarity_score" in movie:
                    st.write(
                        f"🔗 Similarity: "
                        f"{movie['similarity_score']:.3f}"
                    )

                if "hybrid_score" in movie:
                    st.write(
                        f"🧠 Hybrid Score: "
                        f"{movie['hybrid_score']:.3f}"
                    )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )