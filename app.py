import streamlit as st
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.metrics.pairwise import cosine_similarity


# loading the song_dataset [in cache form to minimize resource usage]
@st.cache_data
def load_songData(path_to_file):
    df = pd.read_csv(path_to_file)
    return df


# calling load_songData function
song_df = load_songData("song_dataset.csv")

# load all user IDs
all_users = song_df["user"].unique()

# create a series with song IDs as index and titles as values
song_titles_series = song_df.drop_duplicates(subset=["song"]).set_index("song")["title"]

# sparse item-item similarity: transpose sparse matrix because we want item-item similarity (songs as rows)
interaction_matrix = song_df.pivot_table(
    index="user", columns="song", values="play_count", fill_value=0
)
sparse_matrix = csr_matrix(interaction_matrix)

item_similarity_sparse = cosine_similarity(sparse_matrix.T, dense_output=False)
coo = coo_matrix(item_similarity_sparse)

item_similarity_df = pd.DataFrame(
    {
        "item_1": interaction_matrix.columns[coo.row],
        "item_2": interaction_matrix.columns[coo.col],
        "similarity": coo.data,
    }
)

# for song-based recommendation engine call


def recommend_similar_items_sparse(selected_songs, top_n):
    scores = {}
    for song in selected_songs:
        # getting all rows where item_1 is the selected song
        similar_items = item_similarity_df[item_similarity_df["item_1"] == song]

        for _, row in similar_items.iterrows():
            similar_song = row["item_2"]
            similarity = row["similarity"]
            # filtering out songs already listened to by user
            if similar_song not in selected_songs:
                scores[similar_song] = scores.get(similar_song, 0) + similarity

        recommended_songs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[
            :top_n
        ]
        return [song for song, score in recommended_songs]


# Streamlit Interface
st.subheader("Song Recommendation Engine[Proj_charlie]")

# Geting User inputs
st.write("**Please select a user.**")
selected_user_id = st.selectbox("Select a User ID:", options=all_users)


st.write("**Please select as many songs as user has listened to.**")
all_songs = song_titles_series.index.tolist()
selected_songs = st.multiselect(
    "Select Song(s):",
    options=all_songs,
    format_func=lambda x: song_titles_series[x],
)


# Recommendation Magic

if st.button("Get Recommendations"):
    if not selected_songs:
        st.warning("Please select song(s) you have listened to!")
    else:
        recommendations = recommend_similar_items_sparse(selected_songs, top_n=10)
        if recommendations:
            # displaying selected songs:
            st.subheader(f"Great! {selected_user_id} has listened to:")
            for idx, song in enumerate(selected_songs, start=1):
                st.write(f"{idx}. {song_titles_series.get(song, song)}")

            # displaying recommended songs
            st.subheader(f"Top Recommended songs for {selected_user_id}:")
            for idx, song in enumerate(recommendations, start=1):
                st.write(f"{idx}. {song_titles_series.get(song, song)}")
        else:
            st.info("Oops! No recommendations available for selected songs.")
