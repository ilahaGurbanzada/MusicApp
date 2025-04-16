
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Spotify Recommender", layout="centered")
st.title("🎧 Spotify Track Recommender")

@st.cache_data
def load_data():
    return pd.read_csv("spotify_tracks_small.csv")

df = load_data()

# Features selected for recommendation
features = ['Danceability', 'Energy', 'Loudness', 'Acousticness']
df_clean = df.dropna(subset=features)

# Prepare feature matrix
X = df_clean[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = NearestNeighbors(n_neighbors=5).fit(X_scaled)

# Display song selection
song_options = df_clean['Track'] + " - " + df_clean['Artist']
selected_song = st.selectbox("🎵 Select a song:", song_options)

if st.button("🎧 Recommend Similar Tracks"):
    song_index = song_options.tolist().index(selected_song)
    distances, indices = model.kneighbors([X_scaled[song_index]])
    recommendations = df_clean.iloc[indices[0]][['Track', 'Artist', 'Album', 'Views', 'Likes']]
    st.success("🔊 Recommended Songs:")
    st.dataframe(recommendations.reset_index(drop=True))
