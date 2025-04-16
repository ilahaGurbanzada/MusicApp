
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import kagglehub
import os

st.set_page_config(page_title="Spotify Recommender", layout="centered")
st.title("🎧 Spotify Track Recommender")

@st.cache_data
def load_data():
    # Download from KaggleHub (must be authorized)
    path = kagglehub.dataset_download("sanjanchaudhari/spotify-dataset")
    csv_path = os.path.join(path, "spotify_dataset.csv")
    df = pd.read_csv(csv_path)
    return df

try:
    df = load_data()

    features = ['duration_ms', 'popularity', 'explicit']
    df = df.dropna(subset=features)
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = NearestNeighbors(n_neighbors=5).fit(X_scaled)

    song_options = df['track_name'] + " - " + df['artist_name']
    selected_song = st.selectbox("Select a song:", song_options)

    if st.button("Recommend Similar Songs"):
        song_index = song_options.tolist().index(selected_song)
        distances, indices = model.kneighbors([X_scaled[song_index]])
        recommendations = df.iloc[indices[0]][['track_name', 'artist_name', 'popularity']]
        st.success("🎶 Recommended Tracks:")
        st.dataframe(recommendations.reset_index(drop=True))

except Exception as e:
    st.error(f"⚠️ Failed to load data or build model: {e}")
    st.markdown("Please make sure you have access to KaggleHub and the dataset is available.")
