
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Spotify Recommender", layout="centered")

st.title("🎧 Spotify Track Recommender")
st.markdown("Get song recommendations based on duration, popularity, and explicitness using ML.")

@st.cache_data
def load_data():
    # Replace with actual path if running locally with real CSV
    url = "https://raw.githubusercontent.com/rahman-projects/sample-datasets/main/spotify_tracks_small.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

features = ['duration_ms', 'popularity', 'explicit']
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = NearestNeighbors(n_neighbors=5)
model.fit(X_scaled)

# Song selection
song_options = df['track_name'] + " - " + df['artist_name']
selected_song = st.selectbox("Select a song:", song_options)

if st.button("Recommend Similar Songs"):
    song_index = song_options.tolist().index(selected_song)
    distances, indices = model.kneighbors([X_scaled[song_index]])
    recommendations = df.iloc[indices[0]][['track_name', 'artist_name', 'popularity']]
    st.success("🎶 Recommended Tracks:")
    st.dataframe(recommendations.reset_index(drop=True))
