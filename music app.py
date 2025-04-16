
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Sample Spotify dataset
df = pd.DataFrame({
    'track_name': ['Song A', 'Song B', 'Song C', 'Song D', 'Song E'],
    'artist_name': ['Artist 1', 'Artist 2', 'Artist 3', 'Artist 4', 'Artist 5'],
    'duration_ms': [210000, 180000, 200000, 240000, 220000],
    'popularity': [60, 75, 50, 85, 65],
    'explicit': [0, 1, 0, 1, 0]
})

features = ['duration_ms', 'popularity', 'explicit']
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = NearestNeighbors(n_neighbors=3)
model.fit(X_scaled)

st.title("🎵 Spotify Song Recommender")

st.markdown("Enter the **index** of a song (0–4) to get recommendations based on its audio features.")

song_index = st.number_input("Choose a song index:", min_value=0, max_value=len(df)-1, step=1)

if st.button("Get Recommendations"):
    distances, indices = model.kneighbors([X_scaled[song_index]])
    recommendations = df.iloc[indices[0]]
    st.subheader("Top Recommended Songs:")
    st.table(recommendations[['track_name', 'artist_name']])
