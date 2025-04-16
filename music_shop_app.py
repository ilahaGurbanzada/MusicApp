
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="🎵 Music Shop Application", layout="centered")
st.title("🛒 Design and Development of a Music Shop Application")
st.subheader("A Comprehensive Approach to Database Management and UI/UX Design")

@st.cache_data
def load_data():
    df = pd.read_csv("spotify_tracks_small.csv")
    df = df.dropna(subset=['Danceability', 'Energy', 'Loudness', 'Acousticness'])
    df['Price ($)'] = np.random.randint(5, 100, df.shape[0])
    df['Times Bought'] = np.random.randint(0, 500, df.shape[0])
    return df

df = load_data()

# Features for ML model
features = ['Danceability', 'Energy', 'Loudness', 'Acousticness']
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = NearestNeighbors(n_neighbors=5).fit(X_scaled)

# Song selection
song_options = df['Track'] + " - " + df['Artist']
selected_song = st.selectbox("🎧 Select a song to view details and recommendations:", song_options)

if st.button("🛍️ View Song & Recommendations"):
    song_index = song_options.tolist().index(selected_song)
    song_data = df.iloc[song_index]
    
    st.markdown(f"""### 🎼 {song_data['Track']}  
**Artist:** {song_data['Artist']}  
**Album:** {song_data['Album']}  
**Price:** ${song_data['Price ($)']}  
**Bought:** {song_data['Times Bought']} times  
**Views:** {int(song_data['Views']) if not pd.isna(song_data['Views']) else 'N/A'}  
""")
    
    distances, indices = model.kneighbors([X_scaled[song_index]])
    recommendations = df.iloc[indices[0]][['Track', 'Artist', 'Price ($)', 'Times Bought', 'Views']]
    st.markdown("### 🔁 Recommended Songs:")
    st.dataframe(recommendations.reset_index(drop=True))
