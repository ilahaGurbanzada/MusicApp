
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title='🎶 GroovyGrooves', layout='centered')
st.title('🎶 GroovyGrooves')
st.caption('Your Personalized Music Recommender & Digital Music Shop 🎧🛒')

@st.cache_data
def load_data():
    df = pd.read_csv("spotify_tracks_small.csv")
    df = df.dropna(subset=['Danceability', 'Energy', 'Loudness', 'Acousticness'])
    df['Price ($)'] = np.random.randint(5, 100, df.shape[0])
    df['Times Bought'] = np.random.randint(0, 500, df.shape[0])
    return df

df = load_data()

# Session state for cart and reset flag
if 'cart' not in st.session_state:
    st.session_state.cart = []

if 'just_bought' not in st.session_state:
    st.session_state.just_bought = False

# Show confirmation popup after reload
if st.session_state.just_bought:
    st.success("🛍️ Purchase completed successfully!")
    st.session_state.just_bought = False

# Prepare ML model
features = ['Danceability', 'Energy', 'Loudness', 'Acousticness']
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = NearestNeighbors(n_neighbors=5).fit(X_scaled)

# Song selection
song_options = df['Track'] + " - " + df['Artist']
selected_song = st.selectbox("🎵 Pick a track:", song_options)

if selected_song:
    idx = song_options.tolist().index(selected_song)
    selected_data = df.iloc[idx]

    with st.expander(f"💿 {selected_data['Track']} by {selected_data['Artist']}", expanded=True):
        st.write(f"**💰 Price:** ${selected_data['Price ($)']}")
        st.write(f"**🎧 Album:** {selected_data['Album']}")
        st.write(f"**🔥 Bought:** {selected_data['Times Bought']} times")
        st.write(f"**📺 Views:** {int(selected_data['Views']) if not pd.isna(selected_data['Views']) else 'N/A'}")
        if st.button("🛒 Add to Cart", key='add_main'):
            st.session_state.cart.append({
                "Track": selected_data['Track'],
                "Artist": selected_data['Artist'],
                "Price ($)": selected_data['Price ($)']
            })
            st.success(f"✅ {selected_data['Track']} added to cart!")

    # Recommended songs
    st.subheader("🎯 Recommended Songs")
    distances, indices = model.kneighbors([X_scaled[idx]])
    recommended_df = df.iloc[indices[0]]

    for i, row in recommended_df.iterrows():
        rec_key = f"rec_{i}"
        with st.expander(f"🎵 {row['Track']} by {row['Artist']}", expanded=False):
            st.write(f"**💰 Price:** ${row['Price ($)']}")
            st.write(f"**🎧 Album:** {row['Album']}")
            st.write(f"**🔥 Bought:** {row['Times Bought']} times")
            st.write(f"**📺 Views:** {int(row['Views']) if not pd.isna(row['Views']) else 'N/A'}")
            if st.button("🛒 Add to Cart", key=rec_key):
                st.session_state.cart.append({
                    "Track": row['Track'],
                    "Artist": row['Artist'],
                    "Price ($)": row['Price ($)']
                })
                st.success(f"✅ {row['Track']} added to cart!")

# Cart section
st.markdown("---")
st.subheader("🛒 Your Shopping Cart")

if st.session_state.cart:
    cart_df = pd.DataFrame(st.session_state.cart)
    total = cart_df["Price ($)"].sum()

    st.dataframe(cart_df.reset_index(drop=True), use_container_width=True)
    st.write(f"**Total: ${total}**")

    remove_index = st.number_input("🗑️ Remove item # (index)", min_value=0, max_value=len(cart_df)-1, step=1)
    if st.button("❌ Remove Selected"):
        st.session_state.cart.pop(remove_index)
        st.rerun()

    if st.button("💳 Buy All"):
        st.session_state.cart.clear()
        st.session_state.just_bought = True
        st.rerun()
else:
    st.info("🛒 Your cart is empty. Add songs from above.")
