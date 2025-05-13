import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import tensorflow as tf
import os

# Set page config
st.set_page_config(page_title="🎵 Mood Melody Classifier", layout="centered")
st.title("🎶 Mood Melody Classifier + Playlist Generator")
st.markdown("Upload an image to predict your mood and get music recommendations!")

# Load model once
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("MOODMELODY_model.h5")
    return model

# Load music dataset once
@st.cache_data
def load_music_data():
    return pd.read_csv("songs_MOODMELODY.csv")  # Assuming you now have songs_MOODMELODY.csv

# Load resources
model = load_model()
music_data = load_music_data()

# Check if required columns exist
required_columns = {'energy', 'valence', 'tempo', 'loudness', 'danceability', 'name', 'artists'}
if not required_columns.issubset(set(music_data.columns)):
    st.error("⚠️ Music dataset is missing required columns. Please check your data file.")
    st.stop()

# Emotion classes
emotion_classes = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Mapping emotion to music features
def map_emotion_to_music_features(emotion):
    emotion_mapping = {
        'angry': {'energy': 0.8, 'valence': 0.2, 'tempo': 140, 'loudness': -5, 'danceability': 0.5},
        'disgusted': {'energy': 0.7, 'valence': 0.3, 'tempo': 130, 'loudness': -8, 'danceability': 0.4},
        'fearful': {'energy': 0.6, 'valence': 0.2, 'tempo': 100, 'loudness': -10, 'danceability': 0.3},
        'happy': {'energy': 0.7, 'valence': 0.8, 'tempo': 120, 'loudness': -6, 'danceability': 0.7},
        'neutral': {'energy': 0.5, 'valence': 0.5, 'tempo': 110, 'loudness': -10, 'danceability': 0.5},
        'sad': {'energy': 0.3, 'valence': 0.2, 'tempo': 80, 'loudness': -12, 'danceability': 0.3},
        'surprised': {'energy': 0.8, 'valence': 0.6, 'tempo': 125, 'loudness': -7, 'danceability': 0.6}
    }
    return emotion_mapping.get(emotion, emotion_mapping['neutral'])

# Function to recommend songs
def recommend_songs_by_emotion(emotion, music_data, num_recommendations=10):
    preferred_features = map_emotion_to_music_features(emotion)
    music_df = music_data.copy()

    # Normalize features
    def min_max_scale(x, feature_name):
        min_val = music_df[feature_name].min()
        max_val = music_df[feature_name].max()
        return (x - min_val) / (max_val - min_val) if max_val > min_val else 0

    try:
        music_df['energy_dist'] = music_df['energy'].apply(lambda x: abs(x - preferred_features['energy']))
        music_df['valence_dist'] = music_df['valence'].apply(lambda x: abs(x - preferred_features['valence']))
        music_df['tempo_dist'] = music_df['tempo'].apply(lambda x: min_max_scale(abs(x - preferred_features['tempo']), 'tempo'))
        music_df['loudness_dist'] = music_df['loudness'].apply(lambda x: min_max_scale(abs(x - preferred_features['loudness']), 'loudness'))
        music_df['danceability_dist'] = music_df['danceability'].apply(lambda x: abs(x - preferred_features['danceability']))

        # Overall score
        music_df['total_dist'] = (
            music_df['energy_dist'] * 0.25 +
            music_df['valence_dist'] * 0.3 +
            music_df['tempo_dist'] * 0.15 +
            music_df['loudness_dist'] * 0.1 +
            music_df['danceability_dist'] * 0.2
        )

        # Top recommendations
        recommendations = music_df.sort_values('total_dist').head(num_recommendations)
        return recommendations[['name', 'artists']]
    except Exception as e:
        st.error(f"Error recommending songs: {e}")
        return pd.DataFrame()

# Upload and Predict
uploaded_file = st.file_uploader("📷 Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load and preprocess image
        image = Image.open(uploaded_file).convert("L")  # Grayscale
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img = np.array(image)
        img = cv2.resize(img, (48, 48))
        img = img / 255.0
        img = np.expand_dims(img, axis=-1)  # (48,48,1)
        img = np.expand_dims(img, axis=0)   # (1,48,48,1)

        # Predict emotion
        predictions = model.predict(img)
        if predictions is None or len(predictions) == 0:
            st.error("⚠️ Unable to predict mood. Please try again with a different image.")
            st.stop()
        predicted_index = np.argmax(predictions)
        predicted_emotion = emotion_classes[predicted_index]

        st.markdown("---")
        st.subheader("🧠 Predicted Mood")
        st.success(f"**Mood:** `{predicted_emotion}`")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.stop()

else:
    st.info("📤 No image uploaded. Defaulting to neutral mood.")
    predicted_emotion = "neutral"

# Recommend songs based on predicted emotion
st.subheader("🎵 Recommended Songs")
recommendations = recommend_songs_by_emotion(predicted_emotion, music_data)

if not recommendations.empty:
    for i, row in recommendations.iterrows():
        st.markdown(f"<h4 style='text-align: center;'><a href='#'>{row['name']} - {row['artists']}</a></h4>", unsafe_allow_html=True)
else:
    st.warning("⚠️ No songs found for the detected mood.")
