import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import librosa
import numpy as np

# Function to extract features from sound data
def extract_features(file_path, max_sequence_length=100):
    # Load sound file using librosa
    signal, sample_rate = librosa.load(file_path, sr=None)

    # Cropping & Resampling (similar to what you did during training)
    start_time = 0.4
    end_time = 1.9
    start_frame = int(start_time * sample_rate)
    end_frame = int(end_time * sample_rate)
    signal = signal[start_frame:end_frame]

    # Resample
    signal = librosa.resample(signal, orig_sr=sample_rate, target_sr=16000)

    # Extract features for the sound data
    spectrogram = librosa.feature.melspectrogram(y=signal, sr=16000, n_fft=2048, hop_length=512)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(spectrogram), n_mfcc=13)

    # Transpose features to have shape (num_features, sequence_length)
    features = mfcc.T

    # Pad or truncate features to a fixed length
    features = np.pad(features, ((0, max_sequence_length - features.shape[0]), (0, 0)), mode='constant')

    return features


def predict_emotion(sound_file_path,loaded_model):
    # Extract features
    sound_features = extract_features(sound_file_path)
    
    # Ensure sound_features has the correct shape (adjust if necessary)
    sound_features = np.expand_dims(sound_features, axis=0)
    
    # Make predictions using the loaded model
    emotion_probabilities = loaded_model.predict(sound_features)
    
    # Convert the probabilities to emotion labels
    predicted_emotion = np.argmax(emotion_probabilities, axis=1)
    
    # Map the predicted label to the actual emotion class (e.g., using a dictionary)
    emotion_mapping = {0: 'Neutral', 1: 'Happy', 2: 'Surprise', 3: 'Unpleasant'}
    predicted_emotion_label = emotion_mapping[predicted_emotion[0]]

    return predicted_emotion_label

# Streamlit UI
def main():
    st.set_page_config(page_title="Audio Emotion Classifier", page_icon=":sound:")
    st.title("Audio Emotion Classifier")

    # File uploader
    audio_file = st.file_uploader("Upload an audio file (WAV or MP3)", type=["wav", "mp3"])

    if audio_file is not None:
        st.audio(audio_file, format='audio/wav', start_time=0)

        # Load the pre-trained model
        saved_model_path = "model.h5"
        loaded_model = load_model(saved_model_path)

        if st.button("Classify"):
            # Classify the audio
            with st.spinner("Classifying..."):
                prediction = predict_emotion(audio_file, loaded_model)
            # Display the result
            st.success(f"Predicted Emotion: {prediction}")

# Run the Streamlit app
main()