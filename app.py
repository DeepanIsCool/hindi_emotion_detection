import streamlit as st
import torch
import librosa
import numpy as np
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import torch.nn.functional as F

# --- Configuration ---
# Path to your local model folder (unzipped) or Hugging Face Hub ID
MODEL_PATH = "hindi_emotion_model"  # Change this if your folder name is different

# --- Load Model & Feature Extractor (Cached for speed) ---
@st.cache_resource
def load_model():
    try:
        # Load the feature extractor and model
        feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)
        model = AutoModelForAudioClassification.from_pretrained(MODEL_PATH)
        return feature_extractor, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

feature_extractor, model = load_model()

# --- App UI ---
st.title("🎙️ Hindi/Indian English Emotion Recognition")
st.markdown("Upload a **.wav** audio file to detect the speaker's emotion.")

# File Uploader
audio_file = st.file_uploader("Upload Audio", type=["wav"])

if audio_file is not None and model is not None:
    # 1. Display Audio Player
    st.audio(audio_file, format="audio/wav")

    # 2. Process Audio
    with st.spinner("Analyzing emotion..."):
        try:
            # Load audio using librosa (handles resampling to 16kHz automatically)
            # Streamlit uploads are BytesIO objects, librosa can read them directly
            audio, sr = librosa.load(audio_file, sr=16000)

            # Preprocess inputs
            inputs = feature_extractor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=16000 * 4 # Cap at 4 seconds like training
            )

            # Inference
            with torch.no_grad():
                logits = model(**inputs).logits

            # Convert to probabilities
            scores = F.softmax(logits, dim=1).detach().numpy()[0]
            
            # Get labels from model config
            id2label = model.config.id2label
            
            # Find top prediction
            predicted_id = np.argmax(scores)
            predicted_label = id2label[predicted_id]
            confidence = scores[predicted_id]

            # 3. Display Results
            st.success(f"### Detected Emotion: **{predicted_label.upper()}**")
            st.metric("Confidence Score", f"{confidence:.1%}")

            # 4. Detailed Probabilities Chart
            st.subheader("Confidence breakdown:")
            probs_dict = {id2label[i]: float(score) for i, score in enumerate(scores)}
            st.bar_chart(probs_dict)

        except Exception as e:
            st.error(f"Error processing audio: {e}")

# Footer
st.markdown("---")
st.caption("Powered by Wav2Vec2-XLSR-53 | Trained on IESC Dataset")