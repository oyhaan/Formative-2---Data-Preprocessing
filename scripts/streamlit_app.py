import streamlit as st
import cv2
import numpy as np
import librosa
import joblib
import tempfile
import pandas as pd

st.title("🔐 Secure Product Recommendation System")

# Load face recognition model
face_model = joblib.load("models/face_recognition_model.pkl")

# Load voiceprint verification components
voice_model = joblib.load("models/voiceprint_verification_model.pkl")
voice_scaler = joblib.load("models/voiceprint_scaler.pkl")
voice_label_encoder = joblib.load("models/voiceprint_label_encoder.pkl")
voice_feature_cols = joblib.load("models/voiceprint_feature_columns.pkl")
person_to_customer_id = joblib.load("models/voiceprint_person_to_customer.pkl")

# Load product recommendation model
product_model = joblib.load("models/product_recommendation_model.joblib")

def extract_histogram_features(img):
    chans = cv2.split(img)
    hist = np.concatenate([
        cv2.calcHist([c], [0], None, [256], [0, 256]).flatten()
        for c in chans
    ])
    return hist

def extract_audio_features(file_path, feature_cols):
    y, sr = librosa.load(file_path, sr=None)
    features = {}
    
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i+1}'] = np.mean(mfccs[i])
    
    # Extract other features
    features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    features['energy'] = np.mean(librosa.feature.rms(y=y))
    features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y))
    features['spectral_contrast'] = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
    
    # Extract harmonic and percussive components
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    features['harmonic_energy'] = np.mean(librosa.feature.rms(y=y_harmonic))
    features['percussive_energy'] = np.mean(librosa.feature.rms(y=y_percussive))
    features['harmonic_percussive_ratio'] = features['harmonic_energy'] / (features['percussive_energy'] + 1e-10)
    
    # Extract fundamental frequency features
    try:
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        features['f0_mean'] = np.nanmean(f0) if not np.all(np.isnan(f0)) else 0
        features['f0_std'] = np.nanstd(f0) if not np.all(np.isnan(f0)) else 0
    except:
        features['f0_mean'] = 0
        features['f0_std'] = 0
    
    # Select only the features used in the model
    feature_values = [features.get(col, 0) for col in feature_cols]
    return np.array(feature_values).reshape(1, -1)

st.header("Step 1: Upload Face Image")
face_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if face_file:
    file_bytes = np.asarray(bytearray(face_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, channels="RGB")
        face_feat = extract_histogram_features(img).reshape(1, -1)
        predicted_user = face_model.predict(face_feat)[0]
        st.success(f"✅ Face recognized: {predicted_user}")
    else:
        st.error("❌ Failed to decode image. Please upload a valid image file.")

    st.header("Step 2: Upload Voice Confirmation")
    audio_file = st.file_uploader("Upload your voice (.wav)", type=["wav", "ogg"])
    if audio_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(audio_file.read())
            audio_feat = extract_audio_features(tmp.name, voice_feature_cols)
            scaled_audio_feat = voice_scaler.transform(audio_feat)
            predicted_speaker_encoded = voice_model.predict(scaled_audio_feat)[0]
            predicted_speaker = voice_label_encoder.inverse_transform([predicted_speaker_encoded])[0]
            predicted_customer_id = person_to_customer_id.get(predicted_speaker, -1)
            
            # For this example, assume predicted_user from face recognition is the customer ID
            # In a real scenario, map face_model output to customer ID as needed
            face_customer_id = predicted_user
            
            if predicted_customer_id == face_customer_id:
                st.success(f"🔊 Voice Verified: Uploaded audio matches {predicted_speaker.title()}\'s image")
                st.header("Step 3: Product Recommendation")

                
                # Dummy input features for product recommendation
                input_data = np.random.rand(1, product_model.n_features_in_)
                predicted_product = product_model.predict(input_data)[0]
                st.success(f"🎯 Recommended Product: {predicted_product}")
            else:
                st.error("❌ Voice authentication failed. Access denied.")