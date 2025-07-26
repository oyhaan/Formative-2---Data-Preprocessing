import streamlit as st
import cv2
import numpy as np
import librosa
import joblib
import tempfile

st.title("🔐 Secure Product Recommendation System")

face_model = joblib.load("../models/face_recognition_model.pkl")
#voice_model = joblib.load("models/voice_verification_model.pkl")
#product_model = joblib.load("models/product_recommendation_model.pkl")

def extract_histogram_features(img):
    chans = cv2.split(img)
    hist = np.concatenate([
        cv2.calcHist([c], [0], None, [256], [0, 256]).flatten()
        for c in chans
    ])
    return hist

def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    return np.mean(mfcc, axis=1)

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

    #st.header("Step 2: Upload Voice Confirmation")
    #audio_file = st.file_uploader("Upload your voice (.wav)", type=["wav"])
    #if audio_file:
        #with tempfile.NamedTemporaryFile(delete=False) as tmp:
            #tmp.write(audio_file.read())
            #audio_feat = extract_audio_features(tmp.name).reshape(1, -1)
            #phrase = voice_model.predict(audio_feat)[0]
            #st.success(f"🔊 Voice Verified: '{phrase}'")

            #if phrase.lower() in ["yes", "approve", "confirm transaction"]:
                #st.header("Step 3: Product Recommendation")
                # Dummy input features
                #input_data = np.random.rand(1, product_model.n_features_in_)
                #predicted_product = product_model.predict(input_data)[0]
                #st.success(f"🎯 Recommended Product: {predicted_product}")
            #else:
                #st.error("❌ Voice authentication failed. Access denied.")
