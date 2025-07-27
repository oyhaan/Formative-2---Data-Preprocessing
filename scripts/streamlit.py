import pandas as pd
import streamlit as st
import cv2
import numpy as np
import librosa
import joblib
import tempfile
import os


# Page setup
st.set_page_config(page_title="Secure Product Recommendation", page_icon="🔐")
st.title("🔐 Secure Product Recommendation System")

# Load models
face_model = joblib.load("../models/face_recognition_model.pkl")
voice_model = joblib.load("../models/voiceprint_verification_model.pkl")
label_encoder = joblib.load("../models/voiceprint_label_encoder.pkl")  # ✅ Load label encoder
product_model = joblib.load("../models/product_recommendation_model.joblib")

customer_id_to_person = {
    187: 'Owen',
    177: 'Nicolas',
    189: 'Abiodun',
    120: 'Gaius',
    103: 'Anissa'
}

# Feature extraction functions
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

def verify_identity(img, audio_path):
    face_feat = extract_histogram_features(img).reshape(1, -1)
    pred_img_person = face_model.predict(face_feat)[0]

    audio_feat = extract_audio_features(audio_path).reshape(1, -1)
    voice_class_index = voice_model.predict(audio_feat)[0]
    pred_voice_person = int(label_encoder.inverse_transform([voice_class_index])[0])

    return pred_img_person, pred_voice_person

# UI: Upload Face Image
st.header("Step 1: Upload Face Image")
face_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if face_file:
    file_bytes = np.asarray(bytearray(face_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption="Uploaded Face", use_container_width=True)

        # Step 1: Facial Recognition
        face_feat_raw = extract_histogram_features(img).reshape(1, -1)
        feature_names = [f'feat_{i}' for i in range(face_feat_raw.shape[1])]
        face_feat = pd.DataFrame(face_feat_raw, columns=feature_names)

        try:
            face_proba = face_model.predict_proba(face_feat)
            max_conf = face_proba.max()
            pred_img_person = face_model.predict(face_feat)[0]

            if max_conf < 0.9:
                st.error(f"❌ User Not Recognized. Access denied.")
                st.stop()
            else:
                person_name = customer_id_to_person.get(pred_img_person, f"Unknown ({pred_img_person})")
                st.success(f"🧑 Face identified as: *{person_name}*")

        except Exception as e:
            st.error(f"❌ Face recognition failed: {str(e)}")
            st.stop()

        # Step 2: Run Product Recommendation Immediately After Face Match
        st.header("Step 2: Product Recommendation (Preliminary)")
        input_data = np.random.rand(1, product_model.n_features_in_)
        try:
            predicted_product = product_model.predict(input_data)[0]
            st.info("✅ Product computed and waiting for voice verification...")
        except Exception as e:
            st.error(f"❌ Product model failed: {str(e)}")
            st.stop()

        # Step 3: Upload and Validate Voice
        st.header("Step 3: Voice Confirmation")
        audio_file = st.file_uploader("Upload your voice (.wav)", type=["wav","ogg"])

        if audio_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_file.read())
                audio_path = tmp.name

            st.audio(audio_path, format="audio/wav")

            try:
                audio_feat = extract_audio_features(audio_path).reshape(1, -1)
                pred_voice_person = voice_model.predict(audio_feat)[0]
                print(f"Voice predicted: {pred_voice_person}")
                st.success(f"🎤 Voice identified as: *{pred_voice_person}*")

                if pred_voice_person == pred_img_person:
                    st.success("✅ Identity Verified: Face and voice match.")
                    st.header("🎯 Final Recommendation")
                    st.success(f"Recommended Product: *{predicted_product}*")
                else:
                    st.error("❌ Identity Mismatch: Voice does not match face. Access Denied.")

            except Exception as e:
                st.error(f"❌ Voice validation failed: {str(e)}")
    else:
        st.error("❌ Failed to decode image. Please upload a valid image file.")
