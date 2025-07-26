import os
import cv2
import numpy as np
import pandas as pd
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder


# Constants
IMAGE_FOLDER = 'images_data'
AUDIO_FOLDER = 'audio_data'
IMAGE_FEATURES_CSV = 'image_features.csv'
AUDIO_FEATURES_CSV = 'audio_features.csv'
SOCIAL_PROFILE_CSV = 'data/customer_social_profiles.csv'
TRANSACTION_CSV = 'data/customer_transactions.csv'
MERGED_CSV = 'data/merged_data.csv'

# Load pretrained models
face_model = joblib.load('../models/face_recognition_model.pkl')
voice_model = joblib.load('../models/voiceprint_verification_model.pkl')
product_model = joblib.load('../models/product_recommendation_model.joblib')
label_encoder = joblib.load('../models/product_label_encoder.pkl')

# Utility functions
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

# Inference Functions
def predict_face_owner(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    features = extract_histogram_features(img)
    return face_model.predict([features])[0]

def predict_speaker(audio_path):
    features = extract_audio_features(audio_path)
    return voice_model.predict([features])[0]

def recommend_product(customer_id):
    merged_df = pd.read_csv(MERGED_CSV)
    row = merged_df[merged_df['customer_id'] == customer_id]
    if row.empty:
        raise ValueError(f"No data found for customer_id: {customer_id}")
    row = row.drop(['product', 'customer_id'], axis=1)
    pred = product_model.predict(row)[0]
    return label_encoder.inverse_transform([pred])[0]


def verify_identity(image_path, audio_path):
    face_model = joblib.load('models/face_recognition_model.pkl')
    voice_model = joblib.load('models/voice_verification_model.pkl')

    # Process image
    img = cv2.imread(image_path)
    features_img = extract_histogram_features(img).reshape(1, -1)
    pred_person_from_image = face_model.predict(features_img)[0]

    # Process audio
    features_audio = extract_audio_features(audio_path).reshape(1, -1)
    pred_person_from_audio = voice_model.predict(features_audio)[0]

    print(f"Image predicted: {pred_person_from_image}")
    print(f"Audio predicted: {pred_person_from_audio}")

    if pred_person_from_image == pred_person_from_audio:
        print("✅ Voice matches image identity.")
        return True
    else:
        print("❌ Voice does NOT match image identity.")
        return False


if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    print("Pipeline completed successfully.")
