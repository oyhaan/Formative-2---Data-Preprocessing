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

IMAGE_FOLDER = 'images_data'
AUDIO_FOLDER = 'audio_data'
IMAGE_FEATURES_CSV = 'image_features.csv'
AUDIO_FEATURES_CSV = 'audio_features.csv'
SOCIAL_PROFILE_CSV = 'data/customer_social_profiles.csv'
TRANSACTION_CSV = 'data/customer_transactions.csv'
MERGED_CSV = 'data/merged_data.csv'

def augment_image(img):
    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    flipped = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return [rotated, flipped, gray_rgb]

def extract_histogram_features(img):
    chans = cv2.split(img)
    hist = np.concatenate([
        cv2.calcHist([c], [0], None, [256], [0, 256]).flatten()
        for c in chans
    ])
    return hist

def process_images():
    records = []
    for file in os.listdir(IMAGE_FOLDER):
        if not file.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        img_path = os.path.join(IMAGE_FOLDER, file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        person = file.split("-")[0]
        expression = file.split("-")[1].split(".")[0]
        variations = [("original", img)] + list(zip(["rotated", "flipped", "grayscale"], augment_image(img)))
        for label, variant_img in variations:
            features = extract_histogram_features(variant_img)
            record = {
                'person': person,
                'expression': expression,
                'variation': label,
                **{f'feat_{i}': val for i, val in enumerate(features)}
            }
            records.append(record)
    pd.DataFrame(records).to_csv(IMAGE_FEATURES_CSV, index=False)

def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    return np.mean(mfcc, axis=1)

def process_audio():
    records = []
    for file in os.listdir(AUDIO_FOLDER):
        if not file.endswith('.wav'):
            continue
        person, label = file.replace(".wav", "").split("-")
        file_path = os.path.join(AUDIO_FOLDER, file)
        try:
            features = extract_audio_features(file_path)
            record = {
                'person': person,
                'phrase': label,
                **{f'mfcc_{i}': val for i, val in enumerate(features)}
            }
            records.append(record)
        except Exception as e:
            print(f"Error processing {file}: {e}")
    pd.DataFrame(records).to_csv(AUDIO_FEATURES_CSV, index=False)

def merge_tabular_data():
    df1 = pd.read_csv(SOCIAL_PROFILE_CSV)
    df2 = pd.read_csv(TRANSACTION_CSV)
    merged = pd.merge(df1, df2, on='customer_id')
    merged.to_csv(MERGED_CSV, index=False)

def train_models():
    # Face recognition
    img_df = pd.read_csv(IMAGE_FEATURES_CSV)
    X_face = img_df.filter(like='feat_')
    y_face = img_df['person']
    face_model = RandomForestClassifier().fit(X_face, y_face)
    joblib.dump(face_model, 'models/face_recognition_model.pkl')

    # Voice recognition
    audio_df = pd.read_csv(AUDIO_FEATURES_CSV)
    X_audio = audio_df.filter(like='mfcc_')
    y_audio = audio_df['phrase']
    voice_model = RandomForestClassifier().fit(X_audio, y_audio)
    joblib.dump(voice_model, 'models/voiceprint_verification_model.pkl')

    # Product prediction
    merged_df = pd.read_csv(MERGED_CSV)
    merged_df = merged_df.dropna()
    label_encoder = LabelEncoder()
    merged_df['product'] = label_encoder.fit_transform(merged_df['product'])
    X = merged_df.drop(['product', 'customer_id'], axis=1)
    y = merged_df['product']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = RandomForestClassifier().fit(X_train, y_train)
    preds = clf.predict(X_test)
    print("Product Model Accuracy:", accuracy_score(y_test, preds))
    print("Product Model F1 Score:", f1_score(y_test, preds, average='weighted'))
    joblib.dump(clf, 'models/product_recommendation_model.pkl')

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    process_images()
    process_audio()
    merge_tabular_data()
    train_models()
    print("Pipeline completed successfully.")
