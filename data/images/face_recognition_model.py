import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, log_loss, classification_report
import joblib

IMAGE_FEATURES_PATH = "image_features.csv"
CUSTOMER_DATA_PATH = "../tables/merged_customer_data.csv"
MODEL_OUTPUT_PATH = "face_recognition_model.pkl"

features_df = pd.read_csv(IMAGE_FEATURES_PATH)
customer_df = pd.read_csv(CUSTOMER_DATA_PATH)

user_id_map = {
    'owen': 187,
    'nicolas': 177,
    'abiodun': 189,
    'gaius': 120,
    'anissa': 103
}
features_df['user_id'] = features_df['person'].map(user_id_map)

features_df = features_df.dropna(subset=['user_id'])

X = features_df.drop(columns=['person', 'expression', 'variation', 'user_id'])
y = features_df['user_id'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

accuracy = model.score(X_test, y_test)
f1 = f1_score(y_test, y_pred, average='weighted')
loss = log_loss(y_test, y_proba)

print(f"Test Accuracy: {accuracy:.2f}")
print(f"Weighted F1-Score: {f1:.2f}")
print(f"Log Loss: {loss:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

joblib.dump(model, MODEL_OUTPUT_PATH)
print(f"Model saved to: {MODEL_OUTPUT_PATH}")
