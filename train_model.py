import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

FEATURES = [
    "koi_score", "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_prad",
    "koi_teq", "koi_duration", "koi_depth", "koi_insol", "koi_model_snr",
    "koi_steff", "koi_slogg", "koi_srad", "koi_kepmag"
]
LABEL_COL = "koi_disposition"
label_map = {"CONFIRMED": 1, "CANDIDATE": 0, "FALSE POSITIVE": -1}

def train_and_save_model(data_path):
    df = pd.read_csv(data_path)
    df = df.dropna(subset=FEATURES + [LABEL_COL])
    df[LABEL_COL] = df[LABEL_COL].str.strip()
    X = df[FEATURES]
    y = df[LABEL_COL].map(label_map)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = RandomForestClassifier(n_estimators=150, random_state=42)
    clf.fit(X_scaled, y)

    joblib.dump(scaler, "scaler.joblib")
    joblib.dump(clf, "model.joblib")

    print("Training complete. Model saved as 'model.joblib' and scaler as 'scaler.joblib'.")

if __name__ == "__main__":
    train_and_save_model("a.csv")
