
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def create_artifacts():
    # Define paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'cars_clean.csv')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    # 1. Frequency Encoding
    print("Generating frequency maps...")
    freq_manufacturer = df['manufacturer'].value_counts(normalize=True)
    freq_model = df['model'].value_counts(normalize=True)

    df['manufacturer_freq'] = df['manufacturer'].map(freq_manufacturer)
    df['model_freq'] = df['model'].map(freq_model)

    df = df.drop(columns=['manufacturer', 'model'])

    # 2. One Hot Encoding
    categorical_cols = ['fuel_type', 'gear_type', 'drive_type', 'color', 'category']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # 3. Prepare X
    if 'car_id' in df.columns:
        df = df.drop(columns=['car_id'])
        
    X = df.drop(columns=['price'])
    
    feature_names = X.columns.tolist()

    # 4. Split (to ensure we fit scaler on the exact same training set distribution)
    # Note: We must replicate the split to fit the scaler on X_train exactly as the model saw it.
    y = df['price'] # Dummy y needed for split
    X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Fit Scaler
    print("Fitting scaler on training data...")
    scaler = StandardScaler()
    scaler.fit(X_train)

    # 6. Save Artifacts (ONLY preprocessing artifacts, NOT the model)
    print(f"Saving preprocessing artifacts to {MODELS_DIR}...")
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, 'model_features.pkl'))
    joblib.dump(freq_manufacturer, os.path.join(MODELS_DIR, 'freq_manufacturer.pkl'))
    joblib.dump(freq_model, os.path.join(MODELS_DIR, 'freq_model.pkl'))

    print("Done! Model file was NOT touched.")

if __name__ == "__main__":
    create_artifacts()
