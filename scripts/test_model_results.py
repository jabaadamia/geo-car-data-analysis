import json
import pandas as pd
import joblib

# Load mappings
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE_DIR, 'id_mappings', 'id_mappings.json'), 'r') as f:
    data1 = json.load(f)

with open(os.path.join(BASE_DIR, 'id_mappings', 'model_id_mapping.json'), 'r') as f:
    data2 = json.load(f)

category_map = {item['category_id']: item['title'] for item in data1['vehicle_category_mappings']}
manufacturer_map = {item['man_id']: item['title'].strip() for item in data1['manufacturer_mappings']}
transmission_map = {item['id']: item['title'] for item in data1['transmission_type_mappings']}
color_map = {item['color_id']: item['title'] for item in data1['color_mappings']}
drive_map = {item['drive_type_id']: item['title'] for item in data1['drive_type_mappings']}
model_map = {item['model_id']: item['title'].strip() for item in data2}

# Load JSON test car data
json_file = os.path.join(BASE_DIR, 'id_mappings', 'test_cars.json')
with open(json_file, 'r', encoding='utf-8') as f:
    cars_json = json.load(f)

# Function to convert single car JSON to model row

def json_to_model_row(info, current_year=2025):
    row = {
        "price": info.get("price", 0),
        "car_age": current_year - info.get("prod_year", current_year),
        "engine_volume": info.get("engine_volume", 0) / 1000,  # cc to liters
        "cylinders": info.get("cylinders", 0),
        "car_run_km": info.get("car_run_km", info.get("car_run", 0)),
        "safety_feature_count": sum([
            info.get("abs", False),
            info.get("esd", False),
            info.get("airbags", 0) > 0
        ]),
        "comfort_feature_count": sum([
            info.get("el_windows", False),
            info.get("conditioner", False),
            info.get("leather", False),
            info.get("hydraulics", False),
            info.get("chair_warming", False),
            info.get("climat_control", False),
            info.get("start_stop", False),
            info.get("back_camera", False),
            info.get("has_turbo", False)
        ]),
        # Categorical fields
        "manufacturer": manufacturer_map.get(info.get("man_id"), "Unknown"),
        "model": model_map.get(info.get("model_id"), "Unknown"),
        "fuel_type": transmission_map.get(info.get("fuel_type_id"), "Unknown"),
        "gear_type": transmission_map.get(info.get("gear_type_id"), "Unknown"),
        "drive_type": drive_map.get(info.get("drive_type_id"), "Unknown"),
        "category": category_map.get(info.get("category_id"), "Unknown"),
        "color": color_map.get(info.get("color_id"), "Unknown"),
        # Binary features
        "abs": int(info.get("abs", False)),
        "esd": int(info.get("esd", False)),
        "el_windows": int(info.get("el_windows", False)),
        "conditioner": int(info.get("conditioner", False)),
        "leather": int(info.get("leather", False)),
        "hydraulics": int(info.get("hydraulics", False)),
        "chair_warming": int(info.get("chair_warming", False)),
        "climat_control": int(info.get("climat_control", False)),
        "start_stop": int(info.get("start_stop", False)),
        "back_camera": int(info.get("back_camera", False)),
        "has_turbo": int(info.get("has_turbo", False)),
        "right_wheel": int(info.get("right_wheel", False)),
        "tech_inspection": int(info.get("tech_inspection", False)),
        "customs_passed": int(info.get("customs_passed", False)),
        "is_dealer": int(info.get("dealer_user_id", 0) > 0),
        "is_spec": int(info.get("special_persons", False)),
    }
    
    return pd.DataFrame([row])

# Convert all cars to DataFrame

df_list = [json_to_model_row(car) for car in cars_json]
df_new = pd.concat(df_list, ignore_index=True)

# Align with model's training features and Scale

# Load artifacts
models_dir = os.path.join(BASE_DIR, 'models')
feature_columns = joblib.load(os.path.join(models_dir, 'model_features.pkl'))
scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
freq_man = joblib.load(os.path.join(models_dir, 'freq_manufacturer.pkl'))
freq_model = joblib.load(os.path.join(models_dir, 'freq_model.pkl'))
model_rf = joblib.load(os.path.join(models_dir, 'random_forest_model.pkl'))

# Feature Engineering: Frequency Encoding
# Map manufacturer and model to their frequencies. 
# Use 0 for unknown values (not seen in training).

df_new['manufacturer_freq'] = df_new['manufacturer'].map(freq_man).fillna(0)
df_new['model_freq'] = df_new['model'].map(freq_model).fillna(0)

# Drop original categorical columns that were dropped in training
df_new_processed = df_new.drop(columns=['manufacturer', 'model'])

# One-hot encode others with drop_first=True (matching training)
categorical_cols = ['fuel_type', 'gear_type', 'drive_type', 'color', 'category']
df_new_encoded = pd.get_dummies(df_new_processed, columns=categorical_cols, drop_first=True)

# Reindex to ensure all columns match training data, filling missing with 0
df_new_encoded = df_new_encoded.reindex(columns=feature_columns, fill_value=0)

# Scale
X_scaled = scaler.transform(df_new_encoded)

# predict

predicted_prices = model_rf.predict(X_scaled)
df_new["predicted_price"] = predicted_prices
df_new["prediction_error"] = df_new["price"] - df_new["predicted_price"]
df_new["prediction_error"] = df_new["prediction_error"].abs()

print(df_new[["manufacturer", "model", "car_age", "predicted_price", "price", "prediction_error"]])
