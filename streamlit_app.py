import streamlit as st
import pandas as pd
import json
import joblib
import os

# Page Config
st.set_page_config(
    page_title="Car Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for aesthetics
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        height: 3rem;
        font-weight: bold;
    }
    .stMetric {
        background-color: #000;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    h1 {
        color: #1f1f1f;
    }
    h2 {
        color: #3f3f3f;
        font-size: 1.5rem;
        margin-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Paths
    models_dir = os.path.join(base_dir, 'models')
    mappings_dir = os.path.join(base_dir, 'id_mappings')
    
    # Load Models & Artifacts
    feature_columns = joblib.load(os.path.join(models_dir, 'model_features.pkl'))
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    freq_man = joblib.load(os.path.join(models_dir, 'freq_manufacturer.pkl'))
    freq_model = joblib.load(os.path.join(models_dir, 'freq_model.pkl'))
    model_rf = joblib.load(os.path.join(models_dir, 'random_forest_model.pkl'))
    
    # Load Mappings JSON
    with open(os.path.join(mappings_dir, 'id_mappings.json'), 'r', encoding="utf-8") as f:
        id_mappings = json.load(f)
        
    return feature_columns, scaler, freq_man, freq_model, model_rf, id_mappings

try:
    feature_columns, scaler, freq_man, freq_model, model_rf, id_mappings = load_data()
    
    # Process Mappings for Dropdowns
    # Extract unique titles for dropdowns
    
    # Manufacturers from frequency dict keys (assures model validity)
    manufacturers = sorted(list(freq_man.keys()))
    
    # Models from frequency dict keys
    car_models = sorted(list(freq_model.keys()))
    
    # Mappings from JSON
    # Structure: item['title']
    fuel_types = sorted([item['title'] for item in id_mappings.get('fuel_type_mappings', [])])
    gear_types = sorted([item['title'] for item in id_mappings.get('transmission_type_mappings', [])])
    drive_types = sorted([item['title'] for item in id_mappings.get('drive_type_mappings', [])])
    colors = sorted([item['title'] for item in id_mappings.get('color_mappings', [])])
    categories = sorted([item['title'] for item in id_mappings.get('vehicle_category_mappings', [])])
    
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# UI layout

st.title("price prediction on georgian car market")
st.markdown("Enter the vehicle details below to get an estimated market price based on our machine learning model.")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("General Info")
    selected_man = st.selectbox("Manufacturer", manufacturers)
    selected_model = st.selectbox("Model", car_models)
    
    category = st.selectbox("Category", categories)
    prod_year = st.number_input("Production Year", min_value=1950, max_value=2025, value=2015, step=1)
    color = st.selectbox("Color", colors)

with col2:
    st.subheader("Technical Specs")
    fuel = st.selectbox("Fuel Type", fuel_types)
    gear = st.selectbox("Gearbox Type", gear_types)
    drive = st.selectbox("Drive Type", drive_types)
    
    engine_vol = st.number_input("Engine Volume (Liters)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
    cylinders = st.number_input("Cylinders", min_value=1, max_value=16, value=4, step=1)
    mileage_km = st.number_input("Mileage (km)", min_value=0, max_value=2000000, value=50000, step=1000)
    
    has_turbo = st.checkbox("Turbo Engine", value=False)
    right_wheel = st.checkbox("Right Wheel (RHD)", value=False)

with col3:
    st.subheader("Features & Conditions")
    
    with st.expander("Comfort Features", expanded=True):
        f_el_windows = st.checkbox("Electric Windows", value=True)
        f_conditioner = st.checkbox("Air Conditioning", value=True)
        f_leather = st.checkbox("Leather Interior", value=False)
        f_hydraulics = st.checkbox("Hydraulics", value=False)
        f_chair_warming = st.checkbox("Heated Seats", value=False)
        f_climat_control = st.checkbox("Climate Control", value=False)
        f_start_stop = st.checkbox("Start/Stop System", value=False)
        f_back_camera = st.checkbox("Back Camera", value=False)
    
    with st.expander("Safety features", expanded=True):
        f_abs = st.checkbox("ABS", value=True)
        f_esd = st.checkbox("ESD (Electronic Stability)", value=True)
        f_airbags = st.checkbox("Airbags", value=True) # Used for count logic
    
    with st.expander("Status"):
        f_tech_inspection = st.checkbox("Tech Inspection Passed", value=True)
        f_customs_passed = st.checkbox("Customs Cleared", value=True)
        f_is_dealer = st.checkbox("Dealer Seller", value=False)
        f_is_spec = st.checkbox("Special Vehicle", value=False)


# Prediction logic

if st.button("Calculate Price"):
    
    # 1. Feature Construction
    current_year = 2025
    car_age = current_year - prod_year
    
    # Counts
    safety_count = sum([f_abs, f_esd, f_airbags])
    comfort_count = sum([
        f_el_windows, f_conditioner, f_leather, f_hydraulics, 
        f_chair_warming, f_climat_control, f_start_stop, 
        f_back_camera
    ])
    
    # Form Dictionary matching 'json_to_model_row' keys basically, but we build DataFrame directly
    data = {
        "car_age": [car_age],
        "engine_volume": [engine_vol], 
        "cylinders": [cylinders],
        "car_run_km": [mileage_km],
        "safety_feature_count": [safety_count],
        "comfort_feature_count": [comfort_count],
        
        # Raw Categorical Columns (for encoding)
        "manufacturer": [selected_man],
        "model": [selected_model],
        "fuel_type": [fuel],
        "gear_type": [gear],
        "drive_type": [drive],
        "color": [color],
        "category": [category],
        
        # Binary Features
        "abs": [int(f_abs)],
        "esd": [int(f_esd)],
        "el_windows": [int(f_el_windows)],
        "conditioner": [int(f_conditioner)],
        "leather": [int(f_leather)],
        "hydraulics": [int(f_hydraulics)],
        "chair_warming": [int(f_chair_warming)],
        "climat_control": [int(f_climat_control)],
        "start_stop": [int(f_start_stop)],
        "back_camera": [int(f_back_camera)],
        "has_turbo": [int(has_turbo)],
        "right_wheel": [int(right_wheel)],
        "tech_inspection": [int(f_tech_inspection)],
        "customs_passed": [int(f_customs_passed)],
        "is_dealer": [int(f_is_dealer)],
        "is_spec": [int(f_is_spec)],
    }
    
    df = pd.DataFrame(data)
    
    # 2. Feature Engineering
    
    # Frequency Encoding
    # Map raw strings to frequencies from artifacts. Default 0.
    df['manufacturer_freq'] = df['manufacturer'].map(freq_man).fillna(0)
    df['model_freq'] = df['model'].map(freq_model).fillna(0)
    
    # Drop originals
    df_processed = df.drop(columns=['manufacturer', 'model'])
    
    # One-Hot Encoding
    categorical_cols = ['fuel_type', 'gear_type', 'drive_type', 'color', 'category']
    df_encoded = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
    
    # Reindex to match training columns
    # This aligns columns and fills missing dummy columns (e.g. if we selected 'Petrol' but model has 'Petrol' column it keeps it,
    # if we select something unseen it fills everything 0)
    df_final = df_encoded.reindex(columns=feature_columns, fill_value=0)
    
    # 3. Scaling
    try:
        X_scaled = scaler.transform(df_final)
        
        # 4. Predict
        prediction = model_rf.predict(X_scaled)[0]
        
        # Display
        st.success("Prediction Complete!")
        st.metric(label="Estimated Price", value=f"${prediction:,.2f}")
        
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.write("Debug info - Shape:", df_final.shape)
        st.write("Debug info - Columns:", df_final.columns.tolist())
