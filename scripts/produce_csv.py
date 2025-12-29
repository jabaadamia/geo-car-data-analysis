"""
produces csv file of about 60k cars from raw jl file
then, removes unnecessary columns.
input: cars_data.jl with 118 cols
output: cars_60k.csv with 31 cols
"""

import json
import csv
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "raw" / "cars_data.jl"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "cars_60k.csv"

MAX_ROWS = 100_000
EXCLUDED_LOCATION_ID = 21 # remove cars located in USA
MOTO_ID = 2 # vehicle type id for motorcycles

def flatten_record(record: dict) -> dict:
    try:
        return record["inner_api_call"]["info"]
    except KeyError:
        return None

def jl_to_csv_sample(input_path: Path, output_path: Path, max_rows: int):
    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    fieldnames = None

    if not input_path.exists():
        print(f"Error: Could not find input file at {input_path}")
        return

    with input_path.open("r", encoding="utf-8") as jl_file, \
         output_path.open("w", encoding="utf-8", newline="") as csv_file:

        writer = None

        for line_number, line in enumerate(jl_file, start=1):
            if rows_written >= max_rows:
                break

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            flat = flatten_record(record)
            if flat is None:
                continue

            # Filter out location_id 21
            if flat.get("location_id") == EXCLUDED_LOCATION_ID:
                continue

            # Filter out rentals
            if flat.get("for_rent", False):
                continue

            # Filter out motorcycles
            if flat.get("vehicle_type") == MOTO_ID:
                continue
        

            if writer is None:
                fieldnames = list(flat.keys())
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()


            row = {k: flat.get(k) for k in fieldnames}
            writer.writerow(row)
            rows_written += 1

            if rows_written % 5000 == 0:
                print(f"{rows_written} rows written...")

    print(f"Done! {rows_written} rows saved to: {output_path}")

good_cols = [
    'car_id', 
    'prod_year',
    'man_id',
    'model_id',
    'price',
    'predicted_price',
    'fuel_type_id', 
    'gear_type_id', 
    'drive_type_id',
    'color_id',
    'cylinders',
    'car_run_km',
    'engine_volume',
    'abs', 
    'esd', 
    'el_windows', 
    'conditioner', 
    'leather',
    'hydraulics', 
    'chair_warming', 
    'climat_control',
    'customs_passed',
    'tech_inspection',
    'has_turbo',
    'right_wheel', 
    'vehicle_type', # 0= cars, 1= spec
    'category_id',
    'start_stop',
    'back_camera',
    'user_type',  # 1= dealer, 0= private
    'comfort_features',
]

# transform the full csv to only selected columns
def select_base_cols(input_csv: Path, output_csv: Path):
    df = pd.read_csv(input_csv, low_memory=False)
    df = df[good_cols]
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    jl_to_csv_sample(INPUT_FILE, OUTPUT_FILE, MAX_ROWS)
    select_base_cols(OUTPUT_FILE, BASE_DIR / "data" / "processed" / "cars_60k.csv")