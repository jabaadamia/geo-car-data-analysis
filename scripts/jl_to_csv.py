import json
import csv
from pathlib import Path

INPUT_FILE = "cars_data.jl"
OUTPUT_FILE = "cars_60k.csv"
MAX_ROWS = 60_000


def flatten_record(record: dict) -> dict:
    """
    Extract and flatten inner_api_call.info.
    Returns None if structure is invalid.
    """
    try:
        return record["inner_api_call"]["info"]
    except KeyError:
        return None


def jl_to_csv_sample(input_path: str, output_path: str, max_rows: int):
    input_path = Path(input_path)
    output_path = Path(output_path)

    rows_written = 0
    fieldnames = None

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

            # Initialize CSV writer once we see first valid row
            if writer is None:
                fieldnames = list(flat.keys())
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()

            # Keep only known columns (safety)
            row = {k: flat.get(k) for k in fieldnames}
            writer.writerow(row)
            rows_written += 1

            if rows_written % 5000 == 0:
                print(f"{rows_written} rows written...")

    print(f"Done. {rows_written} rows saved to {output_path}")


if __name__ == "__main__":
    jl_to_csv_sample(INPUT_FILE, OUTPUT_FILE, MAX_ROWS)
