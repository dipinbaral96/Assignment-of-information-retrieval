import ujson
import pandas as pd
import numpy as np
import nltk
import os

# NLTK package for stop words
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# File paths
DATA_FILE = "./files/res.json"
OUTPUT_FILE = "./files/train_test.csv"

# Load scraped data
try:
    with open(DATA_FILE, "r", encoding='utf-8') as file:
        data = ujson.load(file)

    print(f"Data type: {type(data)}")
    print(f"First 500 characters: {str(data)[:500]}")

    # Extract publications list if the JSON is a dictionary
    if isinstance(data, dict) and "publications" in data:
        data = data["publications"]

    if not isinstance(data, list) or len(data) == 0:
        print("Warning: Invalid or empty data in res.json!")
        exit(1)

    # Convert to DataFrame
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} publications from res.json")

except FileNotFoundError:
    print("Error: res.json file not found!")
    exit(1)
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# Ensure required columns exist before processing
required_columns = ["title", "authors", "year", "abstract"]
for col in required_columns:
    if col not in df.columns:
        df[col] = ""

# Fill missing values
df["title"] = df["title"].fillna("Unknown Title")
df["authors"] = df["authors"].apply(lambda x: x if isinstance(x, list) else [])
df["year"] = df["year"].fillna("Unknown Year")
df["abstract"] = df["abstract"].fillna("No abstract available")

# Convert authors list to a string format for easier processing
df["authors"] = df["authors"].apply(lambda x: "; ".join([a["name"] for a in x]) if isinstance(x, list) else "Unknown Author")

# Convert years to integers where possible
try:
    df["year"] = df["year"].astype(str).str.extract(r"(\d{4})")[0]  # Extract year format (YYYY)
    df["year"] = pd.to_numeric(df["year"], errors='coerce').fillna(0).astype(int)
except Exception as e:
    print(f"Error converting years: {e}")

# Save processed data
df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved processed data with {len(df)} rows to {OUTPUT_FILE}")
