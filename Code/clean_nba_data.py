import os
import pandas as pd

# Define folder paths
raw_folder = "../Raw"  # Adjust based on your actual path
filtered_folder = "../Filtered"

# Ensure the Filtered folder exists
os.makedirs(filtered_folder, exist_ok=True)

# Define the required columns to keep
required_columns = [
    "game", "season", "home_team", "away_team", "starting_min",
    "home_0", "home_1", "home_2", "home_3", "home_4",
    "away_0", "away_1", "away_2", "away_3", "away_4", "outcome"
]

# Process each CSV file in the Raw folder
for filename in os.listdir(raw_folder):
    if filename.endswith(".csv"):  # Only process CSV files
        file_path = os.path.join(raw_folder, filename)
        print(f"Processing {filename}...")

        # Load the dataset
        df = pd.read_csv(file_path)

        # Keep only required columns
        df = df[required_columns]

        # Save the cleaned file to the Filtered folder
        clean_path = os.path.join(filtered_folder, filename)
        df.to_csv(clean_path, index=False)
        print(f"✅ Saved cleaned file: {clean_path}")

print("All files have been cleaned and saved in the Filtered folder!")
