import os
import pandas as pd
import pickle

# ---------
# Paths
# ---------
test_folder = "../Test"
encoded_test_folder = "../Encoded_Test"
dictionary_folder = "../Dictionaries"
os.makedirs(encoded_test_folder, exist_ok=True)

# ---------
# Load dictionaries
# ---------
with open(os.path.join(dictionary_folder, "season_encoding.pkl"), "rb") as f:
    season_dict = pickle.load(f)
with open(os.path.join(dictionary_folder, "team_encoding.pkl"), "rb") as f:
    team_dict = pickle.load(f)
with open(os.path.join(dictionary_folder, "player_encoding.pkl"), "rb") as f:
    player_dict = pickle.load(f)

# ---------
# Columns to encode
# ---------
season_column = "season"
team_columns = ["home_team", "away_team"]
player_columns = [f"{side}_{i}" for side in ["home", "away"] for i in range(5)]

# ---------
# Encode professor_test_data.csv
# ---------
filename = "professor_test_data.csv"
file_path = os.path.join(test_folder, filename)

if os.path.exists(file_path):
    print(f"📂 Processing {filename}...")

    df = pd.read_csv(file_path)

    # Remove unwanted columns
    if "starting_min" in df.columns:
        df = df.drop(columns=["starting_min"])

    # Encode season
    df[season_column] = df[season_column].astype(str).map(season_dict)

    # Encode teams
    for col in team_columns:
        df[col] = df[col].map(team_dict)

    # Encode players
    for col in player_columns:
        df[col] = df[col].map(player_dict).fillna(-1)  # Use -1 for any unknowns

    # Check for encoding errors
    if df[["season"] + team_columns + player_columns].isnull().any().any():
        print("⚠️ Warning: Some values could not be encoded (check your dictionaries)")

    # Save output
    encoded_path = os.path.join(encoded_test_folder, filename)
    df.to_csv(encoded_path, index=False)
    print(f"✅ Encoded test data saved at: {encoded_path}")
else:
    print(f"❌ Test file not found: {file_path}")

print("🎯 Test encoding complete.")
