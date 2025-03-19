import os
import pandas as pd
import pickle

# Define paths
test_folder = "../Test"
encoded_test_folder = "../Encoded_Test"
dictionary_folder = "../Dictionaries"

# Ensure the encoded test folder exists
os.makedirs(encoded_test_folder, exist_ok=True)

# Load encoding dictionaries
with open(os.path.join(dictionary_folder, "season_encoding.pkl"), "rb") as f:
    season_dict = pickle.load(f)

with open(os.path.join(dictionary_folder, "team_encoding.pkl"), "rb") as f:
    team_dict = pickle.load(f)

with open(os.path.join(dictionary_folder, "player_encoding.pkl"), "rb") as f:
    player_dict = pickle.load(f)

# Reverse dictionaries for decoding (if needed)
reverse_player_dict = {v: k for k, v in player_dict.items()}
reverse_team_dict = {v: k for k, v in team_dict.items()}
reverse_season_dict = {v: k for k, v in season_dict.items()}

# Columns to process
player_columns = ["home_0", "home_1", "home_2", "home_3", "home_4",
                  "away_0", "away_1", "away_2", "away_3", "away_4"]
team_columns = ["home_team", "away_team"]
season_column = "season"

# ---------------------------
# Step 2: Process Each Test File
# ---------------------------
for filename in os.listdir(test_folder):
    if filename.endswith("a.csv"):  # Process only CSV files
        file_path = os.path.join(test_folder, filename)
        print(f"📂 Processing {filename}...")

        # Load test data
        df = pd.read_csv(file_path)

        # ✅ Normalize column names to avoid KeyErrors
        df.columns = df.columns.str.strip().str.lower()

        # Debugging step to confirm column names
        print(f"🔍 Columns found: {df.columns.tolist()}")

        # Drop unwanted columns
        if "starting_min" in df.columns:
            df = df.drop(columns=["starting_min"])

        # Encode the season column
        if season_column in df.columns:
            df[season_column] = df[season_column].map(season_dict)

        # Encode team columns
        for col in team_columns:
            df[col] = df[col].map(team_dict)

        # Encode player columns (handle missing player '?')
        for col in player_columns:
            df[col] = df[col].map(player_dict).fillna(-1)  # -1 for missing players

        # Save the encoded test data
        encoded_path = os.path.join(encoded_test_folder, filename)
        df.to_csv(encoded_path, index=False)
        print(f"✅ Encoded test data saved at: {encoded_path}")

# Save updated dictionaries
with open(os.path.join(dictionary_folder, "player_encoding.pkl"), "wb") as f:
    pickle.dump(player_dict, f)

with open(os.path.join(dictionary_folder, "team_encoding.pkl"), "wb") as f:
    pickle.dump(team_dict, f)

print("✅ Updated dictionaries saved with new players and teams!")

print("🎯 All test files have been processed and saved!")
