import os
import pandas as pd
import pickle

# Define paths
filtered_folder = "../Filtered"  # Where the cleaned CSVs are stored
output_folder = "../Dictionaries"  # Where we'll save the dictionaries
os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist

# Encoding dictionary storage
season_dict = {}  # Season encoding
team_dict = {}  # Team encoding
player_dict = {}  # Player encoding

season_index = 0
team_index = 0
player_index = 0

# Define columns
player_columns = ["home_0", "home_1", "home_2", "home_3", "home_4",
                  "away_0", "away_1", "away_2", "away_3", "away_4"]
team_columns = ["home_team", "away_team"]
season_column = "season"

# Read all files in the Filtered folder
for filename in os.listdir(filtered_folder):
    if filename.endswith(".csv"):
        file_path = os.path.join(filtered_folder, filename)
        print(f"Processing {filename}...")

        # Load data
        df = pd.read_csv(file_path)

        # Encode seasons
        for season in df[season_column].unique():
            if season not in season_dict:
                season_dict[season] = season_index
                season_index += 1

        # Encode teams
        for col in team_columns:
            for team in df[col].unique():
                if team not in team_dict:
                    team_dict[team] = team_index
                    team_index += 1

        # Encode players
        for col in player_columns:
            for player in df[col].unique():
                if player not in player_dict:
                    player_dict[player] = player_index
                    player_index += 1

# Save encoding dictionaries
with open(os.path.join(output_folder, "season_encoding.pkl"), "wb") as f:
    pickle.dump(season_dict, f)
print(f"✅ Season dictionary saved with {len(season_dict)} entries.")

with open(os.path.join(output_folder, "team_encoding.pkl"), "wb") as f:
    pickle.dump(team_dict, f)
print(f"✅ Team dictionary saved with {len(team_dict)} entries.")

with open(os.path.join(output_folder, "player_encoding.pkl"), "wb") as f:
    pickle.dump(player_dict, f)
print(f"✅ Player dictionary saved with {len(player_dict)} entries.")
