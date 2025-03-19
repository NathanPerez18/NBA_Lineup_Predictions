import os
import pandas as pd
import pickle

# Define folder paths
filtered_folder = "../Filtered"
train_folder = "../Train"

# Define the correct path to the Dictionaries folder
dict_folder = "../Dictionaries"

# Load encoding dictionaries with full paths
player_encoding_path = os.path.join(dict_folder, "player_encoding.pkl")
season_encoding_path = os.path.join(dict_folder, "season_encoding.pkl")
team_encoding_path = os.path.join(dict_folder, "team_encoding.pkl")

# Ensure files exist before loading
if not os.path.exists(player_encoding_path):
    raise FileNotFoundError(f"❌ Missing file: {player_encoding_path}")
if not os.path.exists(season_encoding_path):
    raise FileNotFoundError(f"❌ Missing file: {season_encoding_path}")
if not os.path.exists(team_encoding_path):
    raise FileNotFoundError(f"❌ Missing file: {team_encoding_path}")

# Load the dictionaries
with open(player_encoding_path, "rb") as f:
    player_dict = pickle.load(f)

with open(season_encoding_path, "rb") as f:
    season_dict = pickle.load(f)

with open(team_encoding_path, "rb") as f:
    team_dict = pickle.load(f)

print("✅ Successfully loaded all encoding dictionaries!")




# Columns to process
player_columns = ["home_0", "home_1", "home_2", "home_3", "home_4"]

# Process each CSV file in the Filtered folder
for filename in os.listdir(filtered_folder):
    if filename.endswith(".csv"):  # Only process CSV files
        file_path = os.path.join(filtered_folder, filename)
        print(f"Processing {filename}...")

        # Load the dataset
        df = pd.read_csv(file_path)

        # Encode season, teams, and players
        df["season"] = df["season"].map(season_dict)
        df["home_team"] = df["home_team"].map(team_dict)
        df["away_team"] = df["away_team"].map(team_dict)

        for col in player_columns:
            df[col] = df[col].map(player_dict)

        # Keep only the winning team (outcome = 1)
        df_winners = df[df["outcome"] == 1].copy()

        # Transform each winning game into 5 training rows (one missing player at a time)
        train_data = []
        for _, row in df_winners.iterrows():
            season = row["season"]
            team = row["home_team"]
            players = [row[col] for col in player_columns]

            # Create 5 rows, each with 4 players and the missing player as the target
            for i in range(5):
                input_players = players[:i] + players[i+1:]  # Exclude the i-th player
                target_player = players[i]  # The missing player
                
                train_data.append([season, team] + input_players + [target_player])

        # Convert to DataFrame
        train_df = pd.DataFrame(train_data, columns=["season", "team", "player_0", "player_1", "player_2", "player_3", "target_player"])

        # Save the cleaned training data
        train_file_path = os.path.join(train_folder, filename)
        train_df.to_csv(train_file_path, index=False)
        print(f"✅ Saved training file: {train_file_path}")

print("All training files have been processed and saved in the Train folder!")
