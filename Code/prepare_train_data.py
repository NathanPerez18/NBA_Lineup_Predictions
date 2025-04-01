import os
import pandas as pd
import pickle

# ---------
# Path setup
# ---------
filtered_folder = "../Filtered"
train_folder = "../Train"
dict_folder = "../Dictionaries"
os.makedirs(train_folder, exist_ok=True)

# ---------
# Load encoding dictionaries
# ---------
with open(os.path.join(dict_folder, "player_encoding.pkl"), "rb") as f:
    player_dict = pickle.load(f)
with open(os.path.join(dict_folder, "season_encoding.pkl"), "rb") as f:
    season_dict = pickle.load(f)
with open(os.path.join(dict_folder, "team_encoding.pkl"), "rb") as f:
    team_dict = pickle.load(f)

print("Successfully loaded all encoding dictionaries!")

# ---------
# Columns to encode
# ---------
player_columns = [f"{side}_{i}" for side in ["home", "away"] for i in range(5)]

# ---------
# Process each filtered file
# ---------
for filename in os.listdir(filtered_folder):
    if filename.endswith(".csv"):
        file_path = os.path.join(filtered_folder, filename)
        print(f"Processing {filename}...")

        df = pd.read_csv(file_path)

        # Encode values using dictionaries
        df["season"] = df["season"].astype(str).map(season_dict)
        df["home_team"] = df["home_team"].map(team_dict)
        df["away_team"] = df["away_team"].map(team_dict)

        for col in player_columns:
            df[col] = df[col].map(player_dict)

        # Warn if any mappings failed
        if df[["season", "home_team", "away_team"] + player_columns].isnull().any().any():
            print(f"Warning: Missing mappings in {filename}")

        # Keep only winning games
        df_winners = df[df["outcome"] == 1].copy()

        # Generate training data
        train_data = []
        for _, row in df_winners.iterrows():
            season = row["season"]
            team = row["home_team"]
            players = [row[col] for col in ["home_0", "home_1", "home_2", "home_3", "home_4"]]

            for i in range(5):
                input_players = players[:i] + players[i+1:]
                target_player = players[i]
                train_data.append([season, team] + input_players + [target_player])

        train_df = pd.DataFrame(train_data, columns=["season", "team", "player_0", "player_1", "player_2", "player_3", "target_player"])

        # Save output
        output_path = os.path.join(train_folder, filename)
        train_df.to_csv(output_path, index=False)
        print(f"Saved training file: {output_path}")

print("All training files processed and saved in the Train folder!")
