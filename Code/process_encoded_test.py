import os
import pandas as pd

# Define paths
encoded_test_folder = "../Encoded_Test"
preprocessed_test_folder = "../Pre-Processed-Test"

# Ensure the pre-processed test folder exists
os.makedirs(preprocessed_test_folder, exist_ok=True)

# Columns to check for missing players (-1)
player_columns = ["home_0", "home_1", "home_2", "home_3", "home_4",
                  "away_0", "away_1", "away_2", "away_3", "away_4"]
team_columns = ["home_team", "away_team"]
season_column = "season"

# ---------------------------
# Step 1: Process Each Encoded Test File
# ---------------------------
for filename in os.listdir(encoded_test_folder):
    if filename.endswith(".csv"):  # Process only CSV files
        file_path = os.path.join(encoded_test_folder, filename)
        print(f"📂 Processing {filename}...")

        # Load encoded test data
        df = pd.read_csv(file_path)

        # ✅ Normalize column names to avoid mismatches
        df.columns = df.columns.str.strip().str.lower()

        # Store the processed rows
        processed_rows = []

        # Iterate over test rows
        for _, row in df.iterrows():
            # Identify which column has the missing player (-1)
            missing_col = None
            for col in player_columns:
                if row[col] == -1:
                    missing_col = col
                    break  # Stop once we find the missing player
            
            if missing_col is None:
                continue  # Skip if no missing player found
            
            # Determine the team associated with the missing player
            team_col = "home_team" if "home_" in missing_col else "away_team"
            team = row[team_col]

            # Get the other 4 players from the same team
            team_players = [col for col in player_columns if team_col.split("_")[0] in col and col != missing_col]
            player_values = row[team_players].values.tolist()

            # Store processed data (season, team, 4 players)
            processed_rows.append([row[season_column], team] + player_values)

        # Convert to DataFrame
        processed_df = pd.DataFrame(processed_rows, columns=["season", "team", "player_0", "player_1", "player_2", "player_3"])

        # Save processed test data
        processed_path = os.path.join(preprocessed_test_folder, filename)
        processed_df.to_csv(processed_path, index=False)
        print(f"✅ Processed test data saved at: {processed_path}")

print("🎯 All test files have been pre-processed and saved!")
