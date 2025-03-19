import os
import pandas as pd
import pickle

# Define file paths
test_file_path = "../Test/professor_test_data.csv"  # Original test file
encoded_test_file_path = "../Test/encoded_test_data.csv"  # Encoded version
player_encoding_path = "player_encoding.pkl"  # Encoding dictionary

# Load player encoding dictionary
with open(player_encoding_path, "rb") as f:
    player_dict = pickle.load(f)

# Create a reverse lookup dictionary to decode predictions
reverse_player_dict = {v: k for k, v in player_dict.items()}

# Load test dataset
test_data = pd.read_csv(test_file_path)

# Identify the missing player column
missing_player_col = []
player_columns = ["home_0", "home_1", "home_2", "home_3", "home_4"]

for index, row in test_data.iterrows():
    for col in player_columns:
        if str(row[col]).strip() == "?":  # If the player name is missing
            missing_player_col.append(col)
            test_data.at[index, col] = -1  # Replace missing player with -1
            break  # Only one missing player per row

# Encode all remaining player names using the dictionary
for col in player_columns + ["away_0", "away_1", "away_2", "away_3", "away_4"]:
    test_data[col] = test_data[col].map(player_dict).fillna(-1).astype(int)  # Encode players, use -1 for unknowns

# Save the processed test file
test_data.to_csv(encoded_test_file_path, index=False)
print(f"✅ Encoded test file saved at: {encoded_test_file_path}")

# Save missing player column info for later
missing_info_path = "../Test/missing_player_positions.csv"
pd.DataFrame({"Game_ID": test_data.index, "Missing_Player_Column": missing_player_col}).to_csv(missing_info_path, index=False)
print(f"✅ Missing player positions saved at: {missing_info_path}")
