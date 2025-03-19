import os
import pandas as pd
import pickle  # To save the player encoding dictionary

# Define folder paths
filtered_folder = "../Filtered"  # Source folder with cleaned CSV files
encoded_folder = "../Encoded"  # Destination folder for encoded files
os.makedirs(encoded_folder, exist_ok=True)  # Create the folder if it doesn't exist

# Initialize an empty dictionary for player encoding
player_dict = {}
player_counter = 0  # Counter for assigning unique numbers

# Columns that contain player names
player_columns = ["home_0", "home_1", "home_2", "home_3", "home_4", 
                  "away_0", "away_1", "away_2", "away_3", "away_4"]

### Step 1: Scan all CSV files to collect unique player names ###
for filename in os.listdir(filtered_folder):
    if filename.endswith(".csv"):  # Only process CSV files
        file_path = os.path.join(filtered_folder, filename)
        df = pd.read_csv(file_path)
        
        # Loop through all player columns and add unseen names to the dictionary
        for col in player_columns:
            for player in df[col].dropna().unique():  # Drop NaN values (missing data)
                if player not in player_dict:
                    player_dict[player] = player_counter
                    player_counter += 1  # Assign the next available number

# Save the player encoding dictionary for future use
with open("player_encoding.pkl", "wb") as f:
    pickle.dump(player_dict, f)
print(f"✅ Saved player encoding dictionary with {len(player_dict)} players!")

### Step 2: Replace player names with their encoded values in all files ###
for filename in os.listdir(filtered_folder):
    if filename.endswith(".csv"):
        file_path = os.path.join(filtered_folder, filename)
        df = pd.read_csv(file_path)

        # Replace player names with their encoded values
        for col in player_columns:
            df[col] = df[col].map(player_dict).fillna(-1).astype(int)  # Use -1 for missing players
        
        # Save the encoded file to the new Encoded folder
        encoded_path = os.path.join(encoded_folder, filename)
        df.to_csv(encoded_path, index=False)
        print(f"✅ Encoded and saved: {encoded_path}")

print("All CSV files have been encoded and saved in the 'Encoded' folder!")
