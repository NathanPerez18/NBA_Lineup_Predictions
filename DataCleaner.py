import os
import pandas as pd
import pickle  # To save the encoder

def process_csv_files(raw_folder, clean_folder, encoder_path="player_encoder.pkl"):
    # Ensure clean folder exists
    os.makedirs(clean_folder, exist_ok=True)

    # Define the allowed columns based on metadata
    allowed_columns = [
        "game", "season", "home_team", "away_team", "starting_min",
        "home_0", "home_1", "home_2", "home_3", "home_4",
        "away_0", "away_1", "away_2", "away_3", "away_4", "outcome"
    ]
    
    # Collect all unique player names first
    all_players = set()
    
    # Scan files for unique player names
    for filename in os.listdir(raw_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(raw_folder, filename)
            df = pd.read_csv(file_path, usecols=allowed_columns)
            for col in ["home_0", "home_1", "home_2", "home_3", "home_4", 
                        "away_0", "away_1", "away_2", "away_3", "away_4"]:
                all_players.update(df[col].dropna().unique())
    
    # Fit LabelEncoder on all unique players
    player_encoder = {name: idx for idx, name in enumerate(sorted(all_players))}

    # Save encoder for later use
    with open(encoder_path, "wb") as f:
        pickle.dump(player_encoder, f)

    # Process each CSV file
    for filename in os.listdir(raw_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(raw_folder, filename)
            df = pd.read_csv(file_path, usecols=allowed_columns)
            
            # Encode player names using the saved mapping
            for col in ["home_0", "home_1", "home_2", "home_3", "home_4", 
                        "away_0", "away_1", "away_2", "away_3", "away_4"]:
                df[col] = df[col].map(player_encoder).fillna(-1).astype(int)  # Fill missing values
            
            # Save the cleaned file
            clean_path = os.path.join(clean_folder, filename)
            df.to_csv(clean_path, index=False)
            print(f"Processed and saved: {clean_path}")

# Define folder paths (adjust if necessary)
raw_folder = "RawData"
clean_folder = "Clean"

# Run the processing function
process_csv_files(raw_folder, clean_folder)
    
   # The script above processes all CSV files in the  RawData  folder, cleans the data, and saves the cleaned files in the  Clean  folder. It also saves a mapping of player names to unique integers using the  LabelEncoder  from  scikit-learn. This mapping is saved in a file named  player_encoder.pkl  for later use. 
    #The script reads the CSV files, extracts the player names, and encodes them using the saved mapping. It then saves the cleaned files in the  Clean  folder. 
    #Step 4: Train a Model to Predict Outcomes 
    #Now that we have cleaned the data, we can train a model to predict the outcome of a game based on the starting lineup of each team. We will use a simple logistic regression model to predict the outcome of the game. 
    #Here is the Python script to train the model: