import os
import pandas as pd
import pickle
import xgboost as xgb

# Define paths
model_path = "../Model/xgboost_model_v2.pkl"
preprocessed_test_folder = "../Pre-Processed-Test"
dictionary_folder = "../Dictionaries"
output_predictions_path = "../Model/final_predictions.csv"

# Load encoding dictionaries
with open(os.path.join(dictionary_folder, "player_encoding.pkl"), "rb") as f:
    player_dict = pickle.load(f)

# Reverse dictionary for decoding predictions
reverse_player_dict = {v: k for k, v in player_dict.items()}

# Load trained model
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Process test files
final_predictions = []

for filename in os.listdir(preprocessed_test_folder):
    if filename.endswith(".csv"):  # Process only CSV files
        file_path = os.path.join(preprocessed_test_folder, filename)
        print(f"📂 Making predictions for {filename}...")

        # Load preprocessed test data
        df = pd.read_csv(file_path)

        # ✅ Ensure test data is not empty
        if df.empty:
            print(f"⚠ Warning: {filename} is empty. Skipping...")
            continue

        # ✅ Ensure columns match the training format
        required_columns = ["season", "team", "player_0", "player_1", "player_2", "player_3"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"❌ ERROR: Missing columns {missing_columns} in {filename}. Skipping...")
            continue

        # ✅ Define `X_test`
        X_test = df[required_columns]


        # Make predictions
        y_pred = model.predict(X_test)

        # Convert predictions back to player names
        predicted_players = [reverse_player_dict.get(int(player), "Unknown Player") for player in y_pred]

        # Store results
        for i, row in df.iterrows():
            final_predictions.append([predicted_players[i]])

# Convert to DataFrame
final_df = pd.DataFrame(final_predictions, columns=["missing_player"])

# Save predictions
final_df.to_csv(output_predictions_path, index=False)
print(f"✅ Final predictions saved at: {output_predictions_path}")
