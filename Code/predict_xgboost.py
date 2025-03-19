import os
import pandas as pd
import pickle

# Define folder paths
model_folder = "../Model"
test_file_path = "../Test/encoded_test_data.csv"
missing_info_path = "../Test/missing_player_positions.csv"
output_predictions_path = os.path.join(model_folder, "final_predictions.csv")

# Load test data
X_test = pd.read_csv(test_file_path)
missing_info = pd.read_csv(missing_info_path)

# ✅ Ensure only test cases from the 2007 season are used
X_test = X_test[X_test["season"] == 2007]  # Filter test data
filtered_missing_info = missing_info[missing_info["Game_ID"].isin(X_test.index)]

# Drop non-numeric columns from test data
X_test = X_test.select_dtypes(include=["number"])

# Load player encoding dictionary
with open("player_encoding.pkl", "rb") as f:
    player_dict = pickle.load(f)

# Reverse dictionary for decoding predictions
reverse_player_dict = {v: k for k, v in player_dict.items()}

# Initialize final predictions DataFrame
final_predictions = X_test.copy()

print("🔄 Making predictions on test data...")

for i, row in filtered_missing_info.iterrows():
    game_id = row["Game_ID"]
    missing_col = row["Missing_Player_Column"]

    print(f"🟢 Using Model for Game {game_id}, Missing: {missing_col}")

    # Find the corresponding model
    model_index = ["home_0", "home_1", "home_2", "home_3", "home_4"].index(missing_col)
    model_path = os.path.join(model_folder, f"xgboost_model_{model_index}.pkl")

    # Load the correct model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Prepare the input for prediction
    X_sample = X_test.loc[[game_id]].drop(columns=[missing_col])

    # Make prediction
    y_pred = model.predict(X_sample)[0]

    # Convert back to player name
    predicted_player = reverse_player_dict.get(y_pred, f"Unknown_Player_{y_pred}")


    # Ensure the column is of string type before inserting player names
    final_predictions[missing_col] = final_predictions[missing_col].astype(str)
    final_predictions.at[game_id, missing_col] = predicted_player


# Extract only the predicted players and save the final output
final_output = pd.DataFrame({"missing_player": final_predictions[missing_col]})
final_output.to_csv(output_predictions_path, index=False)

print(f"✅ Final predictions saved at: {output_predictions_path}")
