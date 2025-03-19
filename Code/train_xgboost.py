import os
import pandas as pd
import pickle
import xgboost as xgb

# Define file paths
model_folder = "../Model"
test_file_path = "../Test/encoded_test_data.csv"
missing_info_path = "../Test/missing_player_positions.csv"
output_predictions_path = os.path.join(model_folder, "final_predictions.csv")

# Load the player encoding dictionary
with open("player_encoding.pkl", "rb") as f:
    player_dict = pickle.load(f)

# Reverse dictionary for decoding predictions
reverse_player_dict = {v: k for k, v in player_dict.items()}

# Load training data
X_train = pd.read_csv(os.path.join(model_folder, "X_train.csv"))
y_train = pd.read_csv(os.path.join(model_folder, "y_train.csv"))

# Drop non-numeric columns from training data (just in case)
X_train = X_train.select_dtypes(include=["number"])
y_train = y_train.iloc[:, 0]  # Ensure y_train is a Series, not a DataFrame

# Load test data
X_test = pd.read_csv(test_file_path)
missing_info = pd.read_csv(missing_info_path)

# Drop non-numeric columns from test data
X_test = X_test.select_dtypes(include=["number"])

# Ensure `y_train` is properly encoded with continuous labels
unique_players = sorted(y_train.unique())  # Get sorted unique player IDs
player_to_class = {player_id: idx for idx, player_id in enumerate(unique_players)}
y_train = y_train.map(player_to_class)  # Map to sequential class labels
num_classes = len(unique_players)  # Correct number of unique player classes

# Initialize XGBoost model
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    objective="multi:softmax",
    num_class=num_classes,
    eval_metric="mlogloss"
)

# Train the model
print("🔄 Training XGBoost model...")
model.fit(X_train, y_train)

# Save the trained model
model_path = os.path.join(model_folder, "xgboost_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f"✅ Model saved at: {model_path}")

# Make predictions on the test set
print("🔄 Making predictions on test data...")
y_pred = model.predict(X_test)

# Convert predictions back to player names
predicted_players = [reverse_player_dict[player] for player in y_pred]

# Restore predictions into the correct `home_x` columns
final_predictions = X_test.copy()
for i, row in missing_info.iterrows():
    game_id = row["Game_ID"]
    missing_col = row["Missing_Player_Column"]
    final_predictions.at[game_id, missing_col] = predicted_players[i]

# Save final predictions
final_predictions.to_csv(output_predictions_path, index=False)
print(f"✅ Final predictions saved at: {output_predictions_path}")
