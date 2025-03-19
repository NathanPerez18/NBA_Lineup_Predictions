import os
import pandas as pd
import pickle
import xgboost as xgb

# Define folder paths
encoded_folder = "../Encoded"  # Training data folder
model_folder = "../Model"
os.makedirs(model_folder, exist_ok=True)

# Load the player encoding dictionary
with open("player_encoding.pkl", "rb") as f:
    player_dict = pickle.load(f)

# Reverse dictionary for decoding predictions
reverse_player_dict = {v: k for k, v in player_dict.items()}

# Columns that contain player names
player_columns = ["home_0", "home_1", "home_2", "home_3", "home_4",
                  "away_0", "away_1", "away_2", "away_3", "away_4"]

# Game-related columns
game_columns = ["season", "starting_min"]

# ---------------------------
# Step 1: Train 5 Different Models
# ---------------------------
for missing_index, missing_col in enumerate(["home_0", "home_1", "home_2", "home_3", "home_4"]):
    print(f"🔄 Training Model {missing_index+1} (Missing {missing_col})...")

    data_list = []
    for filename in os.listdir(encoded_folder):
        if filename.endswith("2007.csv"):
            file_path = os.path.join(encoded_folder, filename)
            df = pd.read_csv(file_path)

            # Set target player
            df["target_player"] = df[missing_col]

            # Remove the missing player column from features
            df = df.drop(columns=[missing_col])

            data_list.append(df)

    # Combine data and extract features & target
    train_data = pd.concat(data_list, ignore_index=True)
    X_train = train_data[game_columns + [col for col in player_columns if col != missing_col]]
    y_train = train_data["target_player"]

    # Ensure `y_train` is properly encoded with continuous labels
    unique_players = sorted(y_train.unique())  # Get sorted unique player IDs
    player_to_class = {player_id: idx for idx, player_id in enumerate(unique_players)}
    y_train = y_train.map(player_to_class)  # Convert to sequential class labels
    num_classes = len(unique_players)

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
    model.fit(X_train, y_train)

    # Save the trained model
    model_path = os.path.join(model_folder, f"xgboost_model_{missing_index}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    print(f"✅ Model {missing_index+1} saved at: {model_path}")
