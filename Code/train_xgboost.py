import os
import pandas as pd
import pickle
import xgboost as xgb

# Define folder paths
train_folder = "../Train"  # Your new processed training data folder
model_folder = "../Model"
os.makedirs(model_folder, exist_ok=True)

# Load encoding dictionaries
with open("../Dictionaries/season_encoding.pkl", "rb") as f:
    season_dict = pickle.load(f)

with open("../Dictionaries/team_encoding.pkl", "rb") as f:
    team_dict = pickle.load(f)

with open("../Dictionaries/player_encoding.pkl", "rb") as f:
    player_dict = pickle.load(f)

# Reverse dictionary for decoding predictions
reverse_player_dict = {v: k for k, v in player_dict.items()}

# Columns for features
feature_columns = ["season", "team", "player_0", "player_1", "player_2", "player_3"]

# ---------------------------
# Step 1: Load & Prepare Training Data
# ---------------------------
print("📂 Loading training data...")

data_list = []
for filename in os.listdir(train_folder):
    if filename.endswith(".csv"):  # Process all training files
        file_path = os.path.join(train_folder, filename)
        df = pd.read_csv(file_path)
        data_list.append(df)

# Combine all years into a single DataFrame
train_data = pd.concat(data_list, ignore_index=True)

# Extract features (X) and labels (y)
X_train = train_data[feature_columns]
y_train = train_data["target_player"]  # The player to predict

# Ensure `y_train` is properly encoded
unique_players = sorted(y_train.unique())
player_to_class = {player_id: idx for idx, player_id in enumerate(unique_players)}
y_train = y_train.map(player_to_class)  # Convert to sequential class labels
num_classes = len(unique_players)

# ---------------------------
# Step 2: Train the XGBoost Model
# ---------------------------
print("🔄 Training the XGBoost model...")

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    objective="multi:softmax",
    num_class=num_classes,
    eval_metric="mlogloss",
    tree_method="hist",  # Use hist, NOT gpu_hist
    device="cuda"  # Enable GPU acceleration
)


model.fit(X_train, y_train)

# ---------------------------
# Step 3: Save the Trained Model
# ---------------------------
model_path = os.path.join(model_folder, "xgboost_model_v2.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"✅ Model saved at: {model_path}")
