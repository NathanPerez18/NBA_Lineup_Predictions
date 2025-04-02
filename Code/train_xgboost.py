import os
import pickle
import pandas as pd
import xgboost as xgb

# ---------------------------
# 📁 Folder Setup
# ---------------------------
model_folder = "../Model"
os.makedirs(model_folder, exist_ok=True)

# ---------------------------
# 📂 Load Preprocessed Data
# ---------------------------
X_train = pd.read_csv(os.path.join(model_folder, "X_train.csv"))
y_train = pd.read_csv(os.path.join(model_folder, "y_train.csv"))["target_player"]

with open(os.path.join(model_folder, "class_to_player_mapping.pkl"), "rb") as f:
    class_to_player = pickle.load(f)

num_classes = len(class_to_player)

# ---------------------------
# 🤖 Train the XGBoost Model
# ---------------------------
print("🔄 Training the XGBoost model...")

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=12,
    learning_rate=0.03,
    subsample=0.8 ,
    colsample_bytree=0.8,
    gamma=0.1,
    min_child_weight=1,
    reg_lambda=1,
    reg_alpha=0.5,
    objective="multi:softprob",
    num_class=num_classes,
    eval_metric="mlogloss",
    tree_method="hist",
    device="cuda"
)

model.fit(X_train, y_train)

# ---------------------------
# 💾 Save the Trained Model
# ---------------------------
model_path = os.path.join(model_folder, "xgboost_model_v9.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"✅ Model saved at: {model_path}")
