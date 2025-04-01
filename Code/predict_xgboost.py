import os
import pandas as pd
import pickle
import xgboost as xgb
import numpy as np
import json

# ----------------------
# File paths
# ----------------------
model_path = "../Model/xgboost_model_v9.pkl"
test_data_path = "../Model/X_test.csv"
dictionary_folder = "../Dictionaries"
roster_json_path = "../my_lists/team_rosters_by_season.json"
role_freq_path = "../my_lists/role_frequencies_by_season_team.json"
output_predictions_path = "../Model/final_predictions_top10.csv"
class_mapping_path = "../Model/class_to_player_mapping.pkl"


# Add near top with other imports
FALLBACK_THRESHOLD = 10  # Number of predictions we want to ensure


# ----------------------
# Load encoding dictionaries
# ----------------------
with open(os.path.join(dictionary_folder, "player_encoding.pkl"), "rb") as f:
    player_dict = pickle.load(f)
with open(os.path.join(dictionary_folder, "season_encoding.pkl"), "rb") as f:
    season_dict = pickle.load(f)
with open(os.path.join(dictionary_folder, "team_encoding.pkl"), "rb") as f:
    team_dict = pickle.load(f)
with open(class_mapping_path, "rb") as f:
    class_to_player = pickle.load(f)

reverse_season_dict = {v: k for k, v in season_dict.items()}
reverse_team_dict = {v: k for k, v in team_dict.items()}

# ----------------------
# Load trained model
# ----------------------
with open(model_path, "rb") as f:
    model = pickle.load(f)

# ----------------------
# Load supporting JSONs
# ----------------------
with open(roster_json_path, "r") as f:
    season_team_rosters = json.load(f)
with open(role_freq_path, "r") as f:
    role_frequencies = json.load(f)

# ----------------------
# Load Test Data
# ----------------------
df_test = pd.read_csv(test_data_path)
required_cols = ["season", "team", "player_0", "player_1", "player_2", "player_3", "missing_position", "missing_player"]
if not all(col in df_test.columns for col in required_cols):
    raise ValueError("❌ Missing one or more required columns in X_test.csv")

X_test = df_test[["season", "team", "player_0", "player_1", "player_2", "player_3"]]
missing_positions = df_test["missing_position"].astype(str).tolist()
true_missing_players = df_test["missing_player"].tolist()

print(f"Test data shape: {X_test.shape}")

# ----------------------
# Make Predictions
# ----------------------
print("🔍 Making predictions...")

y_probs = model.predict_proba(X_test)
final_predictions = []

for i, row_probs in enumerate(y_probs):
    season_id = X_test.iloc[i]["season"]
    team_id = X_test.iloc[i]["team"]

    # Decode season and team to strings
    season_str = reverse_season_dict.get(season_id)
    team_str = reverse_team_dict.get(team_id)

    valid_names = set()
    if season_str and team_str:
        valid_names = set(season_team_rosters.get(season_str, {}).get(team_str, []))

    valid_players = {
        player_dict[name]
        for name in valid_names
        if name in player_dict
    }

    # Top 15 predicted player IDs
    top_15_indices = np.argsort(row_probs)[::-1][:15]

    # Filter by team roster
    filtered = [(idx, row_probs[idx]) for idx in top_15_indices if idx in valid_players]
    filtered = filtered[:10]

        # If not enough predictions, apply role fallback using frequency data
    if len(filtered) < FALLBACK_THRESHOLD and season_str and team_str:
        role_key = f"home_{missing_positions[i]}"
        fallback_list = role_frequencies.get(season_str, {}).get(team_str, {}).get(role_key, {})

        if fallback_list:
            sorted_fallbacks = sorted(fallback_list.items(), key=lambda x: -x[1])  # most frequent first
            for player_name, _ in sorted_fallbacks:
                encoded_id = player_dict.get(player_name)
                already_used_ids = [idx for idx, _ in filtered]
                if encoded_id and encoded_id not in already_used_ids:
                    filtered.append((encoded_id, 0.0))
                    if len(filtered) == FALLBACK_THRESHOLD:
                        break


    # Final padding
    while len(filtered) < 10:
        filtered.append((None, 0.0))

   # Decode predictions using class_to_player mapping, fallback to player_dict
    top_10_players = []
    for idx, _ in filtered:
        if idx is None:
            top_10_players.append("Unknown Player")
        elif idx in class_to_player:
            top_10_players.append(class_to_player[idx])
        else:
            # Fallback: decode using reverse player_dict
            name = next((name for name, encoded in player_dict.items() if encoded == idx), "Unknown Player")
            top_10_players.append(name)

    top_10_probs = [round(prob, 4) for _, prob in filtered]

    # True value
    actual_decoded = df_test.iloc[i]["missing_player"]

    result_row = {
        "season": season_str,
        "actual_player": actual_decoded
    }
    for rank, (player, prob) in enumerate(zip(top_10_players, top_10_probs), start=1):
        result_row[f"top{rank}_player"] = player
        result_row[f"top{rank}_prob"] = prob

    final_predictions.append(result_row)  # ✅ inside the loop!
# ----------------------
# Save Output
# ----------------------
final_df = pd.DataFrame(final_predictions)
final_df.to_csv(output_predictions_path, index=False)
print(f"✅ Final top 10 predictions saved to: {output_predictions_path}")
