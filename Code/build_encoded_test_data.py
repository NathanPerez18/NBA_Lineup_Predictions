import os
import pandas as pd
import pickle

# -------------------------------
# 📁 Path Setup
# -------------------------------
test_folder = "../Test"
encoded_folder = "../Encoded_Test"
dictionary_folder = "../Dictionaries"
output_path = "../Model/X_test.csv"
os.makedirs(encoded_folder, exist_ok=True)
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# -------------------------------
# 📦 Load Dictionaries
# -------------------------------
with open(os.path.join(dictionary_folder, "season_encoding.pkl"), "rb") as f:
    season_dict = pickle.load(f)
with open(os.path.join(dictionary_folder, "team_encoding.pkl"), "rb") as f:
    team_dict = pickle.load(f)
with open(os.path.join(dictionary_folder, "player_encoding.pkl"), "rb") as f:
    player_dict = pickle.load(f)

# -------------------------------
# 📄 Load and Encode Test Data
# -------------------------------
test_file = os.path.join(test_folder, "professor_test_data.csv")
answer_file = os.path.join(test_folder, "professor_answers.csv")
encoded_test_path = os.path.join(encoded_folder, "professor_test_data.csv")
encoded_answer_path = os.path.join(encoded_folder, "professor_answers_encoded.csv")

if not os.path.exists(test_file) or not os.path.exists(answer_file):
    raise FileNotFoundError("❌ Test or answer file not found.")

print("🔄 Encoding test data...")
df_test = pd.read_csv(test_file)
df_answers = pd.read_csv(answer_file)

if "missing_player" not in df_answers.columns:
    raise KeyError("❌ 'missing_player' column missing in professor_answers.csv")

# Drop unwanted columns
if "starting_min" in df_test.columns:
    df_test = df_test.drop(columns=["starting_min"])

# Encode test
df_test["season"] = df_test["season"].astype(str).map(season_dict)
for col in ["home_team", "away_team"]:
    df_test[col] = df_test[col].map(team_dict)

player_cols = [f"{side}_{i}" for side in ["home", "away"] for i in range(5)]
for col in player_cols:
    df_test[col] = df_test[col].map(player_dict).fillna(-1)

df_test.to_csv(encoded_test_path, index=False)
print(f"✅ Encoded test data saved: {encoded_test_path}")

# Encode answers
df_answers["missing_player"] = df_answers["missing_player"].map(player_dict)
if df_answers["missing_player"].isnull().any():
    print("⚠️ Warning: Some players in 'missing_player' could not be encoded.")

df_answers.to_csv(encoded_answer_path, index=False)
print(f"✅ Encoded answers saved: {encoded_answer_path}")

# -------------------------------
# 🔍 Build Final X_test.csv
# -------------------------------
print("🧩 Assembling X_test for model input...")

home_cols = [f"home_{i}" for i in range(5)]
processed_rows = []

for idx, row in df_test.iterrows():
    home_players = row[home_cols].tolist()
    try:
        missing_index = home_players.index(-1)
    except ValueError:
        continue  # skip rows with no missing player

    input_players = home_players[:missing_index] + home_players[missing_index+1:]
    missing_player = df_answers.loc[idx, "missing_player"]
    processed_row = [row["season"], row["home_team"]] + input_players + [missing_index, missing_player]
    processed_rows.append(processed_row)

X_test_df = pd.DataFrame(processed_rows, columns=[
    "season", "team", "player_0", "player_1", "player_2", "player_3", "missing_position", "missing_player"
])
X_test_df.to_csv(output_path, index=False)

print(f"✅ Final X_test file saved at: {output_path}")
