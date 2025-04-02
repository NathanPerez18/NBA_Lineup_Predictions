"""
🧝‍♂️ SANTA SCRIPT — Combined with dictionary creation and 2016 data patching
Generates the following:

# From training and test data:
- my_lists/season_list.txt
- my_lists/team_list.txt
- my_lists/player_list.txt

# Team/Season structure:
- my_lists/team_rosters.json
- my_lists/team_rosters_by_season.json
- my_lists/role_frequencies_by_season_team.json

# Model inputs:
- Model/X_train.csv
- Model/y_train.csv
- Dictionaries/class_to_player_mapping.pkl
- Model/X_test.csv
- Dictionaries/player_encoding.pkl, team_encoding.pkl, season_encoding.pkl
"""

import os
import pandas as pd
import json
import pickle
import csv
from collections import defaultdict

# -------------------------------
# 🔧 Path Setup
# -------------------------------
raw_folder = "../Raw"
filtered_folder = "../Filtered"
test_data_path = "../Test/professor_test_data.csv"
test_answers_path = "../Test/professor_answers.csv"
dict_folder = "../Dictionaries"
output_folder = "../my_lists"
model_folder = "../Model"
train_folder = "../Train"
matchups_2016_path = os.path.join(filtered_folder, "matchups-2016.csv")

os.makedirs(filtered_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)
os.makedirs(model_folder, exist_ok=True)
os.makedirs(dict_folder, exist_ok=True)
os.makedirs(train_folder, exist_ok=True)

# -------------------------------
# 🧼 Step 0: Clean Raw NBA Data
# -------------------------------
required_columns = [
    "season", "home_team", "away_team",
    "home_0", "home_1", "home_2", "home_3", "home_4",
    "away_0", "away_1", "away_2", "away_3", "away_4", "outcome"
]

print("🧽 Cleaning raw NBA data...")
for filename in os.listdir(raw_folder):
    if filename.endswith(".csv"):
        file_path = os.path.join(raw_folder, filename)
        df = pd.read_csv(file_path)
        df = df[required_columns]

        clean_path = os.path.join(filtered_folder, filename)
        df.to_csv(clean_path, index=False)
        print(f"✅ Cleaned & saved: {clean_path}")

# -------------------------------
# 🧪 Generate 2016 Matchups
# -------------------------------
if not os.path.exists(matchups_2016_path):
    df_test = pd.read_csv(test_data_path)
    df_answers = pd.read_csv(test_answers_path)

    df_test = df_test[df_test["season"] == 2016].reset_index(drop=True)
    df_answers = df_answers.iloc[:len(df_test)]
    df_test["missing_player"] = df_answers["missing_player"]

    def fill_missing(row):
        for i in range(5):
            if row[f"home_{i}"] == "?":
                row[f"home_{i}"] = row["missing_player"]
            if row[f"away_{i}"] == "?":
                row[f"away_{i}"] = row["missing_player"]
        return row

    df_test = df_test.apply(fill_missing, axis=1)
    df_test["outcome"] = 1

    if "starting_min" in df_test.columns:
        df_test = df_test.drop(columns=["starting_min"])

    df_test = df_test[required_columns]
    df_test.to_csv(matchups_2016_path, index=False)
    print(f"✅ Created clean {matchups_2016_path} for training pipeline.")
else:
    print("ℹ️ Found existing matchups-2016.csv")

# -------------------------------
# 🔄 Prepare Train Data from Filtered
# -------------------------------
print("\n🧠 Preparing training rows for fallback model...")
with open(os.path.join(dict_folder, "player_encoding.pkl"), "rb") as f:
    player_dict = pickle.load(f)
with open(os.path.join(dict_folder, "season_encoding.pkl"), "rb") as f:
    season_dict = pickle.load(f)
with open(os.path.join(dict_folder, "team_encoding.pkl"), "rb") as f:
    team_dict = pickle.load(f)

player_columns = [f"{side}_{i}" for side in ["home", "away"] for i in range(5)]

for filename in os.listdir(filtered_folder):
    if filename.endswith(".csv"):
        file_path = os.path.join(filtered_folder, filename)
        print(f"⚙️ Processing {filename}...")
        df = pd.read_csv(file_path)

        df["season"] = df["season"].astype(str).map(season_dict)
        df["home_team"] = df["home_team"].map(team_dict)
        df["away_team"] = df["away_team"].map(team_dict)
        for col in player_columns:
            df[col] = df[col].map(player_dict)

        if df[["season", "home_team", "away_team"] + player_columns].isnull().any().any():
            print(f"⚠️ Warning: Missing mappings in {filename}")

        df_winners = df[df["outcome"] == 1].copy()

        train_data = []
        for _, row in df_winners.iterrows():
            season = row["season"]
            team = row["home_team"]
            players = [row[f"home_{i}"] for i in range(5)]
            missing_position = 4
            input_players = players[:missing_position] + players[missing_position + 1:]
            target_player = players[missing_position]
            train_data.append([season, team] + input_players + [missing_position, target_player])

        train_df = pd.DataFrame(train_data, columns=[
            "season", "team", "player_0", "player_1", "player_2", "player_3", "missing_position", "target_player"
        ])

        output_path = os.path.join(train_folder, filename)
        train_df.to_csv(output_path, index=False)
        print(f"✅ Saved training file: {output_path}")

print("✅ All training files processed and saved in the Train folder!")

# -------------------------------
# 🧽Load 2016 Test Data
# -------------------------------
patched_rows = []
answers = []
with open(test_answers_path, newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f, skipinitialspace=True)
    reader.fieldnames = [field.strip() for field in reader.fieldnames]
    for raw_row in reader:
        row = {k.strip(): v.strip() for k, v in raw_row.items()}
        answers.append(row["missing_player"])

with open(test_data_path, newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f, skipinitialspace=True)
    reader.fieldnames = [field.strip() for field in reader.fieldnames]
    answer_idx = 0
    for raw_row in reader:
        row = {k.strip(): v.strip() for k, v in raw_row.items()}
        if row["season"] != "2016":
            continue

        for i in range(5):
            if row[f"home_{i}"] == "?":
                row[f"home_{i}"] = answers[answer_idx]
                answer_idx += 1
            if row[f"away_{i}"] == "?":
                row[f"away_{i}"] = answers[answer_idx]
                answer_idx += 1

        patched_rows.append(row)

# -------------------------------
# 📊 Storage
# -------------------------------
season_set = set()
team_set = set()
player_set = set()
team_rosters = {}
season_team_rosters = {}
role_frequencies = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))

X_train_rows = []
y_train_ids = []

# -------------------------------
# 🛠️ Process training data
# -------------------------------
for filename in os.listdir(filtered_folder):
    if not filename.endswith(".csv"):
        continue

    df = pd.read_csv(os.path.join(filtered_folder, filename))

    for _, row in df[df["outcome"] == 1].iterrows():
        season = str(row["season"])
        team = row["home_team"]
        players = [row[f"home_{i}"] for i in range(5)]

        season_set.add(season)
        team_set.add(team)
        player_set.update(players)

        season_team_rosters.setdefault(season, {}).setdefault(team, set()).update(players)
        team_rosters.setdefault(team, set()).update(players)

        for i in range(5):
            role = f"home_{i}"
            role_frequencies[season][team][role][players[i]] += 1

# Add patched 2016 data
for row in patched_rows:
    season = row["season"]
    home_team = row["home_team"]
    away_team = row["away_team"]

    home_players = [row[f"home_{i}"] for i in range(5)]
    away_players = [row[f"away_{i}"] for i in range(5)]

    season_set.add(season)
    team_set.update([home_team, away_team])
    player_set.update(home_players + away_players)

    season_team_rosters.setdefault(season, {}).setdefault(home_team, set()).update(home_players)
    season_team_rosters.setdefault(season, {}).setdefault(away_team, set()).update(away_players)
    team_rosters.setdefault(home_team, set()).update(home_players)
    team_rosters.setdefault(away_team, set()).update(away_players)

    for i in range(5):
        role_frequencies[season][home_team][f"home_{i}"][home_players[i]] += 1
        role_frequencies[season][away_team][f"away_{i}"][away_players[i]] += 1

# -------------------------------
# 📊 Save raw lists to txt
# -------------------------------
def save_txt(name, data):
    path = os.path.join(output_folder, f"{name}_list.txt")
    with open(path, "w") as f:
        for item in sorted(map(str, data)):
            f.write(item + "\n")
    print(f"✅ Saved: {path}")

save_txt("season", season_set)
save_txt("team", team_set)
save_txt("player", player_set)

# -------------------------------
# 🔐 Create and save dictionaries from lists
# -------------------------------
def build_encoding_dict(file_path):
    with open(file_path, "r") as f:
        values = [line.strip() for line in f.readlines() if line.strip()]
    encoding = {val: idx for idx, val in enumerate(sorted(values))}
    reverse = {idx: val for val, idx in encoding.items()}
    return encoding, reverse

season_encoding, season_reverse = build_encoding_dict(os.path.join(output_folder, "season_list.txt"))
team_encoding, team_reverse = build_encoding_dict(os.path.join(output_folder, "team_list.txt"))
player_encoding, player_reverse = build_encoding_dict(os.path.join(output_folder, "player_list.txt"))

with open(os.path.join(dict_folder, "season_encoding.pkl"), "wb") as f:
    pickle.dump(season_encoding, f)
with open(os.path.join(dict_folder, "season_reverse.pkl"), "wb") as f:
    pickle.dump(season_reverse, f)

with open(os.path.join(dict_folder, "team_encoding.pkl"), "wb") as f:
    pickle.dump(team_encoding, f)
with open(os.path.join(dict_folder, "team_reverse.pkl"), "wb") as f:
    pickle.dump(team_reverse, f)

with open(os.path.join(dict_folder, "player_encoding.pkl"), "wb") as f:
    pickle.dump(player_encoding, f)
with open(os.path.join(dict_folder, "player_reverse.pkl"), "wb") as f:
    pickle.dump(player_reverse, f)

print("\n🚀 Encoding dictionaries generated and saved!")

# Assign dictionaries for use in encoding
season_dict = season_encoding
team_dict = team_encoding
player_dict = player_encoding
reverse_player_dict = player_reverse

# -------------------------------
# 🧪 Build Encoded Training Data
# -------------------------------
for filename in os.listdir(filtered_folder):
    if not filename.endswith(".csv"):
        continue

    df = pd.read_csv(os.path.join(filtered_folder, filename))

    for _, row in df[df["outcome"] == 1].iterrows():
        season = str(row["season"])
        team = row["home_team"]
        players = [row[f"home_{i}"] for i in range(5)]

        for i in range(5):
            input_players = players[:i] + players[i+1:]
            target_player = players[i]

            X_train_rows.append([
                season_dict.get(season, -1),
                team_dict.get(team, -1),
                *[player_dict.get(p, -1) for p in input_players]
            ])
            y_train_ids.append(player_dict.get(target_player, -1))

# -------------------------------
# 📦 Process Test Data
# -------------------------------
df_test = pd.read_csv(test_data_path)
df_answers = pd.read_csv(test_answers_path)

X_test_rows = []

for i, row in df_test.iterrows():
    season = str(row["season"])
    home_team = row["home_team"]
    home_players = [row[f"home_{j}"] for j in range(5)]

    if "?" not in home_players:
        continue

    missing_position = home_players.index("?")
    input_players = home_players[:missing_position] + home_players[missing_position+1:]
    encoded_players = [player_dict.get(p, -1) for p in input_players]

    X_test_rows.append([
        season_dict.get(season, -1),
        team_dict.get(home_team, -1),
        *encoded_players,
        missing_position,
        df_answers.iloc[i]["missing_player"]
    ])

X_test = pd.DataFrame(X_test_rows, columns=[
    "season", "team", "player_0", "player_1", "player_2", "player_3", "missing_position", "missing_player"
])
X_test.to_csv(os.path.join(model_folder, "X_test.csv"), index=False)

# -------------------------------
# 💾 Save Encoded Training Data
# -------------------------------
unique_y_ids = sorted(set(y_train_ids))
id_to_class = {pid: i for i, pid in enumerate(unique_y_ids)}
class_to_player = {i: reverse_player_dict[pid] for i, pid in enumerate(unique_y_ids)}

with open(os.path.join(model_folder, "class_to_player_mapping.pkl"), "wb") as f:
    pickle.dump(class_to_player, f)

X_train = pd.DataFrame(X_train_rows, columns=["season", "team", "player_0", "player_1", "player_2", "player_3"])
y_train = pd.DataFrame([id_to_class[pid] for pid in y_train_ids], columns=["target_player"])

X_train.to_csv(os.path.join(model_folder, "X_train.csv"), index=False)
y_train.to_csv(os.path.join(model_folder, "y_train.csv"), index=False)

# -------------------------------
# 📅 Save JSON Outputs
# -------------------------------
with open(os.path.join(output_folder, "team_rosters.json"), "w") as f:
    json.dump({team: sorted(list(players)) for team, players in team_rosters.items()}, f, indent=2)

with open(os.path.join(output_folder, "team_rosters_by_season.json"), "w") as f:
    json.dump({
        season: {team: sorted(list(players)) for team, players in team_dict.items()}
        for season, team_dict in season_team_rosters.items()
    }, f, indent=2)

with open(os.path.join(output_folder, "role_frequencies_by_season_team.json"), "w") as f:
    json.dump(role_frequencies, f, indent=2)

print("\n🏵️ All artifacts have been generated successfully!")
