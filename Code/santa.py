"""
🧝‍♂️ SANTA SCRIPT — Generates the following:

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
"""

import os
import pandas as pd
import json
import pickle
from collections import defaultdict

# -------------------------------
# 🔧 Path Setup
# -------------------------------
filtered_folder = "../Filtered"
test_data_path = "../Test/professor_test_data.csv"
test_answers_path = "../Test/professor_answers.csv"
dict_folder = "../Dictionaries"
output_folder = "../my_lists"
model_folder = "../Model"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(model_folder, exist_ok=True)

# -------------------------------
# 📦 Load encoding dictionaries
# -------------------------------
with open(os.path.join(dict_folder, "season_encoding.pkl"), "rb") as f:
    season_dict = pickle.load(f)
with open(os.path.join(dict_folder, "team_encoding.pkl"), "rb") as f:
    team_dict = pickle.load(f)
with open(os.path.join(dict_folder, "player_encoding.pkl"), "rb") as f:
    player_dict = pickle.load(f)

reverse_player_dict = {v: k for k, v in player_dict.items()}

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
# 🔁 Process training data
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

            input_players = players[:i] + players[i+1:]
            target_player = players[i]

            X_train_rows.append([
                season_dict.get(season, -1),
                team_dict.get(team, -1),
                *[player_dict.get(p, -1) for p in input_players]
            ])
            y_train_ids.append(player_dict.get(target_player, -1))

# -------------------------------
# 🧪 Process test data (to add missing 2016 season players)
# -------------------------------
df_test = pd.read_csv(test_data_path)
df_answers = pd.read_csv(test_answers_path)

test_rows = []

for i, row in df_test.iterrows():
    season = str(row["season"])
    home_team = row["home_team"]
    away_team = row["away_team"]

    season_set.add(season)
    team_set.update([home_team, away_team])

    for side in ["home", "away"]:
        team = row[f"{side}_team"]
        for j in range(5):
            player = row[f"{side}_{j}"]
            if player != "?":
                player_set.add(player)

    # Prepare for prediction task
    home_players = [row[f"home_{j}"] for j in range(5)]
    if "?" not in home_players:
        continue

    missing_position = home_players.index("?")
    input_players = home_players[:missing_position] + home_players[missing_position+1:]
    encoded_players = [player_dict.get(p, -1) for p in input_players]

    encoded_season = season_dict.get(season, -1)
    encoded_team = team_dict.get(row["home_team"], -1)
    missing_player = df_answers.iloc[i]["missing_player"]

    test_rows.append([
        encoded_season,
        encoded_team,
        *encoded_players,
        missing_position,
        missing_player
    ])

X_test = pd.DataFrame(test_rows, columns=[
    "season", "team", "player_0", "player_1", "player_2", "player_3", "missing_position", "missing_player"
])
X_test.to_csv(os.path.join(model_folder, "X_test.csv"), index=False)

# -------------------------------
# 🧠 Encode y_train and save mapping
# -------------------------------
unique_y_ids = sorted(set(y_train_ids))
id_to_class = {pid: i for i, pid in enumerate(unique_y_ids)}
class_to_player = {i: reverse_player_dict[pid] for i, pid in enumerate(unique_y_ids)}

y_train = pd.DataFrame([id_to_class[pid] for pid in y_train_ids], columns=["target_player"])
X_train = pd.DataFrame(X_train_rows, columns=["season", "team", "player_0", "player_1", "player_2", "player_3"])

X_train.to_csv(os.path.join(model_folder, "X_train.csv"), index=False)
y_train.to_csv(os.path.join(model_folder, "y_train.csv"), index=False)

with open(os.path.join(model_folder, "class_to_player_mapping.pkl"), "wb") as f:
    pickle.dump(class_to_player, f)

# -------------------------------
# 📤 Save lists
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

# JSON outputs
with open(os.path.join(output_folder, "team_rosters.json"), "w") as f:
    json.dump({team: sorted(list(players)) for team, players in team_rosters.items()}, f, indent=2)

with open(os.path.join(output_folder, "team_rosters_by_season.json"), "w") as f:
    json.dump({
        season: {team: sorted(list(players)) for team, players in team_dict.items()}
        for season, team_dict in season_team_rosters.items()
    }, f, indent=2)

with open(os.path.join(output_folder, "role_frequencies_by_season_team.json"), "w") as f:
    json.dump(role_frequencies, f, indent=2)

print("🎁 All artifacts have been generated successfully!")
