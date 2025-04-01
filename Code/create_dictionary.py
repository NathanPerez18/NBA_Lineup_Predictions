import os
import pickle

# ---------
# Paths
# ---------
input_folder = "../my_lists"
output_folder = "../Dictionaries"
os.makedirs(output_folder, exist_ok=True)

# ---------
# Helper: load and encode list
# ---------
def build_encoding_dict(file_path):
    with open(file_path, "r") as f:
        values = [line.strip() for line in f.readlines() if line.strip()]
    encoding = {val: idx for idx, val in enumerate(sorted(values))}
    reverse = {idx: val for val, idx in encoding.items()}
    return encoding, reverse

# ---------
# Encode season
# ---------
season_encoding, season_reverse = build_encoding_dict(os.path.join(input_folder, "season_list.txt"))
with open(os.path.join(output_folder, "season_encoding.pkl"), "wb") as f:
    pickle.dump(season_encoding, f)
with open(os.path.join(output_folder, "season_reverse.pkl"), "wb") as f:
    pickle.dump(season_reverse, f)
print(f"✅ Season dictionary saved with {len(season_encoding)} entries.")

# ---------
# Encode team
# ---------
team_encoding, team_reverse = build_encoding_dict(os.path.join(input_folder, "team_list.txt"))
with open(os.path.join(output_folder, "team_encoding.pkl"), "wb") as f:
    pickle.dump(team_encoding, f)
with open(os.path.join(output_folder, "team_reverse.pkl"), "wb") as f:
    pickle.dump(team_reverse, f)
print(f"✅ Team dictionary saved with {len(team_encoding)} entries.")

# ---------
# Encode player
# ---------
player_encoding, player_reverse = build_encoding_dict(os.path.join(input_folder, "player_list.txt"))
with open(os.path.join(output_folder, "player_encoding.pkl"), "wb") as f:
    pickle.dump(player_encoding, f)
with open(os.path.join(output_folder, "player_reverse.pkl"), "wb") as f:
    pickle.dump(player_reverse, f)
print(f"✅ Player dictionary saved with {len(player_encoding)} entries.")
