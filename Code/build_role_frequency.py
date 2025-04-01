import os
import pandas as pd
import json
from collections import defaultdict

# ---------
# Paths
# ---------
filtered_folder = "../Filtered"
output_path = "../my_lists/role_frequencies_by_season_team.json"

# ---------
# Initialize nested frequency structure
# ---------
frequencies = defaultdict(
    lambda: defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
)

# ---------
# Process filtered files
# ---------
for filename in os.listdir(filtered_folder):
    if filename.endswith(".csv"):
        df = pd.read_csv(os.path.join(filtered_folder, filename))

        for _, row in df.iterrows():
            season = str(row["season"])
            home_team = row["home_team"]
            away_team = row["away_team"]

            for i in range(5):
                home_role = f"home_{i}"
                away_role = f"away_{i}"

                home_player = row.get(home_role)
                away_player = row.get(away_role)

                if pd.notna(home_player) and pd.notna(home_team):
                    frequencies[season][home_team][home_role][home_player] += 1
                if pd.notna(away_player) and pd.notna(away_team):
                    frequencies[season][away_team][away_role][away_player] += 1

# ---------
# Save to JSON
# ---------
# Convert defaultdicts to dicts
frequencies = {
    season: {
        team: {
            role: dict(player_counts)
            for role, player_counts in team_roles.items()
        }
        for team, team_roles in season_data.items()
    }
    for season, season_data in frequencies.items()
}

with open(output_path, "w") as f:
    json.dump(frequencies, f, indent=2)

print(f"✅ Saved role frequencies to: {output_path}")
