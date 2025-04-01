import os
import pandas as pd

# ----------------------------
# File Paths
# ----------------------------
encoded_test_path = "Encoded_Test/professor_test_data.csv"
answers_path = "Test/professor_answers.csv"
output_path = "Model/X_test.csv"

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# ----------------------------
# Load Data
# ----------------------------
df_test = pd.read_csv(encoded_test_path)
df_answers = pd.read_csv(answers_path)

# ----------------------------
# Configuration
# ----------------------------
home_cols = [f"home_{i}" for i in range(5)]
season_column = "season"
team_column = "home_team"
answer_column = "missing_player"

# ----------------------------
# Processing Logic
# ----------------------------
processed_rows = []

for idx, row in df_test.iterrows():
    player_values = row[home_cols].values.tolist()
    
    # Identify missing position
    try:
        missing_index = player_values.index(-1)
    except ValueError:
        continue  # skip if no missing player

    # Remove the missing player
    player_values.pop(missing_index)

    # Grab the answer from the same row in the answers file
    actual_missing_player = df_answers.loc[idx, answer_column]

    # Assemble the new row
    new_row = [row[season_column], row[team_column]] + player_values + [missing_index, actual_missing_player]
    processed_rows.append(new_row)

# ----------------------------
# Save Output
# ----------------------------
columns = ["season", "team", "player_0", "player_1", "player_2", "player_3", "missing_position", "missing_player"]
X_test_df = pd.DataFrame(processed_rows, columns=columns)
X_test_df.to_csv(output_path, index=False)

print(f"✅ Saved: {output_path}")
