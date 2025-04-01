import os
import pandas as pd
import pickle

# ---------
# Paths
# ---------
answers_path = "../Test/professor_answers.csv"
encoded_output_path = "../Encoded_Test/professor_answers_encoded.csv"
player_dict_path = "../Dictionaries/player_encoding.pkl"

# ---------
# Load encoding dictionary
# ---------
with open(player_dict_path, "rb") as f:
    player_dict = pickle.load(f)

# ---------
# Load and encode
# ---------
if os.path.exists(answers_path):
    df = pd.read_csv(answers_path)

    if "missing_player" not in df.columns:
        raise KeyError("❌ 'missing_player' column not found in professor_answers.csv")

    df["missing_player"] = df["missing_player"].map(player_dict)

    # Check for missing encodings
    if df["missing_player"].isnull().any():
        print("⚠️ Warning: Some players in 'missing_player' could not be encoded.")

    # Save encoded answers
    df.to_csv(encoded_output_path, index=False)
    print(f"✅ Encoded answers saved at: {encoded_output_path}")
else:
    print(f"❌ File not found: {answers_path}")
