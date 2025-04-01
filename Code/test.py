# Load mappings
import pickle

with open("../Model/class_to_player_mapping.pkl", "rb") as f:
    class_to_player = pickle.load(f)

with open("../Dictionaries/player_encoding.pkl", "rb") as f:
    player_dict = pickle.load(f)

# Create reverse player dict
reverse_player_dict = {v: k for k, v in player_dict.items()}

# Example: try decoding 1070
predicted_class = 1070
if predicted_class in class_to_player:
    player_id = class_to_player[predicted_class]
    player_name = reverse_player_dict.get(player_id, "❌ Unknown")
    print(f"✅ Class {predicted_class} → Player ID {player_id} → Name: {player_name}")
else:
    print(f"❌ Class {predicted_class} not found in class_to_player")
