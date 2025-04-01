import pandas as pd

# ----------------------------
# 🔍 Load predictions
# ----------------------------
predictions_path = "../Model/final_predictions_top10.csv"
df = pd.read_csv(predictions_path)

# ----------------------------
# ✅ Compare actual vs. predictions
# ----------------------------
results = []
season_accuracy = {}

for _, row in df.iterrows():
    actual = row["actual_player"]
    season = row["season"]

    # Get list of top-10 predicted players
    predicted_players = [row[f"top{i}_player"] for i in range(1, 11)]

    # Check if actual is in top-10
    is_correct = actual in predicted_players
    results.append(is_correct)

    # Track accuracy by season
    if season not in season_accuracy:
        season_accuracy[season] = {"correct": 0, "total": 0}

    season_accuracy[season]["total"] += 1
    if is_correct:
        season_accuracy[season]["correct"] += 1

# ----------------------------
# 📊 Print results
# ----------------------------
overall_accuracy = sum(results) / len(results) if results else 0
print(f"\n🎯 Overall Top-10 Accuracy: {overall_accuracy * 100:.2f}%")

print("\n📅 Accuracy by Season:")
for season, stats in sorted(season_accuracy.items()):
    correct = stats["correct"]
    total = stats["total"]
    acc = (correct / total) * 100 if total > 0 else 0
    print(f"  {season}: {acc:.2f}% ({correct}/{total})")
