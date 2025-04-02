import pandas as pd

# ----------------------------
# 🔍 Load predictions
# ----------------------------
predictions_path = "../Model/final_predictions_top10.csv"
df = pd.read_csv(predictions_path)

# ----------------------------
# ✅ Compare actual vs. predictions
# ----------------------------
results = {"top1": [], "top5": [], "top10": []}
season_accuracy = {}

for _, row in df.iterrows():
    actual = row["actual_player"]
    season = row["season"]

    # Get list of top-N predicted players
    top_1 = [row["top1_player"]]
    top_5 = [row[f"top{i}_player"] for i in range(1, 6)]
    top_10 = [row[f"top{i}_player"] for i in range(1, 11)]

    # Evaluate Top-N accuracy
    is_top1 = actual in top_1
    is_top5 = actual in top_5
    is_top10 = actual in top_10

    results["top1"].append(is_top1)
    results["top5"].append(is_top5)
    results["top10"].append(is_top10)

    # Track accuracy by season
    if season not in season_accuracy:
        season_accuracy[season] = {
            "top1": 0, "top5": 0, "top10": 0, "total": 0
        }

    season_accuracy[season]["total"] += 1
    if is_top1:
        season_accuracy[season]["top1"] += 1
    if is_top5:
        season_accuracy[season]["top5"] += 1
    if is_top10:
        season_accuracy[season]["top10"] += 1

# ----------------------------
# 📊 Print results
# ----------------------------
print("\n🎯 Overall Accuracy:")
for level in ["top1", "top5", "top10"]:
    acc = sum(results[level]) / len(results[level]) * 100 if results[level] else 0
    print(f"  {level.upper()} Accuracy: {acc:.2f}%")

print("\n📅 Accuracy by Season:")
for season in sorted(season_accuracy.keys()):
    stats = season_accuracy[season]
    total = stats["total"]
    print(f"  {season}:")
    for level in ["top1", "top10"]:
        correct = stats[level]
        acc = (correct / total) * 100 if total > 0 else 0
        print(f"    {level.upper():<6} Accuracy: {acc:.2f}% ({correct}/{total})")
