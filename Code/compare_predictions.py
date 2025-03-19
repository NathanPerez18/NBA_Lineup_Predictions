import pandas as pd
import os

# Define file paths
answers_path = "../Test/professor_answers.csv"
predictions_path = "../Model/final_predictions.csv"

# Load data
answers_df = pd.read_csv(answers_path)
predictions_df = pd.read_csv(predictions_path)

# Ensure column names match
answers_df.columns = ["missing_player"]
predictions_df.columns = ["predicted_player"]

# Check if both files have the same number of rows
if len(answers_df) != len(predictions_df):
    print(f"⚠️ Warning: Answer file has {len(answers_df)} rows, but predictions file has {len(predictions_df)} rows.")

# Compare row by row
correct = 0
total = len(answers_df)

for i, (true_value, predicted_value) in enumerate(zip(answers_df["missing_player"], predictions_df["predicted_player"])):
    if true_value == predicted_value:
        correct += 1

    # Print running total every 10 rows
    if (i + 1) % 10 == 0:
        accuracy = (correct / (i + 1)) * 100
        print(f"✅ Checked {i + 1} rows - Accuracy so far: {accuracy:.2f}%")

# Final accuracy
final_accuracy = (correct / total) * 100
print(f"🎯 Final Accuracy: {final_accuracy:.2f}% ({correct}/{total} correct predictions)")
