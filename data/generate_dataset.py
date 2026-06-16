# data/generate_dataset.py
# Synthetic dataset generator for the AI Tutor PLR.
#
# Run this script ONCE before starting the app:
#   python data/generate_dataset.py
#
# Output:
#   data/student_scores.csv  (200 rows)
#
# Why synthetic data?
#   - No real student data needed (no privacy concerns)
#   - Full control over label distribution
#   - Labels assigned by same rules as the Rule Engine
#   - Slight noise added so Decision Tree learns soft boundaries
#     instead of memorizing perfect thresholds
#
# Sample dataset used by the app.
# Dataset used to train DecisionTreeClassifier.

import pandas as pd
import numpy as np
import os

# RANDOM SEED
# Fix the seed so the dataset is identical every time
# this script is run. Reproducibility is important for
# consistent model training results.
np.random.seed(42)

# CONFIGURATION
# All generation parameters defined here at the top.
# Avoid hardcoded values inside logic blocks.

NUM_RECORDS          = 200
LOW_SCORE_THRESHOLD  = 40     # Must match utils/helpers.py
HIGH_SCORE_THRESHOLD = 70     # Must match utils/helpers.py

TOPICS = [
    "OOP",
    "Functions",
    "Loops",
    "Arrays",
    "Recursion",
    "Variables",
    "Data Structures",
    "Algorithms",
    "File Handling",
    "Databases"
]

CONFIDENCE_LEVELS = ["High", "Medium", "Low"]

# LABEL ASSIGNMENT FUNCTION
# Assigns a recommendation label based on score + noise.
# Noise of +/-5 points creates soft boundaries so the
# Decision Tree cannot simply memorize the exact thresholds.
# This makes the trained model more realistic and prevents
# perfect overfitting on training data.

def assign_label(score):
    """
    Assigns a recommendation label to a student record.

    Applies the same IF-THEN logic as the Rule Engine
    but adds +/-5 point random noise to the score before
    checking the threshold. This simulates real-world
    variability and prevents the Decision Tree from
    achieving unrealistically perfect accuracy.

    Parameters:
        score (float): The student's quiz score (0 to 100).

    Returns:
        str: One of "Review Basics", "Practice More", "Next Topic".
    """
    # Add small random noise to blur the hard threshold boundary
    noise       = np.random.uniform(-5, 5)
    noisy_score = score + noise

    if noisy_score < LOW_SCORE_THRESHOLD:
        return "Review Basics"
    elif noisy_score < HIGH_SCORE_THRESHOLD:
        return "Practice More"
    else:
        return "Next Topic"

# RECORD GENERATION LOOP
# Generates NUM_RECORDS student quiz records.
# Each record has realistic correlations between features:
#   - Low scorers tend to take longer (confused)
#   - High scorers tend to report High confidence
#   - Previous score is close to current score (realistic trend)

records = []

for i in range(1, NUM_RECORDS + 1):

    # -- Score: uniform distribution across full range (5 to 100)
    # Starting at 5 avoids unrealistic 0% scores
    score_pct = round(np.random.uniform(5, 100), 1)

    # -- Response time: inversely correlated with score
    # Low scorers take longer (confused or guessing)
    # High scorers answer faster (confident and fluent)
    # Formula: base_time = 120 - score gives range ~20s to ~115s
    base_time     = 120 - score_pct
    response_time = round(
        max(10, base_time + np.random.normal(0, 15)), 1
    )

    # -- Confidence: loosely correlated with score
    # High scorers more likely to report High confidence
    # Low scorers more likely to report Low confidence
    if score_pct >= HIGH_SCORE_THRESHOLD:
        confidence = np.random.choice(
            ["High", "Medium"],
            p=[0.7, 0.3]
        )
    elif score_pct >= LOW_SCORE_THRESHOLD:
        confidence = np.random.choice(
            ["High", "Medium", "Low"],
            p=[0.2, 0.6, 0.2]
        )
    else:
        confidence = np.random.choice(
            ["Medium", "Low"],
            p=[0.3, 0.7]
        )

    # -- Previous score: close to current score with small variation
    # np.clip ensures the value stays within 0 to 100
    prev_score = round(
        np.clip(score_pct + np.random.normal(0, 10), 0, 100), 1
    )

    # -- Topic: randomly selected from the fixed topic list
    topic = np.random.choice(TOPICS)

    # -- Label: assigned by rule logic with noise
    recommendation = assign_label(score_pct)

    # Append completed record to list
    records.append({
        "student_id":     i,
        "topic":          topic,
        "score_pct":      score_pct,
        "response_time":  response_time,
        "confidence":     confidence,
        "prev_score":     prev_score,
        "recommendation": recommendation
    })

# SAVE TO CSV
# Converts the list of dicts to a DataFrame and saves it.
# os.makedirs ensures the data/ folder exists before writing.
# Sample dataset used by the app.

df = pd.DataFrame(records)

# Create data/ folder if it does not already exist
os.makedirs("data", exist_ok=True)

output_path = os.path.join("data", "student_scores.csv")
df.to_csv(output_path, index=False)

# Print confirmation summary to terminal
print(f"[OK]   Dataset created: {output_path}")
print(f"[INFO] Total rows: {len(df)}")
print(f"\n[INFO] Class distribution:")
print(df["recommendation"].value_counts())
print(f"\n[INFO] First 5 rows:")
print(df.head())