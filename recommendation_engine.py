# This file contains ALL AI logic for the project maintainability and modularity.
# Functions Implemented:
#   1. load_data(path)                  - Load and validate dataset
#   2. preprocess_data(df)              - Data cleaning and encoding
#   3. run_rules(...)                   - Rule-Based Engine (Option 1)
#   4. generate_explanation(...)        - Natural language explanation
#   5. train_model(df)                  - Train Decision Tree (Option 3)
#   6. run_model(input_dict)            - Make prediction using ML model
#   7. get_tree_explanation(input_dict) - Explain Decision Tree path
#   8. create_visuals(df, result)       - Generate all Plotly charts
#   9. evaluate_model(df)               - Compare both AI approaches         

import os
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from utils.helpers import (
    encode_confidence,
    get_next_topic,
    LOW_SCORE_THRESHOLD,
    HIGH_SCORE_THRESHOLD,
    HIGH_TIME_THRESHOLD
)
# FILE PATHS
# Using os.path.join avoids hardcoded paths.

MODEL_PATH = os.path.join("models", "decision_tree.pkl")
DATA_PATH  = os.path.join("data",   "student_scores.csv")

# FEATURE COLUMN ORDER
# Must be identical during training and prediction.
# If order changes, model predictions will be wrong.
FEATURE_COLS = [
    "score_pct",
    "response_time",
    "confidence_encoded",
    "prev_score"
]

# LABEL ORDER
# Fixed order used for the confusion matrix axes.
# Changing this order will misalign the matrix display.
LABEL_ORDER = ["Review Basics", "Practice More", "Next Topic"]

# RULE ENGINE CONSTANTS
# Defined here so thresholds are easy to find and change by Using meaningful variable names.
PRACTICE_LOW               = 10   # Base questions for Review Basics
PRACTICE_MEDIUM            = 6    # Base questions for Practice More
PRACTICE_HIGH              = 3    # Base questions for Next Topic
EXTRA_QUESTIONS_TIME       = 3    # Added when response time is slow
EXTRA_QUESTIONS_CONFIDENCE = 2    # Added when confidence is Low

# 1. LOAD DATA
# Let user provide or select input data.
# The dataset is the source of training data for the model
# and also powers the charts in Tab 2.

def load_data(path=DATA_PATH):
    """
    Loads the student quiz dataset from a CSV file.

    Uses a try/except block so the app does not crash if the
    file is missing  it returns None instead and the UI
    shows a friendly error message.

    Parameters:
        path (str): File path to the CSV dataset.
                    Defaults to data/student_scores.csv

    Returns:
        pd.DataFrame: Loaded dataset.
        None: If file is not found.
    """
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        print(f"Dataset not found at: {path}")
        print("Run: python data/generate_dataset.py")
        return None

# 2. PREPROCESS DATA
# Core logic for data preparation before model.
# Show intermediate steps where possible.

def preprocess_data(df):
    """
    Cleans and encodes the raw dataset for model training.

    Steps performed:
        1. Drop rows with any missing values.
        2. Encode the 'confidence' column to numeric:
               High   = 2
               Medium = 1
               Low    = 0
        3. Select only the four feature columns needed.
        4. Return X (features) and y (labels) separately.

    Parameters:
        df (pd.DataFrame): Raw dataset from load_data().

    Returns:
        tuple: (X, y)
            X -- pd.DataFrame of feature columns
            y -- pd.Series of recommendation labels
        Returns (None, None) if input df is None.
    """
    if df is None:
        return None, None

    # Remove any incomplete rows to avoid training errors
    df = df.dropna().copy()

    # Encode string confidence to integer for scikit-learn
    df["confidence_encoded"] = df["confidence"].apply(encode_confidence)

    # Select only the features used during training
    X = df[FEATURE_COLS]
    y = df["recommendation"]

    return X, y

# 3. RULE-BASED ENGINE
# Key factors and rules used in decision.
# This function implements forward-chaining IF-THEN logic.
# Every decision is logged in reasoning_steps for display
# in the Explainability tab (Tab 3).

def run_rules(score, confidence, response_time, topic, prev_score=None):
    """
    Applies IF-THEN rules to generate a personalized recommendation.

    PRIMARY RULES (based on quiz score):
        Rule 1: score < 40          --> Review Basics
        Rule 2: 40 <= score < 70    --> Practice More
        Rule 3: score >= 70         --> Next Topic

    MODIFIER RULES (adjust output after primary rule fires):
        Modifier A: response_time > HIGH_TIME_THRESHOLD
                    --> add EXTRA_QUESTIONS_TIME extra questions
        Modifier B: confidence == "Low"
                    --> add EXTRA_QUESTIONS_CONFIDENCE questions
                    --> set revision_needed = True
        Modifier C: score dropped more than 10% from prev_score
                    --> set revision_needed = True

    Parameters:
        score         (float): Quiz score percentage (0 to 100).
        confidence    (str):   Self-reported level: High/Medium/Low.
        response_time (float): Time taken in seconds.
        topic         (str):   The quiz topic name.
        prev_score    (float): Previous quiz score (optional).

    Returns:
        dict: {
            "recommendation":     str,   -- Review Basics / Practice More / Next Topic
            "rule_triggered":     str,   -- which primary rule fired
            "modifiers_applied":  list,  -- list of modifier descriptions
            "next_topic":         str,   -- suggested next topic to study
            "practice_questions": int,   -- total practice questions recommended
            "revision_needed":    bool,  -- whether revision is flagged
            "reasoning_steps":    list,  -- full step-by-step log
            "slow_response":      bool   -- whether response time was slow
        }
    """
    # Initialize tracking variables
    reasoning_steps   = []    # Step by step log shown in Explainability tab
    modifiers_applied = []    # List of modifier descriptions
    revision_needed   = False
    extra_questions   = 0

    # STEP 1: Apply Primary Rule based on score

    reasoning_steps.append(f"Step 1 -- Checking score: {score}%")

    if score < LOW_SCORE_THRESHOLD:
        # Rule 1: Low score student needs to review fundamentals
        recommendation = "Review Basics"
        rule_triggered = (
            f"Rule 1: score ({score}%) < {LOW_SCORE_THRESHOLD} "
            f"--> Review Basics"
        )
        practice_base  = PRACTICE_LOW
        reasoning_steps.append(
            f"  --> Rule 1 triggered. Score below {LOW_SCORE_THRESHOLD}%. "
            f"Student needs to review fundamentals."
        )

    elif score < HIGH_SCORE_THRESHOLD:
        # Rule 2: Mid score  student understands basics, needs practice
        recommendation = "Practice More"
        rule_triggered = (
            f"Rule 2: {LOW_SCORE_THRESHOLD} <= score ({score}%) "
            f"< {HIGH_SCORE_THRESHOLD} --> Practice More"
        )
        practice_base  = PRACTICE_MEDIUM
        reasoning_steps.append(
            f"  --> Rule 2 triggered. Score between "
            f"{LOW_SCORE_THRESHOLD}% and {HIGH_SCORE_THRESHOLD}%. "
            f"Student understands basics but needs more practice."
        )

    else:
        # Rule 3: High score student is ready to advance
        recommendation = "Next Topic"
        rule_triggered = (
            f"Rule 3: score ({score}%) >= {HIGH_SCORE_THRESHOLD} "
            f"--> Next Topic"
        )
        practice_base  = PRACTICE_HIGH
        reasoning_steps.append(
            f"  --> Rule 3 triggered. Score above {HIGH_SCORE_THRESHOLD}%. "
            f"Student is ready to advance."
        )

    # STEP 2: Modifier A -- Response Time Check
    # Slow response suggests confusion or difficulty recalling.

    reasoning_steps.append(
        f"Step 2 -- Checking response time: {response_time}s "
        f"(threshold: {HIGH_TIME_THRESHOLD}s)"
    )

    if response_time > HIGH_TIME_THRESHOLD:
        extra_questions += EXTRA_QUESTIONS_TIME
        slow_flag        = True
        modifiers_applied.append(
            f"Modifier A: response_time ({response_time}s) > "
            f"{HIGH_TIME_THRESHOLD}s --> +{EXTRA_QUESTIONS_TIME} extra questions"
        )
        reasoning_steps.append(
            f"  --> High response time detected. "
            f"Adding {EXTRA_QUESTIONS_TIME} extra practice questions."
        )
    else:
        slow_flag = False
        reasoning_steps.append(
            "  --> Response time is acceptable. No extra questions added."
        )

    # STEP 3: Modifier B -- Confidence Level Check
    # Low confidence indicates student is unsure of the material.

    reasoning_steps.append(
        f"Step 3 -- Checking confidence level: {confidence}"
    )

    if confidence == "Low":
        extra_questions += EXTRA_QUESTIONS_CONFIDENCE
        revision_needed  = True
        modifiers_applied.append(
            f"Modifier B: confidence is Low --> "
            f"+{EXTRA_QUESTIONS_CONFIDENCE} questions, revision flagged"
        )
        reasoning_steps.append(
            f"  --> Low confidence detected. Revision flagged. "
            f"Adding {EXTRA_QUESTIONS_CONFIDENCE} extra questions."
        )
    elif confidence == "Medium":
        reasoning_steps.append(
            "  --> Medium confidence. No revision flagged from confidence."
        )
    else:
        reasoning_steps.append(
            "  --> High confidence. Positive signal. No modifier applied."
        )

    # STEP 4: Modifier C -- Score Trend Check
    # A significant score drop suggests knowledge regression.

    reasoning_steps.append(
        f"Step 4 -- Checking score trend. "
        f"Previous score: {prev_score if prev_score is not None else 'N/A'}"
    )

    if prev_score is not None:
        score_change = score - prev_score
        if score_change < -10:
            # Score dropped more than 10 points -- flag revision
            revision_needed = True
            modifiers_applied.append(
                f"Modifier C: score dropped {abs(score_change):.1f}% "
                f"({prev_score}% --> {score}%) --> revision flagged"
            )
            reasoning_steps.append(
                f"  --> Score dropped {abs(score_change):.1f}% "
                f"since last attempt. Revision recommended."
            )
        elif score_change > 0:
            reasoning_steps.append(
                f"  --> Score improved by {score_change:.1f}% "
                f"since last attempt. Positive trend."
            )
        else:
            reasoning_steps.append(
                "  --> Score is stable compared to last attempt."
            )
    else:
        reasoning_steps.append(
            "  --> No previous score provided. Skipping trend check."
        )

    # STEP 5: Calculate Final Practice Question Count
    # Base count from primary rule + extras from modifiers.

    practice_questions = practice_base + extra_questions
    reasoning_steps.append(
        f"Step 5 -- Final practice questions: "
        f"{practice_base} (base) + {extra_questions} (modifiers) "
        f"= {practice_questions}"
    )

    # STEP 6: Determine Next Topic
    # Only advance if the primary rule is Next Topic.

    if recommendation == "Next Topic":
        next_topic = get_next_topic(topic)
        reasoning_steps.append(
            f"Step 6 -- Student is ready to advance. "
            f"Next topic: {next_topic}"
        )
    else:
        next_topic = topic
        reasoning_steps.append(
            f"Step 6 -- Student should stay on current topic: {topic}"
        )

    return {
        "recommendation":     recommendation,
        "rule_triggered":     rule_triggered,
        "modifiers_applied":  modifiers_applied,
        "next_topic":         next_topic,
        "practice_questions": practice_questions,
        "revision_needed":    revision_needed,
        "reasoning_steps":    reasoning_steps,
        "slow_response":      slow_flag
    }

# 4. EXPLANATION GENERATOR
# Add short natural-language explanation.
# Show why the app produced that output.
#
# Converts the structured result dict into a readable paragraph
# that any student can understand without technical knowledge.

def generate_explanation(result, context):
    """
    Builds a plain-English explanation paragraph for a recommendation.

    Combines input signals (score, time, confidence, trend) and the
    recommendation result into a multi-sentence paragraph.
    Shown in both Tab 1 (quick) and Tab 3 (full explainability).

    Parameters:
        result  (dict): Output from run_rules() or run_model().
        context (dict): Original user inputs:
                        score_pct, response_time, confidence,
                        topic, prev_score.

    Returns:
        str: Human-readable explanation paragraph.
    """
    # Extract context values with safe defaults
    score         = context.get("score_pct",     0)
    response_time = context.get("response_time", 0)
    confidence    = context.get("confidence",    "Medium")
    topic         = context.get("topic",         "this topic")
    prev_score    = context.get("prev_score",    None)

    # Extract result values with safe defaults
    recommendation  = result.get("recommendation",    "Practice More")
    next_topic      = result.get("next_topic",         topic)
    practice_count  = result.get("practice_questions", 5)
    revision_needed = result.get("revision_needed",    False)
    modifiers       = result.get("modifiers_applied",  [])
    slow_response   = result.get("slow_response",      False)

    sentences = []

    # -- Sentence 1: Score interpretation
    if score < LOW_SCORE_THRESHOLD:
        sentences.append(
            f"Your score of {score}% on {topic} indicates that the core "
            f"concepts have not been fully understood yet."
        )
    elif score < HIGH_SCORE_THRESHOLD:
        sentences.append(
            f"Your score of {score}% on {topic} shows a developing "
            f"understanding -- you have grasped the basics but still "
            f"need more practice."
        )
    else:
        sentences.append(
            f"Your score of {score}% on {topic} is strong, demonstrating "
            f"solid understanding of the material."
        )

    # -- Sentence 2: Response time observation
    if slow_response:
        sentences.append(
            f"Your response time of {response_time} seconds was above "
            f"the expected threshold, which may indicate uncertainty "
            f"or difficulty recalling concepts quickly."
        )
    else:
        sentences.append(
            f"Your response time of {response_time} seconds was within "
            f"an acceptable range."
        )

    # -- Sentence 3: Confidence observation (only for Low or High)
    if confidence == "Low":
        sentences.append(
            f"You reported Low confidence, which reinforces the "
            f"recommendation to revisit this material before moving forward."
        )
    elif confidence == "High":
        sentences.append(
            f"You reported High confidence, which is consistent "
            f"with your score."
        )

    # -- Sentence 4: Score trend (only if previous score is available)
    if prev_score is not None:
        change = score - prev_score
        if change < -10:
            sentences.append(
                f"Your score dropped by {abs(change):.1f}% compared to "
                f"your previous attempt ({prev_score}%), suggesting some "
                f"concepts may need revision."
            )
        elif change > 5:
            sentences.append(
                f"Your score improved by {change:.1f}% from your last "
                f"attempt ({prev_score}%), which shows good progress."
            )

    # -- Sentence 5: Final recommendation action
    if recommendation == "Review Basics":
        sentences.append(
            f"The AI Tutor recommends reviewing the basics of {topic} "
            f"before attempting further practice."
        )
    elif recommendation == "Practice More":
        sentences.append(
            f"The AI Tutor recommends completing {practice_count} "
            f"practice questions on {topic} to strengthen your understanding."
        )
    else:
        sentences.append(
            f"The AI Tutor recommends moving on to {next_topic}. "
            f"You are ready to advance."
        )

    # -- Sentence 6: Revision flag (if triggered)
    if revision_needed:
        sentences.append(
            f"A revision session for {topic} is also recommended "
            f"before your next quiz."
        )

    # -- Sentence 7: Modifiers summary (if any modifiers fired)
    if modifiers:
        sentences.append(
            "Additional adjustments were made based on: "
            + "; ".join(modifiers) + "."
        )

    # Join all sentences into one paragraph
    return " ".join(sentences)

# 5. TRAIN MODEL
# Implement main model -- Option 3 (ML).
# Show performance indicators after training.
#
# Trains a DecisionTreeClassifier on the synthetic dataset.
# Saves the trained model to disk using pickle so it does
# not need to be retrained on every app reload.

def train_model(df):
    """
    Trains a DecisionTreeClassifier on the full student dataset.

    Training steps:
        1. Preprocess data (encode, drop nulls).
        2. Split 80% train / 20% test (stratified by label).
        3. Train DecisionTreeClassifier with max_depth=4.
        4. Evaluate on test set (accuracy, precision, recall, F1).
        5. Generate confusion matrix and classification report.
        6. Save trained model to models/decision_tree.pkl.

    Why max_depth=4?
        Deeper trees memorize training data (overfitting).
        max_depth=4 forces the tree to learn general patterns,
        which is appropriate for a 200-row synthetic dataset.

    Why stratify=y?
        Ensures all three classes (Review Basics, Practice More,
        Next Topic) appear in both train and test splits even
        when the dataset is small.

    Parameters:
        df (pd.DataFrame): Full dataset from load_data().

    Returns:
        dict: {
            "status":           str,          -- "success" or "error"
            "message":          str,          -- description of outcome
            "accuracy":         float,
            "precision":        float,
            "recall":           float,
            "f1":               float,
            "confusion_matrix": np.ndarray,
            "labels":           list,
            "train_size":       int,
            "test_size":        int,
            "report":           str           -- full sklearn report
        }
    """
    # Step 1: Preprocess
    X, y = preprocess_data(df)

    if X is None or len(X) == 0:
        return {
            "status":  "error",
            "message": "Dataset is empty or could not be loaded."
        }

    # Step 2: Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = 0.20,
        random_state = 42,
        stratify     = y
    )

    # Step 3: Train Decision Tree
    model = DecisionTreeClassifier(
        max_depth    = 4,       # Prevents overfitting on small dataset
        random_state = 42,      # Makes results reproducible every run
        criterion    = "gini"   # Gini impurity -- standard for classification
    )
    model.fit(X_train, y_train)

    # Step 4: Evaluate on test set
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)

    # zero_division=0 suppresses warnings when a class has no predictions
    prec = precision_score(
        y_test, y_pred, average="weighted", zero_division=0
    )
    rec  = recall_score(
        y_test, y_pred, average="weighted", zero_division=0
    )
    f1   = f1_score(
        y_test, y_pred, average="weighted", zero_division=0
    )

    # Step 5: Confusion matrix and full report
    # Rows = actual, columns = predicted
    cm     = confusion_matrix(y_test, y_pred, labels=LABEL_ORDER)
    report = classification_report(
        y_test, y_pred, labels=LABEL_ORDER, zero_division=0
    )

    # Step 6: Save model to disk
    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Training accuracy: {acc:.4f}")

    return {
        "status":           "success",
        "message":          f"Model trained and saved to {MODEL_PATH}",
        "accuracy":         round(acc,  4),
        "precision":        round(prec, 4),
        "recall":           round(rec,  4),
        "f1":               round(f1,   4),
        "confusion_matrix": cm,
        "labels":           LABEL_ORDER,
        "train_size":       len(X_train),
        "test_size":        len(X_test),
        "report":           report
    }

# 6. RUN MODEL (PREDICT)
# run_model_or_algorithm(data, params).
# Show intermediate steps (reasoning_steps).
#
# Loads the saved Decision Tree and predicts a recommendation
# for a single student's input.
# Auto-trains the model if no saved file is found.

def run_model(input_dict):
    """
    Loads the trained Decision Tree and predicts a recommendation.

    If models/decision_tree.pkl does not exist, automatically
    trains the model first using the default dataset path.

    Returns the same dict structure as run_rules() so the UI
    (app.py) can treat both outputs identically.

    Parameters:
        input_dict (dict): {
            "score_pct":     float,   -- quiz score 0 to 100
            "response_time": float,   -- seconds taken
            "confidence":    str,     -- High / Medium / Low
            "prev_score":    float,   -- previous quiz score
            "topic":         str      -- quiz topic
        }

    Returns:
        dict: Same structure as run_rules() output.
              recommendation, rule_triggered, modifiers_applied,
              next_topic, practice_questions, revision_needed,
              reasoning_steps, slow_response.
    """
    # Auto-train if saved model is missing
    if not os.path.exists(MODEL_PATH):
        print("No saved model found. Training now...")
        df = load_data()
        if df is None:
            # Return safe fallback if dataset is also missing
            return {
                "recommendation":     "Practice More",
                "rule_triggered":     "Fallback: dataset not found",
                "modifiers_applied":  [],
                "next_topic":         input_dict.get("topic", "OOP"),
                "practice_questions": 5,
                "revision_needed":    False,
                "reasoning_steps":    ["[ERROR] Dataset missing."],
                "slow_response":      False
            }
        train_model(df)

    # Load model from disk
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # Extract and encode input values
    score              = input_dict.get("score_pct",     50.0)
    response_time      = input_dict.get("response_time", 60.0)
    confidence         = input_dict.get("confidence",    "Medium")
    prev_score         = input_dict.get("prev_score",    50.0)
    topic              = input_dict.get("topic",         "OOP")
    confidence_encoded = encode_confidence(confidence)

    # Build single-row DataFrame matching training feature order
    feature_row = pd.DataFrame(
        [[score, response_time, confidence_encoded, prev_score]],
        columns=FEATURE_COLS
    )

    # Run prediction
    prediction    = model.predict(feature_row)[0]
    slow_response = response_time > HIGH_TIME_THRESHOLD

    # Build reasoning steps for Explainability tab
    reasoning_steps = [
        f"Input features sent to model:",
        f"  score_pct          = {score}%",
        f"  response_time      = {response_time}s",
        f"  confidence         = {confidence} (encoded = {confidence_encoded})",
        f"  prev_score         = {prev_score}%",
        f"Decision Tree prediction: {prediction}"
    ]

    # Assign base practice count based on prediction
    if prediction == "Review Basics":
        practice_questions = PRACTICE_LOW
        revision_needed    = True
    elif prediction == "Practice More":
        practice_questions = PRACTICE_MEDIUM
        revision_needed    = False
    else:
        practice_questions = PRACTICE_HIGH
        revision_needed    = False

    # Apply same modifiers as rule engine for consistency
    if slow_response:
        practice_questions += EXTRA_QUESTIONS_TIME
        reasoning_steps.append(
            f"Slow response time (+{EXTRA_QUESTIONS_TIME} extra questions)"
        )

    if confidence == "Low":
        practice_questions += EXTRA_QUESTIONS_CONFIDENCE
        revision_needed     = True
        reasoning_steps.append(
            f"Low confidence (+{EXTRA_QUESTIONS_CONFIDENCE} questions, "
            f"revision flagged)"
        )

    # Determine next topic
    next_topic = get_next_topic(topic) if prediction == "Next Topic" else topic

    reasoning_steps.append(
        f"Final output: recommendation={prediction}, "
        f"practice_questions={practice_questions}, "
        f"next_topic={next_topic}"
    )

    return {
        "recommendation":     prediction,
        "rule_triggered":     f"Decision Tree predicted: {prediction}",
        "modifiers_applied":  [],
        "next_topic":         next_topic,
        "practice_questions": practice_questions,
        "revision_needed":    revision_needed,
        "reasoning_steps":    reasoning_steps,
        "slow_response":      slow_response
    }

# 7. TREE EXPLANATION
# Display key factors/features/rules used.
# Show why the app produced that output.
#
# Describes the likely split path the Decision Tree followed
# for the given input in plain English.
# Displayed in Tab 3 (Explainability) under "Tree Split Path".

def get_tree_explanation(input_dict):
    """
    Returns a plain-English description of the Decision Tree
    split path that likely produced the prediction.

    Based on the known tree structure (max_depth=4, trained on
    rules-labeled data), score_pct is always the most influential
    feature at the root node.

    Parameters:
        input_dict (dict): Same input dict as run_model().

    Returns:
        str: Multi-line split path description.
    """
    score              = input_dict.get("score_pct",     50.0)
    response_time      = input_dict.get("response_time", 60.0)
    confidence         = input_dict.get("confidence",    "Medium")
    prev_score         = input_dict.get("prev_score",    50.0)
    confidence_encoded = encode_confidence(confidence)

    lines = [
        "Decision Tree Split Path:",
        "",
        "  Input features:",
        f"    score_pct          = {score}",
        f"    response_time      = {response_time}",
        f"    confidence_encoded = {confidence_encoded} ({confidence})",
        f"    prev_score         = {prev_score}",
        "",
        "  Most influential feature: score_pct (root node split)"
    ]

    if score < LOW_SCORE_THRESHOLD:
        lines.append(
            f"  Split 1: score_pct ({score}) < {LOW_SCORE_THRESHOLD} "
            f"--> LEFT branch --> Review Basics"
        )

    elif score < HIGH_SCORE_THRESHOLD:
        lines.append(
            f"  Split 1: score_pct ({score}) >= {LOW_SCORE_THRESHOLD} "
            f"--> RIGHT branch"
        )
        lines.append(
            f"  Split 2: score_pct ({score}) < {HIGH_SCORE_THRESHOLD} "
            f"--> LEFT branch --> Practice More"
        )
        # Secondary split on confidence or time in mid-range
        if confidence_encoded == 0:
            lines.append(
                f"  Split 3: confidence_encoded = 0 (Low) "
                f"--> may adjust toward Review Basics"
            )
        elif response_time > HIGH_TIME_THRESHOLD:
            lines.append(
                f"  Split 3: response_time ({response_time}s) > "
                f"{HIGH_TIME_THRESHOLD}s --> reinforces Practice More"
            )

    else:
        lines.append(
            f"  Split 1: score_pct ({score}) >= {LOW_SCORE_THRESHOLD} "
            f"--> RIGHT branch"
        )
        lines.append(
            f"  Split 2: score_pct ({score}) >= {HIGH_SCORE_THRESHOLD} "
            f"--> RIGHT branch --> Next Topic"
        )

    return "\n".join(lines)

# 8. CREATE VISUALS
# Visual UI module (compulsory).
# Required: charts (bar/line/pie/scatter), tables.
#
# Produces four Plotly figures returned as a dict.
# All chart titles use text labels (no emoji) for consistency.
# Charts are displayed in Tab 2 of the Streamlit app.

def create_visuals(df, result):
    """
    Creates all four Plotly charts for the Streamlit UI.

    Charts produced:
        1. bar_chart   -- Average quiz score per topic
        2. pie_chart   -- Recommendation label distribution
        3. line_chart  -- Score trend (current vs previous)
        4. radar_chart -- Student inputs vs dataset averages

    Parameters:
        df     (pd.DataFrame): Full dataset from load_data().
        result (dict):         Input context dict containing
                               score_pct_input, response_time_input,
                               prev_score_input for radar chart.

    Returns:
        dict: {
            "bar_chart":   plotly Figure,
            "pie_chart":   plotly Figure,
            "line_chart":  plotly Figure,
            "radar_chart": plotly Figure
        }
        Returns empty dict if df is None or empty.
    """
    charts = {}

    # Guard: return empty if no data available
    if df is None or df.empty:
        return charts

    # CHART 1: BAR CHART 
    # Shows which topics students perform best and worst on.
    # Charts (bar).

    topic_avg = (
        df.groupby("topic")["score_pct"]
        .mean()
        .reset_index()
        .sort_values("score_pct", ascending=False)
    )
    topic_avg.columns = ["Topic", "Average Score (%)"]

    bar_fig = px.bar(
        topic_avg,
        x     = "Topic",
        y     = "Average Score (%)",
        title = "Average Quiz Score by Topic",
        color = "Average Score (%)",
        color_continuous_scale = "Blues",
        text  = "Average Score (%)"
    )
    bar_fig.update_traces(
        texttemplate = "%{text:.1f}%",
        textposition = "outside"
    )
    bar_fig.update_layout(
        xaxis_title         = "Topic",
        yaxis_title         = "Average Score (%)",
        yaxis_range         = [0, 110],
        coloraxis_showscale = False,
        plot_bgcolor        = "rgba(0,0,0,0)",
        paper_bgcolor       = "rgba(0,0,0,0)"
    )
    charts["bar_chart"] = bar_fig

    # CHART 2: PIE CHART -- Recommendation Distribution
    # Shows how many students fall into each recommendation class.
    # Charts (pie).
    
    rec_counts = df["recommendation"].value_counts().reset_index()
    rec_counts.columns = ["Recommendation", "Count"]

    pie_fig = px.pie(
        rec_counts,
        names  = "Recommendation",
        values = "Count",
        title  = "Recommendation Distribution in Dataset",
        color  = "Recommendation",
        color_discrete_map = {
            "Review Basics": "#FF4B4B",
            "Practice More": "#FFA500",
            "Next Topic":    "#00C853"
        },
        hole   = 0.35    # Donut style for cleaner look
    )
    pie_fig.update_traces(
        textposition = "inside",
        textinfo     = "percent+label"
    )
    pie_fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)"
    )
    charts["pie_chart"] = pie_fig

    # CHART 3: LINE CHART -- Score Trend
    # Compares current score vs previous score for first 40 rows.
    # Charts (line).
    
    sample_df = df.sort_values("student_id").head(40).copy()

    line_fig = px.line(
        sample_df,
        x      = "student_id",
        y      = ["score_pct", "prev_score"],
        title  = "Score Trend -- Current vs Previous (First 40 Students)",
        labels = {
            "student_id": "Student ID",
            "value":      "Score (%)",
            "variable":   "Score Type"
        },
        color_discrete_map = {
            "score_pct":  "#1f77b4",
            "prev_score": "#ff7f0e"
        }
    )
    line_fig.update_layout(
        xaxis_title   = "Student ID",
        yaxis_title   = "Score (%)",
        yaxis_range   = [0, 105],
        plot_bgcolor  = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)",
        legend_title  = "Score Type"
    )
    charts["line_chart"] = line_fig
 
    # CHART 4: RADAR CHART -- Student vs Dataset Average
    # Compares the current student's three key inputs against
    # the overall dataset averages on a normalized 0-100 scale.
    # Charts (scatter/radar).

    # Calculate dataset averages
    avg_score = round(df["score_pct"].mean(), 1)
    avg_time  = round(df["response_time"].mean(), 1)
    avg_prev  = round(df["prev_score"].mean(), 1)

    # Get student's input values (fall back to dataset avg if missing)
    student_score = result.get("score_pct_input",      avg_score)
    student_time  = result.get("response_time_input",  avg_time)
    student_prev  = result.get("prev_score_input",     avg_prev)

    # Invert response time: faster = higher score on radar
    # (180 seconds = maximum expected time = 0 on radar)
    student_time_score = round(max(0, 100 - (student_time / 180 * 100)), 1)
    avg_time_score     = round(max(0, 100 - (avg_time    / 180 * 100)), 1)

    categories = ["Quiz Score", "Response Speed", "Previous Score"]

    radar_fig = go.Figure()

    # Student's performance trace
    radar_fig.add_trace(go.Scatterpolar(
        r          = [student_score, student_time_score, student_prev],
        theta      = categories,
        fill       = "toself",
        name       = "Your Performance",
        line_color = "#1f77b4"
    ))

    # Dataset average trace for comparison
    radar_fig.add_trace(go.Scatterpolar(
        r          = [avg_score, avg_time_score, avg_prev],
        theta      = categories,
        fill       = "toself",
        name       = "Dataset Average",
        line_color = "#ff7f0e",
        opacity    = 0.6
    ))

    radar_fig.update_layout(
        title         = "Your Performance vs Dataset Average",
        polar         = dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend    = True,
        paper_bgcolor = "rgba(0,0,0,0)"
    )
    charts["radar_chart"] = radar_fig

    return charts

# 9. EVALUATE MODEL
# Evaluation module.
# Required: 2-3 performance indicators, compare two approaches.
#
# Evaluates BOTH the Decision Tree AND the Rule-Based Engine
# on the same 20% test split so results are directly comparable.
# This satisfies the Lab Guide requirement to compare at least
# two settings or approaches.

def evaluate_model(df):
    """
    Evaluates both AI approaches on a held-out 20% test split.

    Decision Tree metrics computed using sklearn.
    Rule-Based metrics computed by running run_rules() on every
    test row and comparing predicted labels to actual labels.

    Both use the same random_state=42 split so comparison is fair.

    Parameters:
        df (pd.DataFrame): Full dataset from load_data().

    Returns:
        dict: {
            "tree_accuracy":    float,
            "tree_precision":   float,
            "tree_recall":      float,
            "tree_f1":          float,
            "rules_accuracy":   float,
            "rules_precision":  float,
            "rules_recall":     float,
            "rules_f1":         float,
            "confusion_matrix": np.ndarray,  -- Decision Tree only
            "labels":           list,
            "report":           str,         -- Decision Tree only
            "train_size":       int,
            "test_size":        int
        }
        Returns zeros if df is None or empty.
    """
    # Guard: return zero metrics if dataset is unavailable
    if df is None or df.empty:
        return {
            "tree_accuracy":    0.0, "tree_precision":  0.0,
            "tree_recall":      0.0, "tree_f1":         0.0,
            "rules_accuracy":   0.0, "rules_precision": 0.0,
            "rules_recall":     0.0, "rules_f1":        0.0,
            "confusion_matrix": None,
            "labels":           [],
            "report":           "",
            "train_size":       0,
            "test_size":        0
        }

    # Preprocess and split using same seed as train_model()
    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = 0.20,
        random_state = 42,
        stratify     = y
    )
    # DECISION TREE EVALUATION
    # Load saved model (or train first if missing).

    if not os.path.exists(MODEL_PATH):
        train_model(df)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    y_pred_tree = model.predict(X_test)

    tree_acc  = accuracy_score(y_test, y_pred_tree)
    tree_prec = precision_score(
        y_test, y_pred_tree, average="weighted", zero_division=0
    )
    tree_rec  = recall_score(
        y_test, y_pred_tree, average="weighted", zero_division=0
    )
    tree_f1   = f1_score(
        y_test, y_pred_tree, average="weighted", zero_division=0
    )
    cm     = confusion_matrix(y_test, y_pred_tree, labels=LABEL_ORDER)
    report = classification_report(
        y_test, y_pred_tree, labels=LABEL_ORDER, zero_division=0
    )

    # RULE-BASED ENGINE EVALUATION
    # Re-attach original columns to test set so we can pass
    # topic, confidence and other features to run_rules() function.

    test_indices = X_test.index
    df_test      = df.loc[test_indices].copy()

    y_pred_rules = []
    for _, row in df_test.iterrows():
        rule_result = run_rules(
            score         = row["score_pct"],
            confidence    = row["confidence"],
            response_time = row["response_time"],
            topic         = row["topic"],
            prev_score    = row["prev_score"]
        )
        y_pred_rules.append(rule_result["recommendation"])

    rules_acc  = accuracy_score(y_test, y_pred_rules)
    rules_prec = precision_score(
        y_test, y_pred_rules, average="weighted", zero_division=0
    )
    rules_rec  = recall_score(
        y_test, y_pred_rules, average="weighted", zero_division=0
    )
    rules_f1   = f1_score(
        y_test, y_pred_rules, average="weighted", zero_division=0
    )

    return {
        "tree_accuracy":    round(tree_acc,   4),
        "tree_precision":   round(tree_prec,  4),
        "tree_recall":      round(tree_rec,   4),
        "tree_f1":          round(tree_f1,    4),
        "rules_accuracy":   round(rules_acc,  4),
        "rules_precision":  round(rules_prec, 4),
        "rules_recall":     round(rules_rec,  4),
        "rules_f1":         round(rules_f1,   4),
        "confusion_matrix": cm,
        "labels":           LABEL_ORDER,
        "report":           report,
        "train_size":       len(X_train),
        "test_size":        len(X_test)
    }
