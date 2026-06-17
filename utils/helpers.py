# utils/helpers.py
# Shared helper functions and constants used across the project.
# Imported by both app.py and recommendation_engine.py.
#
# Keeping shared logic here avoids duplication and ensures
# thresholds are defined in exactly one place.
# Use meaningful variable/function names.
# Keep functions short and reusable.
#
# Contents:
#   1. Constants       -- score and time thresholds
#   2. Topic list      -- valid topics for dropdown and dataset
#   3. Encoding        -- confidence string to int conversion
#   4. Decoding        -- recommendation label to display string
#   5. Validation      -- input checking before AI runs
#   6. Color mapping   -- UI color per recommendation type

# 1. CONSTANTS
# Thresholds used by the Rule-Based Engine in run_rules().
# Defined here (not inside the function) so they can be
# imported by both recommendation_engine.py and app.py.
# Avoid hardcoded values inside functions.

# Score thresholds -- determine which primary rule fires
LOW_SCORE_THRESHOLD  = 40    # score < 40  --> Review Basics 
HIGH_SCORE_THRESHOLD = 70    # score >= 70 --> Next Topic
                              # 40 <= score < 70 --> Practice More

# Response time threshold -- above this is considered slow
HIGH_TIME_THRESHOLD  = 80    # seconds; triggers Modifier A in run_rules()

# Flag: whether low confidence penalty is active
# Currently used in generate_explanation() via run_rules() logic
LOW_CONFIDENCE_PENALTY = True

# 2. TOPIC LIST
# Single source of truth for all valid quiz topics.
# Used in:
#   - app.py            : dropdown options in Tab 1
#   - generate_dataset  : topic column values
#   - get_next_topic()  : cycling to the next topic
# To add a new topic, add it to this list only.

def get_topic_list():
    """
    Returns the ordered list of valid quiz topics.

    Used by the UI dropdown and the dataset generator.
    Add new topics here if the curriculum expands.

    Returns:
        list: Ordered list of topic name strings.
    """
    return [
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


def get_next_topic(current_topic):
    """
    Returns the suggested next topic after the current one.

    Cycles back to the first topic if the current topic is
    the last in the list (circular curriculum).

    Used by run_rules() and run_model() to populate the
    "next_topic" field in the result dictionary.

    Parameters:
        current_topic (str): The topic the student just completed.

    Returns:
        str: The next topic name.
             Returns "OOP" as fallback if topic is not found.
    """
    topics = get_topic_list()

    if current_topic in topics:
        current_index = topics.index(current_topic)
        # If not the last topic, return the next one
        if current_index + 1 < len(topics):
            return topics[current_index + 1]
        else:
            # Last topic -- loop back to the beginning
            return topics[0]

    # Fallback if topic string is not recognized
    return "OOP"


# 3. ENCODING
# Converts human-readable confidence strings to integers
# so scikit-learn's DecisionTreeClassifier can process them.
# Preprocess data before model training.

def encode_confidence(level):
    """
    Converts a confidence level string to a numeric value.

    The Decision Tree requires numeric features.
    This encoding preserves the natural ordering:
        High > Medium > Low  maps to  2 > 1 > 0

    Parameters:
        level (str): "High", "Medium", or "Low"

    Returns:
        int: 2 for High, 1 for Medium, 0 for Low.
             Returns 1 (Medium) as default for unknown input.
    """
    mapping = {"High": 2, "Medium": 1, "Low": 0}
    # .get() returns 1 (Medium) if level is not in the dict
    return mapping.get(level, 1)


def decode_recommendation(label):
    """
    Converts an internal recommendation label to a display string.

    Used in the UI result banner (Tab 1) and explainability
    panel (Tab 3) to show a clean, readable label.

    Parameters:
        label (str): "Review Basics", "Practice More", or "Next Topic"

    Returns:
        str: Display string with professional prefix tag.
             Returns original label unchanged if not recognized.
    """
    display = {
        "Review Basics": "Review Basics for stronger foundation",
        "Practice More": "Practice More for better understanding",
        "Next Topic":    "Next Topic: Keep up the great work!"
    }
    return display.get(label, label)

# 4. INPUT VALIDATION
# Validates all user inputs before passing them to the AI.
# Called in app.py immediately after the submit button click.
# Lab Guide [A]: Validate user input and show clear error messages.

def validate_inputs(score, response_time, topic):
    """
    Validates user inputs before the AI engine runs.

    Checks three conditions:
        1. Score must be between 0 and 100 (inclusive).
        2. Response time must be greater than 0 seconds.
        3. Topic must exist in the valid topic list.

    If any check fails, returns False with a descriptive
    error message that is shown to the user via st.error().

    Parameters:
        score         (float): Quiz score percentage.
        response_time (float): Time taken in seconds.
        topic         (str):   Selected quiz topic.

    Returns:
        tuple: (is_valid, error_message)
            is_valid      (bool): True if all inputs pass.
            error_message (str):  Empty string if valid,
                                  description if invalid.
    """
    # Check 1: Score range
    if not (0 <= score <= 100):
        return False, "Score must be between 0 and 100."

    # Check 2: Response time must be positive
    if response_time <= 0:
        return False, "Response time must be greater than 0 seconds."

    # Check 3: Topic must be in the recognized list
    if topic not in get_topic_list():
        return False, (
            f"Topic '{topic}' is not recognized. "
            f"Please select from the dropdown."
        )

    # All checks passed
    return True, ""

# 5. COLOR MAPPING
# Maps each recommendation label to a hex color code.
# Used in app.py for the result banner border and metric cards.
# Color coding gives instant visual signal to the student:
#   Red    = needs attention (Review Basics)
#   Orange = making progress (Practice More)
#   Green  = ready to advance (Next Topic)
# Preprocess data before model training.

def get_recommendation_color(recommendation):
    """
    Returns the hex color code for a recommendation label.

    Colors are consistent across all UI elements (banners,
    metric cards, pie chart slices) so the student gets
    a consistent visual signal throughout the app.

    Parameters:
        recommendation (str): The recommendation label string.

    Returns:
        str: Hex color code string.
             Returns grey (#AAAAAA) for unrecognized labels.
    """
    colors = {
        "Review Basics": "#FF4B4B",    # Red    -- needs attention
        "Practice More": "#FFA500",    # Orange -- making progress
        "Next Topic":    "#00C853"     # Green  -- ready to advance
    }
    return colors.get(recommendation, "#AAAAAA")
