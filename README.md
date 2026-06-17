<div align="center">

![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-lightgrey?style=for-the-badge)

# AI Tutor for Personalized Learning Recommendations

**Group:** AbstractMinds &nbsp;|&nbsp;
**Subject:** Artificial Intelligence Lab &nbsp;|&nbsp;
**Supervisor:** Rajesh Kumar &nbsp;|&nbsp;
**Institution:** IMCS, University of Sindh, Jamshoro
<br/>
</div>

## AbstractMinds

| Member | Roll No | Role | Contribution |
|---|---|---|---|
| Muhammad Ibrahim | 2k23/CSE/94 | Project Lead | Recommendation Engine · Decision Tree · Evaluation · Documentation . Dataset Generation · Report |
| Arsal Jan | 2k23/CSE/34 | UI Developer | Streamlit UI · Tab Layout · Charts Integration · Testing |
| Ali | 2k23/CSE/27 | Utilities & Docs | Helper Utilities · Screenshots · Documentation |

---

## Repository Naming Convention
> This repository follows the naming format required by the Lab Guide:
>
> ```
> ProjectAbbreviation_GroupName
> ```
> ```
> AI_Tutor_PLR_AbstractMinds
> ```
```
A I _ T u t o r _ P  L  R  _ A b s t r a c t M i n d s
│       │         │  │  │    └──────────────────────────┘
│       │         │  │  └─ Recommendations   Team Identity
│       │         │  └─ Learning
│       │         └─ Personalized
└───────┘
  AI Tutor
```

| Component | Full Form | Meaning |
|:---:|:---|:---|
| `AI_Tutor` | Artificial Intelligence Tutor | An intelligent system that acts as a personal tutor |
| `PLR` | Personalized Learning Recommendations | Adaptive content tailored to each student's needs |
| `AbstractMinds` | Our Team Name | The developers behind this project |

---
## 1. Project Overview

This project presents an AI Tutor for Personalized Learning
Recommendations, implemented as a Streamlit based web application.
The system evaluates a student's quiz performance score, response
time, self reported confidence, and prior result and produces a
single, actionable recommendation: review foundational material,
continue practicing, or proceed to the next topic.

Two distinct artificial intelligence methodologies are implemented
and evaluated in parallel:

- **Rule Based Engine** : A Symbolic AI approach using explicit
  IF-THEN inference
- **Decision Tree Classifier** : A supervised Machine Learning
  model implemented using scikit learn

Both methodologies are assessed using standard classification
metrics Accuracy, Precision, Recall, and F1-Score alongside a
Confusion Matrix, with results presented comparatively in the
Evaluation module of the application.

---

## 2. Problem Statement

A numerical quiz score, in isolation, provides limited insight into
a student's true level of understanding. Two students achieving
identical scores may require entirely different interventions, yet
conventional fixed curricula make no such distinction. This project
addresses that limitation by incorporating additional behavioral
indicators response time, self reported confidence, and historical
performance to generate a more accurate, individualized
recommendation.

**System Inputs:**
- Quiz Score (%)
- Response Time (seconds)
- Confidence Level (High / Medium / Low)
- Topic
- Previous Score (%)

**System Outputs:**
- Recommendation (Review Basics / Practice More / Next Topic)
- Next Topic Suggestion
- Practice Question Count
- Revision Flag

---

## 3. AI Methods Used

### Option 1 Rule-Based Engine

A forward chaining inference system implementing the following
decision logic:
```prolog
R1: IF score < 40
    THEN Review Basics

R2: IF 40 <= score < 70
    THEN Practice More

R3: IF score >= 70
    THEN Next Topic
```
### Performance Modifiers

**M1 Slow Response**

```prolog
IF response_time > 80s
THEN add 3 extra practice questions
```

**M2 Low Confidence**

```prolog
IF confidence == "Low"
THEN add 2 extra questions
AND flag for revision
```

**M3 Performance Decline**

```prolog
IF score drop > 10%
THEN flag for revision
```
---
### Option 2 Decision Tree Classifier

A supervised classification model trained on the synthetic student
dataset, with hyperparameters selected to mitigate overfitting on a
relatively small sample size.

- Algorithm: `DecisionTreeClassifier` (scikit learn)
- `max_depth = 4` — constrains model complexity
- `criterion = "gini"` — standard impurity measure for classification
- `random_state = 42` — ensures reproducibility across executions
- 80% training / 20% testing split, stratified by class label

**Feature Set Used for Model Training:**

| Feature | Description |
|---|---|
| `score_pct` | Quiz score percentage |
| `response_time` | Time taken, measured in seconds |
| `confidence_encoded` | High = 2, Medium = 1, Low = 0 |
| `prev_score` | Score from the previous quiz attempt |

---

## 4. Features

- Dynamic switching between Rule-Based and Decision Tree modes
- Color-coded recommendation banner indicating output severity
- Four interactive Plotly visualizations: Bar, Pie, Line, and Radar charts
- Comprehensive explainability panel detailing step by step reasoning
- Direct comparative evaluation of both implemented AI approaches
- Confusion matrix heatmap for detailed performance analysis
- Robust input validation with descriptive error messaging
- On demand model retraining accessible via the sidebar interface
