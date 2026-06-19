<div align="center">

![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-lightgrey?style=for-the-badge)

# AI Tutor for Personalized Learning Recommendations
<div align="center">
</div>

**Group:** *AbstractMinds* &nbsp;|&nbsp;
**Subject:** *Artificial Intelligence Lab* &nbsp;|&nbsp;
**Supervisor:** *Rajesh Kumar* &nbsp;|&nbsp;
**Institution:** *IMCS, University of Sindh, Jamshoro*

*For a quick understanding of the project including Problem Statement, PEAS Framework, Methodology, AI Implementation, Results, Limitations, and Conclusion* **[*Click here→*](report/final_report.md)**
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
---
## 5. Project Structure

```
AI_Tutor_PLR_AbstractMinds/
│
├── app.py                        -- Streamlit main entry point
├── recommendation_engine.py      -- All AI logic
│
├── data/
│   ├── generate_dataset.py       -- Script to generate synthetic CSV
│   └── student_scores.csv        -- Synthetic dataset (200 rows)
│
├── models/
│   └── decision_tree.pkl         -- Saved trained model (auto-generated)
│
├── utils/
│   └── helpers.py                -- Shared constants and utilities
│
├── report/
│   └── final_report.md           -- Short project report
│
├── screenshots/
│   ├── screenshot_input.png
│   ├── screenshot_charts.png
│   ├── screenshot_explainability.png
│   └── screenshot_evaluation.png
│
├── requirements.txt
└── README.md
```
## 6. Dataset Information

Rather than relying on real student records which raise privacy
concerns and are difficult to collect at scale this project uses
a synthetic dataset generated specifically to mirror realistic quiz
behavior. Every record is created using the same rule thresholds
that power the Rule-Based Engine, with a small amount of random
noise (+/-5%) layered on top so the Decision Tree learns soft,
natural boundaries instead of memorizing exact cutoffs.

**File:** `data/student_scores.csv`
**Total Records:** 200 synthetic student quiz entries
**Generator Script:** `python data/generate_dataset.py`

### Column Reference

| Column | Type | Range | Description |
|---|---|---|---|
| `student_id` | int | 1 - 200 | Unique record identifier |
| `topic` | string | 10 topics | Quiz subject area |
| `score_pct` | float | 5.0 - 100.0 | Quiz score percentage |
| `response_time` | float | 10.0 - 150.0 | Time taken, in seconds |
| `confidence` | string | High / Medium / Low | Self reported confidence level |
| `prev_score` | float | 0.0 - 100.0 | Score from the previous quiz attempt |
| `recommendation` | string | 3 classes | Target label used for training |

### Label Distribution (Approximate)

| Label | Count | Share |
|---|---|---|
| Review Basics | ~70 | 35% |
| Practice More | ~80 | 40% |
| Next Topic | ~50 | 25% |

This distribution is intentionally close to balanced, giving the
Decision Tree enough examples of every class to learn from without
favoring one recommendation over another.

### Why Synthetic Data
Using generated data instead of real student records served two
clear purposes:
1. **Privacy**: No student information was ever collected, stored,
   or put at risk. There were no consent concerns to manage because
   no real individual's data existed in the first place.
2. **Control**: Generating the data made it possible to shape the
   class distribution deliberately, keeping the dataset realistic
   and balanced rather than skewed toward whichever outcome happens
   to dominate in a typical classroom.
---
## 7. Tech Stack

Every tool in this project was chosen for a specific reason rather
than out of habit. *Python* keeps the entire codebase in one
language, *Streamlit* removes the need for separate frontend work,
and *scikit learn* provides a dependable, well documented foundation
for the Decision Tree model. Together, this stack made it possible
to build a fully functional, explainable AI application without
introducing unnecessary complexity.

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core programming language for the entire application |
| Streamlit | 1.35.0 | Builds the interactive web interface without separate frontend code |
| scikit-learn | 1.5.0 | Trains and runs the Decision Tree Classifier |
| Pandas | 2.2.2 | Loads, cleans, and preprocesses the dataset |
| NumPy | 1.26.4 | Handles numerical operations and array computations |
| Plotly | 5.22.0 | Renders all interactive charts in the Charts tab |
| Matplotlib | 3.9.0 | Provides supporting static visualizations where needed |

---
## 8. Installation and Setup

Getting the AI Tutor running locally takes only a few minutes.
The steps below walk through cloning the repository, setting up an
isolated Python environment, installing dependencies, generating
the dataset, and launching the application.

### Prerequisites

Before starting, make sure the following are installed on your
system:
- Python 3.10 or higher
- pip (comes bundled with Python)
- Git

### Setup Steps
### Step 1 *Clone the repository*
```bash
git clone https://github.com/YOUR_USERNAME/AI_Tutor_PLR_AbstractMinds.git
cd AI_Tutor_PLR_AbstractMinds
```
### Step 2 *Create and activate a virtual environment*
### *Windows*
```bash
python -m venv venv
venv\Scripts\activate
```
### *Mac/Linux*
```bash
python -m venv venv
source venv/bin/activate
```
### Step 3 *Install all required dependencies*
```bash
pip install -r requirements.txt
```
### Step 4 *Generate the synthetic dataset (run once only)*
```bash
python data/generate_dataset.py
```
### Step 5 *Create the models folder*
```bash
mkdir models
```
### Step 6 *Launch the application*
```bash
streamlit run app.py
```
Once the command runs, the app opens automatically in your default
browser at `http://localhost:8501`. If it does not open on its own,
copy that address into your browser manually.

---
## 9. How to Run and Use

Once the application is running, the interface is organized into
four tabs and a sidebar, each serving a distinct purpose in the
overall workflow from entering a quiz result to fully evaluating
how the AI arrived at its decision.

### Tab 1  *Get Recommendation*

This is where every session starts. Enter the details of a recent
quiz, choose which AI method should analyze them, and receive a
personalized recommendation in return.

1. Select your quiz topic from the dropdown
2. Set your quiz score using the slider (0 to 100)
3. Choose your confidence level (High / Medium / Low)
4. Set your response time and previous score
5. Select an AI mode from the sidebar Rule Based or Decision Tree
6. Click **Get My Recommendation**
7. Review the recommendation, next topic, practice count, and
   revision flag

### Tab 2 *Charts*

A visual breakdown of both the dataset and your own performance,
rendered through four interactive Plotly charts.

- **Bar chart**: average score per topic
- **Pie chart**: distribution of recommendations across the dataset
- **Line chart**: score trend across students
- **Radar chart**: your performance compared to the dataset average

### Tab 3 *Explainability*

This tab answers the question every recommendation system should be
able to answer: *why*. It traces the exact reasoning behind the
result shown in *Tab 1*.

- View which rule or Decision Tree path fired
- See key factors alongside clear signal indicators
- Read the full step-by-step reasoning log
- Check the input-versus-threshold summary table

### Tab 4 *Evaluation*

A direct, side by side comparison of both AI approaches, backed by
standard classification metrics rather than assumptions.

- Compare the Rule-Based Engine against the Decision Tree
- View Accuracy, Precision, Recall, and F1 Score for both
- Inspect the confusion matrix heatmap
- Read the full classification report

### Sidebar Controls

Persistent controls available from any tab, used to switch AI
modes, retrain the model, and review project context without
leaving the current view.

- Switch between the Rule-Based Engine and the Decision Tree
- Click **Retrain Decision Tree** to retrain the model on demand
- View **Project Information** title, subject, supervisor.
- View **Developer Details** team member names, roll numbers,
  roles, and individual contributions

---
