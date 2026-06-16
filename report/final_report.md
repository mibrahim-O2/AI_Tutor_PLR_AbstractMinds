# AI Tutor for Personalized Learning Recommendations (PLR)
## Final Report
## Repository Identity
> This repository follows the naming format required by the Lab Guide:
>
> ```
> ProjectAbbreviation_GroupName
> ```
> ```
> AI_Tutor_PLR_AbstractMinds
> ```

| Component | Full Form | Meaning |
|:---:|:---|:---|
| `AI_Tutor` | Artificial Intelligence Tutor | An intelligent system that acts as a personal tutor |
| `PLR` | Personalized Learning Recommendations | Adaptive content tailored to each student's needs |
| `AbstractMinds` | Our Team Name | The developers behind this project |

---

### Why This Name?

Rather than using a lengthy repository title, we condensed our
project identity into a **precise, meaningful abbreviation** 
where every part represents a **core pillar** of what this system does.
```

A I _ T u t o r _ P  L  R  _  A  b  s  t  r  a  c  t  M  i  n  d  s

│       │         │  │  │     └─────────────────────────────────────┘

│       │         │  │  └─ Recommendations        Team Identity

│       │         │  └─ Learning

│       │         └─ Personalized

└───────┘

AI Tutor
```
## Project Information

| Field | Detail |
|---|---|
| **Project Title** | AI Tutor for Personalized Learning Recommendations |
| **Group Name** | AbstractMinds |
| **Subject** | Artificial Intelligence Lab |
| **Instructor** | Rajesh Kumar |
| **Submission Date** | 15th-July-2026 |
| **Institution** | IMCS, University of Sindh, Jamshoro |

---

## Team_AbstractMinds

| Name | Roll No | Role | LinkedIn |
|---|---|---|---|
| Muhammad Ibrahim | 2k23/CSE/94 | Team Leader · AI Logic · Architecture | [![LinkedIn](https://img.shields.io/badge/LinkedIn-Muhammad%20Ibrahim-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/muhammad-ibrahim-o2/?skipRedirect=true) |
| Arsal Jan | 2k23/CSE/34 | Streamlit UI · Testing · Bug Fixes | [![LinkedIn](https://img.shields.io/badge/LinkedIn-Arsal%20Jan-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/arsal-chandio-b2003a347?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app) |
| Ali | 2k23/CSE/27 | Dataset · Screenshots · Documentation | [![LinkedIn](https://img.shields.io/badge/LinkedIn-Ali-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ali-baloch-515484368?utm_source=share_via&utm_content=profile&utm_medium=member_android) |

---

## 1. Problem Statement

Students often struggle with knowing what to study next after a
quiz. A fixed curriculum ignores individual performance differences.
This project builds an AI Tutor that analyzes a student's quiz score,
response time, and confidence level to generate a personalized
learning recommendation telling the student whether to review
basics, practice more, or advance to the next topic.

### Input

| Input | Type | Range |
|---|---|---|
| Quiz Score | float | 0 — 100 % |
| Response Time | float | 5 — 180 seconds |
| Confidence Level | string | High / Medium / Low |
| Topic | string | 10 predefined topics |
| Previous Score | float | 0 — 100 % |

### Output

| Output | Description |
|---|---|
| Recommendation | Review Basics / Practice More / Next Topic |
| Next Topic | Suggested topic to study next |
| Practice Question Count | Number of practice questions assigned |
| Revision Flag | Whether revision is recommended before next quiz |

---

## 2. PEAS Framework

| PEAS Element | Description |
|---|---|
| **Performance** | Recommendation accuracy, student learning mastery rate |
| **Environment** | Web based educational platform (Streamlit app) |
| **Actuators** | Recommendation output, next topic suggestion, practice plan |
| **Sensors** | Quiz score, response time, confidence level, previous score |

### Environment Classification

**Environment Classification:**
- Partially Observable (cannot directly measure understanding)
- Stochastic (same inputs may yield different outcomes due to context)
- Sequential (past performance informs current recommendation)
- Dynamic (student knowledge changes between sessions)

---

## 3. Methodology

### 3.1 Rule-Based Engine

The Rule-Based Engine applies explicit IF-THEN logic:
#### Primary Rules
## Primary Rules

- **Rule 1:** `score < 40`  
  → Review Basics

- **Rule 2:** `40 <= score < 70`  
  → Practice More

- **Rule 3:** `score >= 70`  
  → Next Topic

## Modifier Rules

- **Modifier A:** `response_time > 80s`  
  → +3 extra practice questions (slow response)

- **Modifier B:** `confidence == "Low"`  
  → +2 extra questions, revision flagged

- **Modifier C:** `score dropped > 10% from previous`  
  → Revision flagged
---

### 3.2 Decision Tree Classifier

A `DecisionTreeClassifier` from scikit-learn was trained on
200 synthetic student quiz records.

#### Configuration

| Parameter | Value | Reason |
|---|---|---|
| `max_depth` | 4 | Prevents overfitting on small dataset |
| `criterion` | gini | Standard Gini impurity for classification |
| `random_state` | 42 | Reproducible results every run |
| `test_size` | 0.20 | 80% train / 20% test split |
| `stratify` | y | Ensures all 3 classes appear in both splits |

#### Features Used for Training
1. `score_pct` — quiz score percentage
2. `response_time` — time taken in seconds
3. `confidence_encoded` — High=2, Medium=1, Low=0
4. `prev_score` — previous quiz score

**Advantages:** Learns patterns from data, interpretable tree
structure, naturally handles non-linear boundaries.

**Disadvantages:** Requires labeled training data, may not
generalize to unseen scenarios without retraining.

---

## 4. Dataset

A synthetic dataset of 200 records was generated using
`data/generate\_dataset.py`. Labels were assigned using the
same thresholds as the Rule Engine, with +/-5% random noise
added to create soft boundaries and prevent perfect overfitting.

### Schema

| Column | Type | Range |
|---|---|---|
| score_pct | float | 5.0 – 100.0 |
| response_time | float | 10.0 – 150.0 |
| confidence | string | High / Medium / Low |
| prev_score | float | 0.0 – 100.0 |
| recommendation | string | 3 classes |

### Class Distribution (Approximate)

| Label | Count | Percentage |
|---|---|---|
| Review Basics | ~70 | 35% |
| Practice More | ~80 | 40% |
| Next Topic | ~50 | 25% |

### Why Synthetic Data?

- No real student data needed (no privacy concerns)
- Full control over label distribution
- Labels assigned by same rules as Rule Engine
- Noise added so Decision Tree learns soft boundaries

---

## 5. AI Integration

Two AI methods are integrated and compared directly
in the Evaluation tab of the app.

Two AI methods are integrated and compared:

| Aspect | Rule-Based Engine | Decision Tree |
|---|---|---|
| Type | Symbolic AI | Machine Learning |
| Training | None required | Supervised learning |
| Interpretability | Very high | High (tree paths) |
| Adaptability | Fixed rules | Learns from data |
| Explainability | Rule triggered shown | Split path shown |

Both methods produce the same output dictionary structure,
allowing direct side-by-side comparison in the Evaluation tab.

---

## 6. Results

Evaluation was performed on a held-out 20% test set (40 records).
Both approaches evaluated on the same split for fair comparison.

| Metric | Rule-Based Engine | Decision Tree |
|---|---|---|
| Accuracy | ~0.80 | ~0.85 |
| Precision | ~0.79 | ~0.85 |
| Recall | ~0.80 | ~0.85 |
| F1 Score | ~0.79 | ~0.84 |

> Note: Exact values vary slightly each run due to dataset noise.
> Run the app and check the Evaluation tab for live results.

**Observation:** The Decision Tree achieves slightly higher
accuracy because it learns soft boundaries from data rather
than relying on fixed hard thresholds. However, the Rule-Based
Engine is fully transparent and requires no training data.

---

## 7. Explainability

For every recommendation the system shows all of the following:

1. **Primary rule or Decision Tree prediction** — what fired
2. **Modifier rules applied** — what adjustments were made
3. **Step-by-step reasoning log** — every decision point
4. **Natural-language explanation** — plain-English paragraph
5. **Key factors table** — score signal, time signal, confidence,
   trend vs threshold

This satisfies the Lab Guide requirement that the system must
help users *"see and understand how the AI reached its result."*

---

## 8. UI Overview

The Streamlit app is organized into four tabs, each mapping
directly to a required Lab Guide module.

| Tab | Lab Guide Module | Key Features |
|---|---|---|
| Get Recommendation | Problem Setup + Core Logic | Input form, validation, result banner, metric cards |
| Charts | Visual UI | Bar, pie, line, radar charts + data table |
| Explainability | Explainability Module | Rule path, reasoning log, factors table |
| Evaluation | Evaluation Module | Metrics, comparison chart, confusion matrix |

---

## 9. Lab Guide Compliance

| Lab Guide Requirement | Our Implementation | Status |
|---|---|---|
| Problem Setup Module | Input form with validation and error messages | Done |
| Core Logic Module | run_rules() + train_model() + run_model() | Done |
| Visual UI Module | 4 Plotly charts + tables + metric cards | Done |
| Explainability Module | Rule path + NL explanation + factors table | Done |
| Evaluation Module | Accuracy/Precision/Recall/F1 + Confusion Matrix | Done |
| Compare two approaches | Rule-Based vs Decision Tree in Tab 4 | Done |
| Modular function structure | load_data/preprocess/run_model/explain/visuals/render_ui | Done |
| No banned frameworks | No Flask/FastAPI/LangChain/paid APIs | Done |
| README.md | Complete with setup and usage instructions | Done |
| requirements.txt | All dependencies pinned | Done |
| Dataset included | data/student_scores.csv (200 rows) | Done |
| Screenshots | screenshots/ folder | Done |

---

## 10. Limitations

| # | Limitation | Impact |
|---|---|---|
| 1 | Synthetic dataset | Results may differ with real student data |
| 2 | Small dataset (200 rows) | Limits model generalization |
| 3 | No user authentication | Single session only |
| 4 | Fixed topic list | Adding topics requires code changes |
| 5 | No session persistence | Results lost on page reload |
| 6 | Hard thresholds (40/70) | May not suit all subjects equally |

---

## 11. Future Improvements

| # | Improvement | Benefit |
|---|---|---|
| 1 | Integrate real student database (SQL/PostgreSQL) | Realistic training data |
| 2 | Add spaced repetition scheduling | Better long-term retention |
| 3 | Replace Decision Tree with Random Forest | Higher accuracy on noisy data |
| 4 | Add multi-student support with login | Real-world usability |
| 5 | NLP-based question generation for practice | Dynamic content |
| 6 | Deploy on Streamlit Cloud | Public access without installation |
| 7 | Add LLM-based explanation panel (optional advanced) | Richer explanations |

---

## 12. Conclusion

This project successfully demonstrates a working AI Tutor
that combines Rule-Based and Machine Learning AI approaches
to generate personalized learning recommendations. The system
satisfies all five modules required by the AI Lab Guide,
uses only the recommended tech stack, and produces explainable,
visual, and evaluable results.

The project was built by Team AbstractMinds as part of the BS Computer Science AI Lab at IMCS,
University of Sindh, Jamshoro under the supervision of **Sir Rajesh Kumar**.

---

## References

1. Russell, S., & Norvig, P. (2022). *Artificial Intelligence:
   A Modern Approach* (4th ed.). Pearson. Chapter 2 — PEAS Framework.
2. Scikit-learn Documentation — DecisionTreeClassifier.
   https://scikit-learn.org/stable/modules/tree.html
3. Streamlit Documentation. https://docs.streamlit.io
4. Plotly Python Documentation. https://plotly.com/python
5. Corbett, A. T., & Anderson, J. R. (1994). Knowledge Tracing:
   Modeling the acquisition of procedural knowledge.
   *User Modeling and User-Adapted Interaction*, 4(4), 253-278.

---

*Report prepared by Team leader of AbstractMinds | Muhammad Ibrahim (2k23/CSE/94) | IMCS, University of Sindh, Jamshoro | 2026*

 