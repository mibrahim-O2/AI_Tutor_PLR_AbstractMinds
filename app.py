
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from recommendation_engine import (
    load_data,
    train_model,
    run_rules,
    run_model,
    generate_explanation,
    get_tree_explanation,
    create_visuals,
    evaluate_model
)
from utils.helpers import (
    get_topic_list,
    validate_inputs,
    decode_recommendation,
    get_recommendation_color
)

# PAGE CONFIGURATION
# it Sets the browser tab title and icon and layout width
st.set_page_config(
    page_title            = "AI Tutor PLR",
    page_icon             = "",
    layout                = "wide",
    initial_sidebar_state = "expanded"
)
#small reusable 'card" widget used across the result panels
def metric_card(label, value, color="#1f77b4"):
    """
    Renders a styled metric card using st.markdown with inline CSS.

    Parameters:
        label (str): Small label text shown above the value.
        value (str): Main value text displayed in bold.
        color (str): Hex color for the left border and text.
    """
    st.markdown(
        f"""
        <div style="
            background-color : {color}18;
            border-left      : 5px solid {color};
            padding          : 14px 18px;
            border-radius    : 8px;
            margin-bottom    : 8px;
        ">
            <p style="margin:0; font-size:13px; color:#888;">
                {label}
            </p>
            <p style="margin:4px 0 0 0; font-size:22px;
                      font-weight:700; color:{color};">
                {value}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Grouped bar chart comparing the Rule-Based Engine against the
# Decision Tree across all four metrics at once.
def comparison_chart(metrics):
    """
    Builds a grouped bar chart comparing Rule-Based Engine vs
    Decision Tree across Accuracy, Precision, Recall, F1.

    Parameters:
        metrics (dict): Output dictionary from evaluate_model().

    Returns:
        plotly.graph_objects.Figure
    """
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]

    rules_vals = [
        metrics["rules_accuracy"],
        metrics["rules_precision"],
        metrics["rules_recall"],
        metrics["rules_f1"]
    ]
    tree_vals = [
        metrics["tree_accuracy"],
        metrics["tree_precision"],
        metrics["tree_recall"],
        metrics["tree_f1"]
    ]

    fig = go.Figure()

    # Rule Based engine bars (orange)
    fig.add_trace(go.Bar(
        name         = "Rule-Based Engine",
        x            = metric_names,
        y            = rules_vals,
        marker_color = "#FFA500",
        text         = [f"{v:.2%}" for v in rules_vals],
        textposition = "outside"
    ))

    # Decision Tree bars are (blue)
    fig.add_trace(go.Bar(
        name         = "Decision Tree",
        x            = metric_names,
        y            = tree_vals,
        marker_color = "#1f77b4",
        text         = [f"{v:.2%}" for v in tree_vals],
        textposition = "outside"
    ))

    fig.update_layout(
        title            = "Rule-Based Engine vs Decision Tree -- Metric Comparison",
        barmode          = "group",
        yaxis_range      = [0, 1.15],
        yaxis_tickformat = ".0%",
        xaxis_title      = "Metric",
        yaxis_title      = "Score",
        plot_bgcolor     = "rgba(0,0,0,0)",
        paper_bgcolor    = "rgba(0,0,0,0)",
        legend           = dict(
            orientation = "h",
            yanchor     = "bottom",
            y           = 1.02,
            xanchor     = "right",
            x           = 1
        )
    )

    return fig

# heatmap of the confusion matrix accuracy alone hides whihch
# topics the model confuses with one another this shows it directly
def confusion_matrix_chart(cm, labels):
    """
    Builds an annotated confusion matrix heatmap using Plotly.

    Rows = are actual labels
    Columns = are predicted labels
    Diagonal = Correct predictions.

    Parameters:
        cm     (np.ndarray): Confusion matrix from evaluate_model().
        labels (list):       Ordered list of class label names.

    Returns:
        plotly.express figure (imshow).
    """
    fig = px.imshow(
        cm,
        labels                 = dict(x="Predicted", y="Actual", color="Count"),
        x                      = labels,
        y                      = labels,
        text_auto              = True,
        color_continuous_scale = "Blues",
        title                  = "Confusion Matrix of Decision Tree Predictions"
    )
    fig.update_layout(
        xaxis_title   = "Predicted Label",
        yaxis_title   = "Actual Label",
        paper_bgcolor = "rgba(0,0,0,0)"
    )
    return fig
# Renders the whole app: header, sidebar, and four tabs
# get Recommendation, cCharts, Explainability and Evaluation.
def render_ui():
    """
    Main UI render function.
    Called at module level so Streamlit can execute it on every rerun.
    """

    # Project title and team info shown at the top of every page.
    st.markdown(
        """
        <div style="padding: 20px 0 10px 0;">
            <h1 style="margin:0;">
                Personalized Learning Recommendations
            </h1>
            <p style="color:#888; margin:6px 0 0 0; font-size:16px;">
                Group:Abstract Minds &nbsp;|&nbsp;
                Rule-Based Engine & Decision Tree Classifier
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.divider()

    # Sidebar::  mode selector and retrain button and project info
    with st.sidebar:
        st.markdown("Settings & Controls")

        # Switches between the Rule-Based Engine and Decision Tree.
        mode = st.radio(
            label   = "Select AI Mode",
            options = ["Rule-Based Engine", "Decision Tree"],
            index   = 0,
            help    = (
                "Rule-Based: uses IF-THEN logic rules.\n"
                "Decision Tree: uses a trained ML model."
            )
        )

        st.divider()
        st.markdown("Model Controls")

        # Retrains the Decision Tree on demand and saves it to
        # models or decision_tree.pkl
        retrain_btn = st.button(
            label               = "Retrain Decision Tree Model",
            use_container_width = True,
            help                = "Retrains the model on the dataset and saves it."
        )

        if retrain_btn:
            with st.spinner("Training Decision Tree... please wait."):
                df_train = load_data()
                if df_train is not None:
                    train_result = train_model(df_train)
                    if train_result["status"] == "success":
                        st.success(
                            f"Model retrained successfully.\n"
                            f"Accuracy: {train_result['accuracy']:.2%}"
                        )
                    else:
                        st.error(f" {train_result['message']}")
                else:
                    st.error(
                        "Dataset not found. "
                        "Run: python data/generate_dataset.py"
                    )

        st.divider()
        st.markdown("### About This Project")
        st.markdown(
            "**Title:** AI Tutor for Personalized Learning Recommendations\n\n"
            "**Subject:** Artificial Intelligence Lab(CSC631)\n\n"
            "**Instructor:** Rajesh Kumar\n\n"
            "**Institution:** IMCS, University of Sindh, Jamshoro\n\n"
            "**Team — AbstractMinds**\n\n"
            "- **Muhammad Ibrahim** `2k23/CSE/94` · Project Lead\n"
            "  - Recommendation Engine · Decision Tree · Evaluation · Documentation . Dataset Generation\n\n"
            "- **Arsal Jan** `2k23/CSE/34` · UI Developer\n"
            "  - Streamlit UI · Tab Layout · Charts Integration · Testing\n\n"
            "- **Ali** `2k23/CSE/27` · Utilities & Docs\n"
            "  - Helper Utilities · Screenshots \n\n"
            "---\n\n"
            "**Stack:** Python · Streamlit · scikit-learn · Plotly · Pandas\n\n"
            "**AI Methods:** Rule-Based Engine + Decision Tree Classifier"
        )

    # Loaded once and shared across all tabs.
    df = load_data()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Get Recommendation",
        "Charts",
        "Explainability",
        "Evaluation"
    ])

    # TAB 1 -- GET RECOMMENDATION
with tab1:

        st.subheader("Enter Your Quiz Details")
        st.caption(
            "Fill in your quiz details below and click "
            "**Get My Recommendation** to receive a personalized study plan."
        )

        # Three columns keep the form compact
        col1, col2, col3 = st.columns(3)

        with col1:
            topic = st.selectbox(
                label   = "Quiz Topic",
                options = get_topic_list(),
                index   = 0,
                help    = "Select the topic you were just tested on."
            )

        with col2:
            score = st.slider(
                label     = "Quiz Score (%)",
                min_value = 0,
                max_value = 100,
                value     = 60,
                step      = 1,
                help      = "Your score on the quiz (0 to 100)."
            )

        with col3:
            confidence = st.radio(
                label   = "Confidence Level",
                options = ["High", "Medium", "Low"],
                index   = 1,
                help    = "How confident did you feel during the quiz?"
            )

        col4, col5 = st.columns(2)

        with col4:
            response_time = st.slider(
                label     = "Response Time (seconds)",
                min_value = 5,
                max_value = 180,
                value     = 60,
                step      = 5,
                help      = "Total time spent answering the quiz."
            )

        with col5:
            prev_score = st.number_input(
                label     = "Previous Score on this Topic (%)",
                min_value = 0.0,
                max_value = 100.0,
                value     = 50.0,
                step      = 0.5,
                help      = "Your score last time you took a quiz on this topic."
            )

        st.divider()

        # Quick summary of the current inputs before submitting
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        col_p1.metric("Topic",         topic)
        col_p2.metric("Your Score",    f"{score}%")
        col_p3.metric("Confidence",    confidence)
        col_p4.metric("Response Time", f"{response_time}s")

        st.divider()

        run_btn = st.button(
            label               = "Get My Recommendation",
            type                = "primary",
            use_container_width = True
        )

        if run_btn:

            # Validate before running either engine
            is_valid, error_msg = validate_inputs(score, response_time, topic)
            if not is_valid:
                st.error(f"{error_msg}")
                st.stop()
# Mode selected in the sidebar decides which engine runs below.
            with st.spinner("Generating recommendation..."):

                # Same input dictionary feeds both engines
                input_data = {
                    "score_pct":           score,
                    "response_time":       response_time,
                    "confidence":          confidence,
                    "prev_score":          prev_score,
                    "topic":               topic,
                    # Extra keys used by radar chart in Tab 2
                    "score_pct_input":     score,
                    "response_time_input": response_time,
                    "prev_score_input":    prev_score
                }

                if mode == "Rule-Based Engine":
                    # Handcrafted IF-THEN logic transparent by design
                    result = run_rules(
                        score         = score,
                        confidence    = confidence,
                        response_time = response_time,
                        topic         = topic,
                        prev_score    = prev_score
                    )
                else:
                    # Trained model Tab 3 reconstructs its reasoning
                    # afterwards instead of just printing a fixed rule
                    result = run_model(input_data)

            # Cached so switching tabs doesn't lose the result or
            # rerun the engine on every Streamlit rerun.
            st.session_state["last_result"]  = result
            st.session_state["last_context"] = input_data
            st.session_state["last_mode"]    = mode

            # Color-coded banner: red = review basics,
            # orange = practice more, green = next topic.
            st.divider()
            rec   = result.get("recommendation", "Practice More")
            color = get_recommendation_color(rec)
            disp  = decode_recommendation(rec)

            st.markdown(
                f"""
                <div style="
                    background  : linear-gradient(135deg, {color}22, {color}08);
                    border-left : 6px solid {color};
                    padding     : 24px 28px;
                    border-radius: 10px;
                    margin-bottom: 16px;
                ">
                    <h2 style="color:{color}; margin:0 0 8px 0;">
                        Final Recommendation: {disp}
                    </h2>
                    <p style="margin:0; color:#666; font-size:15px;">
                        Topic: <strong>{topic}</strong>
                        &nbsp;&middot;&nbsp;
                        Score: <strong>{score}%</strong>
                        &nbsp;&middot;&nbsp;
                        Mode: <strong>{mode}</strong>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Four quick glance cards for the key outputs
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                metric_card("Recommendation", rec, color)

            with c2:
                metric_card(
                    "Suggested Next Topic",
                    result.get("next_topic", topic),
                    "#1f77b4"
                )

            with c3:
                metric_card(
                    "Practice Questions",
                    str(result.get("practice_questions", 5)),
                    "#9467bd"
                )

            with c4:
                # Red if revision needed, green if not
                revision_color = (
                    "#FF4B4B" if result.get("revision_needed")
                    else "#00C853"
                )
                metric_card(
                    "Revision Needed",
                    "Yes" if result.get("revision_needed") else " No",
                    revision_color
                )

            # Full explanation lives in Tab 3; this is the short version
            st.divider()
            st.markdown("Why this recommendation?")
            explanation = generate_explanation(result, input_data)
            st.info(explanation)

            rule_text = result.get("rule_triggered", "")
            if rule_text:
                st.caption(f"{rule_text}")

            st.success(
                "Result saved! Switch to Charts, "
                " Explainability, or Evaluation tabs "
                "for full details."
            )
