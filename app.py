
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
