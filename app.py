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
# TAB 2 -- CHARTS

    with tab2:

        st.subheader("Performance Visualizations")

        if df is None:
            st.error(
                "Dataset not found. "
                "Run: python data/generate_dataset.py"
            )
            st.stop()

        # Summary numbers shown above the charts as context
        st.markdown("#### Dataset Overview")
        ov1, ov2, ov3, ov4 = st.columns(4)
        ov1.metric("Total Records",     len(df))
        ov2.metric("Topics Covered",    df["topic"].nunique())
        ov3.metric("Avg Score",         f"{df['score_pct'].mean():.1f}%")
        ov4.metric("Avg Response Time", f"{df['response_time'].mean():.0f}s")

        st.divider()

        # Builds bar/pie/line/radar figures from the dataset and the
        # last submitted inputs
        result_for_charts = st.session_state.get("last_context", {})
        charts = create_visuals(df, result_for_charts)

        if not charts:
            st.warning(" Charts could not be generated.")
            st.stop()

        # Bar = avg score per topic, Pie = recommendation distribution.
        col_l, col_r = st.columns(2)

        with col_l:
            st.plotly_chart(
                charts["bar_chart"],
                use_container_width=True
            )

        with col_r:
            st.plotly_chart(
                charts["pie_chart"],
                use_container_width=True
            )

        st.divider()

        # Score trend across the first 40 records.
        st.plotly_chart(
            charts["line_chart"],
            use_container_width=True
        )

        st.divider()

        # Compares this students inputs against the dataset averages
        if "last_context" in st.session_state:
            st.plotly_chart(
                charts["radar_chart"],
                use_container_width=True
            )
        else:
            st.info(
                "Run a recommendation in the "
                "Get Recommendation tab to see "
                "your personal radar chart."
            )

        st.divider()

        st.markdown("#### Dataset Sample")
        st.caption("Showing first 20 rows of student_scores.csv")
        st.dataframe(
            df.head(20).copy(),
            use_container_width=True,
            hide_index=True
        )
# TAB 3 -- EXPLAINABILITY

    with tab3:

        st.subheader("How The Explainability Module Works")
        st.caption(
            "This section shows exactly how and why "
            "the AI produced its recommendation."
        )

        # Guard: user must run a recommendation in Tab 1 first
        if "last_result" not in st.session_state:
            st.info(
                "Go to the Get Recommendation tab, "
                "enter your quiz results and click "
                "Get My Recommendation first."
            )
            st.stop()

        result  = st.session_state["last_result"]
        context = st.session_state["last_context"]
        mode    = st.session_state["last_mode"]

        rec   = result.get("recommendation", "Practice More")
        color = get_recommendation_color(rec)

        st.markdown("####  Final Recommendation")
        st.markdown(
            f"""
            <div style="
                background    : {color}18;
                border-left   : 5px solid {color};
                padding       : 16px 20px;
                border-radius : 8px;
            ">
                <h3 style="color:{color}; margin:0;">
                    [RESULT] {decode_recommendation(rec)}
                </h3>
                <p style="margin:6px 0 0 0; color:#555;">
                    AI Mode used: <strong>{mode}</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.divider()

        # Even when the Decision Tree produced the result, we re derive
        # simple, human readable thresholds here so the student always
        # sees WHY a result was given not just a black box prediction
        st.markdown("####  Key Factors in This Decision")

        score         = context.get("score_pct",     0)
        response_time = context.get("response_time", 0)
        confidence    = context.get("confidence",    "Medium")
        topic         = context.get("topic",         "")
        prev_score    = context.get("prev_score",    None)

        kf1, kf2, kf3, kf4 = st.columns(4)

        # Score signal compared against thresholds
        score_signal = (
            "LOW"    if score < 40  else
            "MID"    if score < 70  else
            "HIGH"
        )
        kf1.metric("Quiz Score",    f"{score}%",       score_signal)

        # Response time signa above 80s is slow
        time_signal = (
            "SLOW" if response_time > 80 else "[FAST]"
        )
        kf2.metric("Response Time", f"{response_time}s", time_signal)

        conf_signal = (
            "[LOW CONFIDENCE]"    if confidence == "Low"    else
            "[MID CONFIDENCE]"    if confidence == "Medium" else
            "[HIGH CONFIDENCE]"
        )
        kf3.metric("Confidence", confidence, conf_signal)

        # Score trend signal compares current vs previous
        if prev_score is not None:
            change       = score - prev_score
            trend_label  = f"{'(+)' if change >= 0 else '(-)'} {abs(change):.1f}%"
            trend_signal = "IMPROVING" if change >= 0 else "[DECLINING]"
            kf4.metric("Score Trend", trend_label, trend_signal)
        else:
            trend_signal = "N/A"
            kf4.metric("Score Trend", "N/A", "No previous score")

        st.divider()

        # Plain   English paragraph generated by generate_explanation()
        st.markdown("#### Detailed Explanation")
        explanation = generate_explanation(result, context)
        st.info(explanation)

        st.divider()

        # Shows the exact rule or tree path that fired; differs by mode
        st.markdown("####  AI Decision Path")

        if mode == "Rule-Based Engine":

            st.markdown("**Primary Rule Triggered:**")
            rule = result.get("rule_triggered", "")
            st.code(rule, language=None)

            modifiers = result.get("modifiers_applied", [])
            if modifiers:
                st.markdown("**Modifier Rules Applied:**")
                for mod in modifiers:
                    st.code(mod, language=None)
            else:
                st.success(" No modifier rules were triggered.")

            st.markdown("**Step-by-Step Reasoning Log:**")
            steps = result.get("reasoning_steps", [])
            with st.expander(
                "Show full reasoning log", expanded=False
            ):
                for step in steps:
                    st.text(step)

        else:
            # Decision Tree mode show prediction and split path

            st.markdown("**Decision Tree Prediction:**")
            st.code(
                result.get("rule_triggered", "Decision Tree prediction"),
                language=None
            )

            st.markdown("**Tree Split Path:**")
            tree_exp = get_tree_explanation(context)
            with st.expander("Show tree split path", expanded=True):
                st.code(tree_exp, language=None)

            st.markdown("**Model Reasoning Steps:**")
            steps = result.get("reasoning_steps", [])
            with st.expander("Show reasoning steps", expanded=False):
                for step in steps:
                    st.text(step)

        st.divider()

        # Sidebyside comparison of each input against its threshold
        st.markdown("####  Input vs Threshold Summary")

        thresholds_data = {
            "Feature": [
                "Quiz Score (%)",
                "Response Time (s)",
                "Confidence Level",
                "Score Trend"
            ],
            "Your Value": [
                f"{score}%",
                f"{response_time}s",
                confidence,
                f"{score - prev_score:+.1f}%"
                if prev_score is not None else "N/A"
            ],
            "Threshold": [
                "< 40 / 40-70 / >= 70",
                "<= 80s (fast) / > 80s (slow)",
                "High / Medium / Low",
                "Drop > 10% flags revision"
            ],
            "Signal": [
                score_signal,
                time_signal,
                conf_signal,
                trend_signal if prev_score is not None else "N/A"
            ]
        }

        st.dataframe(
            pd.DataFrame(thresholds_data),
            use_container_width=True,
            hide_index=True
        )
# TAB 4 -- EVALUATION

    with tab4:

        st.subheader("Evaluation Module")
        st.caption(
            "Compares Rule-Based Engine vs Decision Tree across "
            "Accuracy, Precision, Recall, F1-Score, and Confusion Matrix."
        )

        # Guard: dataset must exist
        if df is None:
            st.error(" Dataset not found.")
            st.stop()

        # evaluates both approaches on the same 20% test split
        with st.spinner(
            "PROCESSING the evaluation on test split..."
        ):
            metrics = evaluate_model(df)

        # guard evaluation must succeed
        if not metrics or metrics["test_size"] == 0:
            st.error(
                "Evaluation failed. "
                "Check dataset and model."
            )
            st.stop()

        # how many records were used for training vs testing
        st.markdown("#### Dataset Split (80% Train / 20% Test)")
        ds1, ds2, ds3 = st.columns(3)
        ds1.metric("Total Records", len(df))
        ds2.metric("Training Set",  metrics["train_size"])
        ds3.metric("Test Set",      metrics["test_size"])

        st.divider()

        # rule Based (orange) vs Decision Tree (blue)
        st.markdown("#### Performance Metrics")

        col_rules, col_tree = st.columns(2)

        with col_rules:
            st.markdown(
                "<h4 style='color:#FFA500;'>"
                "[RULES] Rule-Based Engine"
                "</h4>",
                unsafe_allow_html=True
            )
            metric_card("Accuracy",  f"{metrics['rules_accuracy']:.2%}",  "#FFA500")
            metric_card("Precision", f"{metrics['rules_precision']:.2%}", "#FFA500")
            metric_card("Recall",    f"{metrics['rules_recall']:.2%}",    "#FFA500")
            metric_card("F1 Score",  f"{metrics['rules_f1']:.2%}",        "#FFA500")

        with col_tree:
            st.markdown(
                "<h4 style='color:#1f77b4;'>"
                "[TREE] Decision Tree"
                "</h4>",
                unsafe_allow_html=True
            )
            metric_card("Accuracy",  f"{metrics['tree_accuracy']:.2%}",  "#1f77b4")
            metric_card("Precision", f"{metrics['tree_precision']:.2%}", "#1f77b4")
            metric_card("Recall",    f"{metrics['tree_recall']:.2%}",    "#1f77b4")
            metric_card("F1 Score",  f"{metrics['tree_f1']:.2%}",        "#1f77b4")

        st.divider()

        st.markdown("####  Side-by-Side Approach Comparison")
        comp_fig = comparison_chart(metrics)
        st.plotly_chart(comp_fig, use_container_width=True)

        # Winner callout message
        if metrics["tree_accuracy"] > metrics["rules_accuracy"]:
            diff = metrics["tree_accuracy"] - metrics["rules_accuracy"]
            st.success(
                f" Decision Tree outperforms the Rule-Based Engine "
                f"by {diff:.2%} in accuracy on this test set."
            )
        elif metrics["rules_accuracy"] > metrics["tree_accuracy"]:
            diff = metrics["rules_accuracy"] - metrics["tree_accuracy"]
            st.info(
                f" Rule-Based Engine outperforms the Decision Tree "
                f"by {diff:.2%} in accuracy on this test set."
            )
        else:
            st.info(
                "Both approaches achieved equal accuracy "
                "on this test set."
            )

        st.divider()

        # highlights where the Decision Tree gets confused between classes
        st.markdown("#### Confusion Matrix (Decision Tree)")
        st.caption(
            "Rows = Actual labels. "
            "Columns = Predicted labels. "
            "Diagonal cells = Correct predictions."
        )

        if metrics["confusion_matrix"] is not None:
            cm_fig = confusion_matrix_chart(
                metrics["confusion_matrix"],
                metrics["labels"]
            )
            st.plotly_chart(cm_fig, use_container_width=True)
        else:
            st.warning(" Confusion matrix not available.")

        st.divider()

        # per class breakdown, tucked into an expander to keep the page clean
        st.markdown(
            "#### Full Classification Report (Decision Tree)"
        )
        st.caption(
            "Per-class Precision, Recall, F1-Score, and Support."
        )
        with st.expander("Show full report", expanded=False):
            st.code(metrics["report"], language=None)

        st.divider()

        # Plain English meaning of each metric, mainly for the viva panel
        st.markdown("#### How to Interpret These Results")

        st.markdown("""
        | Term | Meaning |
        |---|---|
        | **Accuracy** | Percentage of all predictions that were correct |
        | **Precision** | Of all students predicted as class X, how many actually were X |
        | **Recall** | Of all actual class X students, how many did the model identify |
        | **F1 Score** | Harmonic mean of Precision and Recall -- best single metric |
        | **Confusion Matrix** | Shows where the model confuses one class for another |
        """)

        st.markdown("""
        **Why compare two approaches?**

        The Rule-Based Engine uses hand-crafted IF-THEN logic --
        transparent, explainable, and requires no training data.

        The Decision Tree learns patterns automatically from labeled
        data -- adaptive and data-driven.

        Comparing both satisfies the Lab Guide requirement to evaluate
        *at least two settings or approaches* in the Evaluation Module.
        """)

# Streamlit imports this file as a module, so render_ui() must run
# at module level either way -- not only inside if __name__.

if __name__ == "__main__":
    render_ui()
else:
    render_ui()
