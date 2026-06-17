
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
