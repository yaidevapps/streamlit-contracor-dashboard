# app.py
import streamlit as st
import pandas as pd
import os
import logging
from config import CSS_FILE, REQUIRED_FILES, DATA_DIR
from utils.data_utils import load_all_data # Use the centralized loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Page Setup ---
st.set_page_config(page_title="Hi-Tech Electric Analytics", layout="wide")

# --- Load CSS ---
try:
    with open(CSS_FILE) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    logger.info("CSS loaded successfully.")
except FileNotFoundError:
    logger.warning(f"CSS file not found at {CSS_FILE}. Using default styles.")
    st.warning("CSS file not found. Default styles will be applied.")

# --- Title and Description ---
st.title("Hi-Tech Electric Analytics Dashboard")
st.markdown(
    "Actionable insights for a premium low-voltage contractor serving luxury residences "
    "and design firms in Greater Eastside and Seattle."
)

# --- Data Loading ---
# Load all data once here, potentially accessible via session state if needed across pages
# Or rely on each page loading it via the cached function. Let's keep it simple:
# each page will call load_all_data(), leveraging the cache.
# data = load_all_data() # Optional: Load once here if preferred for global access

# --- Welcome Message & Page Navigation ---
st.markdown(
    """
    Welcome to the analytics dashboard! Use the sidebar to explore various modules, including:
    - **Dashboard**: Executive overview and key performance indicators.
    - **Financials**: Revenue, expenses, and profit margin insights.
    - **Customer**: Customer trends and lifetime value analysis.
    - **Operations**: Technician productivity and inventory management.
    - **Project**: Project pipeline and profitability predictions.
    - **Employee**: Skills matrix and performance metrics.
    - **Marketing**: Campaign performance and customer acquisition.
    - **Warranty & Quality**: Warranty claim trends and quality insights.
    - **Competitive Intelligence**: Bid analysis and pricing strategies.
    """
)

# Divider before raw datasets
st.markdown("---")

# --- Raw Datasets Section ---
st.header("📂 Raw Datasets")
st.markdown("Explore the sample raw datasets used in this dashboard.")

# Load data specifically for this section (or use globally loaded data if implemented)
# Using the cached function ensures it's efficient if called again on other pages.
raw_data_display = load_all_data()

if raw_data_display is None:
    st.error("Failed to load data for display. Cannot show raw datasets.")
else:
    # Define datasets to show (can be selective)
    datasets_to_show = {name.replace('.txt', ''): df for name, df in raw_data_display.items()}

    for dataset_name, df in datasets_to_show.items():
        with st.expander(f"{dataset_name.replace('_', ' ').title()}"):
            if df is not None and not df.empty:
                st.write(f"**{dataset_name.replace('_', ' ').title()}**: {len(df)} records")
                # Use dynamic height or limit rows for very large datasets if needed
                st.dataframe(df, use_container_width=True, height=300)
            elif df is not None and df.empty:
                 st.warning(f"Dataset '{dataset_name}' is empty.")
            else:
                 st.error(f"Dataset '{dataset_name}' failed to load properly.")


# --- Footer ---
st.markdown("---")
st.caption("High-End Contractor Analytics | Data driven insights")