# pages/7_Marketing.py
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import logging
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier # For Prediction
from sklearn.preprocessing import LabelEncoder   # For Prediction

# Import shared utilities
from utils.data_utils import load_all_data
from utils.filter_utils import render_global_filters, apply_filters, recalculate_costs_after_filtering
from utils.ai_utils import (
    get_genai_client, render_ai_explanation, render_chat_interface,
    SUMMARY_ADVICE_PROMPT, VISUAL_EXPLANATION_PROMPT, CHAT_PROMPT_TEMPLATE
)
from config import CSS_FILE, CURRENT_DATE # Import necessary config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Page Setup and Styling ---
st.set_page_config(layout="wide")
try:
    with open(CSS_FILE) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    logger.info("CSS loaded successfully.")
except FileNotFoundError:
    st.warning("CSS file not found. Using default styles.")

st.title("Marketing Dashboard")

# --- AI Client Initialization ---
client = get_genai_client()
if not client:
    st.error("AI Client failed to initialize. AI features will be unavailable.")

# --- Data Loading ---
data_orig = load_all_data()
if data_orig is None:
    st.error("Fatal error loading data. Dashboard cannot be displayed.")
    st.stop()

# Prepare employee rates (standard pattern, needed for cost recalc, minimal impact here)
employee_rates = pd.DataFrame()
if "employees" in data_orig and not data_orig["employees"].empty:
    emp_df_rates = data_orig["employees"].copy()
    emp_df_rates['EmployeeID'] = emp_df_rates['EmployeeID'].astype(str)
    rate_cols = [col for col in ['HourlyRate', 'OvertimeRate'] if col in emp_df_rates.columns]
    if rate_cols:
        employee_rates = emp_df_rates.drop_duplicates(subset=['EmployeeID']).set_index('EmployeeID')[rate_cols]
    else:
        logger.error("HourlyRate or OvertimeRate missing from employees data.")

# --- Global Filters ---
start_date, end_date, city, project_type = render_global_filters(data_orig)

# --- Apply Filters and Recalculate Costs ---
# Note: Filters might have limited effect if marketing campaigns aren't linked to city/project type
filtered_data = apply_filters(data_orig, start_date, end_date, city, project_type)
if not filtered_data:
    st.error("Error applying filters or no data matches filters. Cannot display filtered data.")
    st.stop()

# Recalculate costs (standard step, little effect here unless campaigns linked to projects/expenses)
filtered_data = recalculate_costs_after_filtering(filtered_data, employee_rates)

# --- Extract Filtered DataFrames for this Page ---
marketing_campaigns_final = filtered_data.get('marketing_campaigns', pd.DataFrame())
customers_final = filtered_data.get('customers', pd.DataFrame()) # Keep for context if needed

# --- Calculate Marketing KPIs (Using Filtered Data) ---
total_budget = 0
total_leads = 0
total_new_customers = 0
avg_roi = 0.0
cost_per_lead = 0.0
cost_per_acquisition = 0.0

if not marketing_campaigns_final.empty:
    # Ensure columns are numeric, handle errors/NaNs
    marketing_campaigns_final['Budget'] = pd.to_numeric(marketing_campaigns_final['Budget'], errors='coerce').fillna(0)
    marketing_campaigns_final['LeadsGenerated'] = pd.to_numeric(marketing_campaigns_final['LeadsGenerated'], errors='coerce').fillna(0)
    marketing_campaigns_final['CustomersAcquired'] = pd.to_numeric(marketing_campaigns_final['CustomersAcquired'], errors='coerce').fillna(0)
    marketing_campaigns_final['ROI'] = pd.to_numeric(marketing_campaigns_final['ROI'], errors='coerce') # Keep NaN for mean calc
    marketing_campaigns_final['CostPerLead'] = pd.to_numeric(marketing_campaigns_final['CostPerLead'], errors='coerce') # Keep NaN

    total_budget = marketing_campaigns_final['Budget'].sum()
    total_leads = marketing_campaigns_final['LeadsGenerated'].sum()
    total_new_customers = marketing_campaigns_final['CustomersAcquired'].sum()

    avg_roi = marketing_campaigns_final['ROI'].mean()
    if pd.isna(avg_roi): avg_roi = 0.0 # Default if all ROI are NaN

    # Calculate aggregate CPL and CPA
    if total_leads > 0:
        cost_per_lead = total_budget / total_leads
    if total_new_customers > 0:
        cost_per_acquisition = total_budget / total_new_customers

# --- Render Marketing Sections ---

# Marketing Summary
st.header("📋 Marketing Summary")
st.markdown("Key marketing metrics based on the selected global filters.")
col1, col2, col3 = st.columns(3)
col1.metric("Total Marketing Budget", f"${total_budget:,.0f}")
col1.metric("Total Leads Generated", f"{total_leads:,.0f}")
col2.metric("Total Customers Acquired", f"{total_new_customers:,.0f}")
col2.metric("Avg. Cost Per Lead (CPL)", f"${cost_per_lead:,.2f}" if cost_per_lead > 0 else "N/A")
col3.metric("Avg. ROI", f"{avg_roi:.2f}" if avg_roi != 0 else "N/A") # Display N/A if mean couldn't be calc'd
col3.metric("Avg. Cost Per Acquisition (CPA)", f"${cost_per_acquisition:,.2f}" if cost_per_acquisition > 0 else "N/A")

# AI Marketing Advisor
st.subheader("💡 AI Marketing Advisor")
mktg_summary_context = (
    f"Marketing Snapshot (Filtered Period: {start_date.strftime('%Y-%m-%d') if start_date else 'N/A'} to {end_date.strftime('%Y-%m-%d') if end_date else 'N/A'} - Note: Filters may have limited impact on campaign data):\n"
    f"- Campaigns Analyzed: {len(marketing_campaigns_final)}\n"
    f"- Total Budget: ${total_budget:,.0f}\n"
    f"- Leads Generated: {total_leads:,.0f} (Avg CPL: ${cost_per_lead:.2f})\n"
    f"- Customers Acquired: {total_new_customers:,.0f} (Avg CPA: ${cost_per_acquisition:.2f})\n"
    f"- Average Campaign ROI: {avg_roi:.2f} (based on recorded ROI values)\n"
)
render_ai_explanation(
    "Get AI-Powered Marketing Advice",
    client,
    SUMMARY_ADVICE_PROMPT, # Using the general advice prompt
    mktg_summary_context,
    additional_format_args={'current_date': CURRENT_DATE.strftime('%Y-%m-%d')}
)

st.divider()

# --- Section: Campaign Performance ---
st.header("📊 Campaign Performance Analysis")
st.markdown("Comparing Return on Investment (ROI) across different campaign types based on filtered data.")

if marketing_campaigns_final.empty:
    st.warning("No marketing campaign data available for the selected filters.")
else:
    # Data prep already done for KPIs
    # Handle cases where ROI might be missing or zero for plot meaningfulness
    plot_perf_df = marketing_campaigns_final.dropna(subset=['ROI']) # Only plot campaigns with ROI
    if not plot_perf_df.empty:
        fig_performance = px.bar(
            plot_perf_df,
            x="Type",
            y="ROI",
            color="CustomersAcquired", # Color intensity by customers acquired
            title="Campaign ROI by Type (Color Intensity by Customers Acquired)", # Updated title
            # size="Budget", # <--- REMOVED THIS INVALID ARGUMENT
            hover_data=['Name', 'Budget', 'LeadsGenerated', 'CustomersAcquired', 'CostPerLead'],
            labels={'Type': 'Campaign Type', 'ROI': 'Return on Investment', 'CustomersAcquired': 'Customers Acquired'}
            )
        st.plotly_chart(fig_performance, use_container_width=True)

        # AI Explanation for Campaign Performance
        # Filter context dataframe to match plotted data
        perf_context_df = plot_perf_df.copy()
        perf_context = (
            f"Campaign performance analysis based on {len(perf_context_df)} filtered campaigns with recorded ROI.\n"
            f"- Average ROI across these campaigns: {perf_context_df['ROI'].mean():.2f}\n"
            f"- ROI Range: {perf_context_df['ROI'].min():.2f} to {perf_context_df['ROI'].max():.2f}\n"
            f"- Campaign Types by Avg. ROI: {perf_context_df.groupby('Type')['ROI'].mean().sort_values(ascending=False).to_dict()}\n" # Use filtered data for context
        )
        render_ai_explanation("AI Analysis of Campaign Performance", client, VISUAL_EXPLANATION_PROMPT, perf_context)
    else:
        st.info("No campaigns with recorded ROI found in the filtered data to display.")

st.divider()

# --- Section: Customer Acquisition Trends by Neighborhood ---
st.header("📈 Customer Acquisition Trends by Neighborhood")
st.markdown("Tracking new customers acquired over time, broken down by target neighborhood (based on filtered campaign data).")

# Remove the local date range input - use global filters instead

acquisition_plot_df = pd.DataFrame()
if not marketing_campaigns_final.empty:
    try:
        acq_calc_df = marketing_campaigns_final.copy()
        # Ensure necessary columns exist and have correct types
        acq_calc_df["StartDate"] = pd.to_datetime(acq_calc_df["StartDate"], errors='coerce')
        acq_calc_df["CustomersAcquired"] = pd.to_numeric(acq_calc_df["CustomersAcquired"], errors='coerce').fillna(0)
        acq_calc_df = acq_calc_df.dropna(subset=["StartDate", "TargetNeighborhoods"]) # Ensure date and neighborhoods are present

        if not acq_calc_df.empty:
            # Explode neighborhoods if they are stored as delimited strings
            if isinstance(acq_calc_df["TargetNeighborhoods"].iloc[0], str):
                acq_calc_df["TargetNeighborhoods"] = acq_calc_df["TargetNeighborhoods"].str.split('|')
                acq_calc_df = acq_calc_df.explode("TargetNeighborhoods")
                acq_calc_df["TargetNeighborhoods"] = acq_calc_df["TargetNeighborhoods"].str.strip()

            # Group by month and neighborhood
            acquisition_plot_df = acq_calc_df.groupby([
                pd.Grouper(key="StartDate", freq="ME"),
                "TargetNeighborhoods"
            ])["CustomersAcquired"].sum().reset_index()

            logger.info(f"Calculated acquisition trends for {len(acquisition_plot_df)} month/neighborhood combinations.")
        else:
            logger.info("No valid campaign data with dates and neighborhoods for acquisition trend.")

    except Exception as e:
        logger.error(f"Error processing acquisition trends: {e}", exc_info=True)
        st.error("Could not process acquisition trend data.")
        acquisition_plot_df = pd.DataFrame()

if acquisition_plot_df.empty:
    st.warning("No customer acquisition trend data available for the selected filters.")
else:
    fig_trend = px.line(
        acquisition_plot_df,
        x="StartDate",
        y="CustomersAcquired",
        color="TargetNeighborhoods",
        title="New Customers Acquired by Target Neighborhood (Monthly)",
        markers=True,
        labels={'StartDate': 'Month', 'CustomersAcquired': 'New Customers Acquired'}
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # AI Explanation for Acquisition Trends
    acq_context = (
        f"Customer acquisition trends based on {len(marketing_campaigns_final)} filtered campaigns.\n"
        f"- Data spans from {acquisition_plot_df['StartDate'].min().strftime('%Y-%m')} to {acquisition_plot_df['StartDate'].max().strftime('%Y-%m')}.\n"
        f"- Total customers acquired in period: {acquisition_plot_df['CustomersAcquired'].sum():.0f}\n"
        f"- Top 3 neighborhoods by acquisition: {acquisition_plot_df.groupby('TargetNeighborhoods')['CustomersAcquired'].sum().nlargest(3).to_dict()}\n"
    )
    render_ai_explanation("AI Analysis of Acquisition Trends", client, VISUAL_EXPLANATION_PROMPT, acq_context)

st.divider()

# --- Section: Predict Campaign Success ---
st.header("🔮 Predict Campaign Success (ROI > 1)")
st.markdown("Estimate the likelihood of a new campaign achieving an ROI greater than 1 based on historical data.")

# --- Model Training (using original data, cached resource) ---
@st.cache_resource
def train_success_model():
    """Loads campaign data, preprocesses, and trains the success prediction model."""
    logger.info("Attempting to train campaign success model...")
    data_for_train = load_all_data()
    if data_for_train is None or 'marketing_campaigns' not in data_for_train:
        logger.error("Cannot train success model: Marketing campaign data unavailable.")
        return None, None, None # Model, features, encoder

    campaigns_orig = data_for_train['marketing_campaigns'].copy()
    if campaigns_orig.empty:
        logger.warning("Original marketing campaign data is empty. Cannot train.")
        return None, None, None

    # --- Preprocessing ---
    required_cols = ['ROI', 'Budget', 'LeadsGenerated', 'Type']
    if not all(col in campaigns_orig.columns for col in required_cols):
        logger.error("Cannot train success model: Missing required columns.")
        return None, None, None

    df_ml = campaigns_orig.copy()
    # Convert ROI to numeric, coercing errors. Important for target definition.
    df_ml['ROI_numeric'] = pd.to_numeric(df_ml['ROI'], errors='coerce')
    # Define target based on numeric ROI > 1 (Success = 1, Fail/Unknown = 0)
    df_ml['Success'] = np.where(df_ml['ROI_numeric'] > 1, 1, 0)

    # Ensure features are numeric
    df_ml['Budget'] = pd.to_numeric(df_ml['Budget'], errors='coerce')
    df_ml['LeadsGenerated'] = pd.to_numeric(df_ml['LeadsGenerated'], errors='coerce')

    # Encode Campaign Type
    df_ml['Type'] = df_ml['Type'].astype(str).fillna('Unknown')
    le_campaign = LabelEncoder()
    df_ml['CampaignTypeEncoded'] = le_campaign.fit_transform(df_ml['Type'])

    # Define features and target
    features = ['Budget', 'LeadsGenerated', 'CampaignTypeEncoded']
    target = 'Success'

    # Drop rows with NaN in features or where original ROI was needed but invalid
    df_ml = df_ml.dropna(subset=features + ['ROI_numeric']) # Ensure ROI was valid for target definition

    if df_ml.empty or len(df_ml) < 10:
        logger.warning(f"Insufficient valid data ({len(df_ml)} rows) after preprocessing for success model.")
        return None, None, None

    X = df_ml[features]
    y = df_ml[target]

    if X.isnull().values.any() or y.isnull().values.any():
        logger.error("NaN values detected in final training data (X or y) for success model.")
        return None, None, None

    # --- Training ---
    try:
        # Using Classifier for binary target
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X, y)
        logger.info(f"Campaign success prediction model trained successfully with features: {features}")
        return model, features, le_campaign
    except Exception as e:
        logger.error(f"Error training campaign success model: {e}", exc_info=True)
        return None, None, None

# Train the model (runs once and caches)
success_model, success_features, le_campaign_encoder = train_success_model()

# --- Prediction Form ---
if success_model is None or not success_features or le_campaign_encoder is None:
    st.warning("Campaign success prediction model is unavailable due to training issues.")
else:
    st.markdown("**Enter New Campaign Details:**")
    # Initialize session state
    if 'campaign_success_prediction_result' not in st.session_state:
        st.session_state.campaign_success_prediction_result = None
    if 'campaign_success_prediction_context' not in st.session_state:
        st.session_state.campaign_success_prediction_context = None
    if 'campaign_success_prediction_inputs' not in st.session_state:
        st.session_state.campaign_success_prediction_inputs = {
            'Budget': 10000.0, 'LeadsGenerated': 50,
            'Type': le_campaign_encoder.classes_[0] if len(le_campaign_encoder.classes_) > 0 else 'Unknown'
        }

    with st.form("campaign_success_prediction_form"):
        type_options = sorted(list(le_campaign_encoder.classes_))
        default_type = st.session_state.campaign_success_prediction_inputs.get('Type', type_options[0] if type_options else 'Unknown')
        default_type_index = type_options.index(default_type) if default_type in type_options else 0

        pred_budget = st.number_input("Campaign Budget ($)", min_value=100.0, value=float(st.session_state.campaign_success_prediction_inputs.get('Budget', 10000.0)), step=100.0)
        pred_leads = st.number_input("Expected Leads Generated", min_value=0, value=int(st.session_state.campaign_success_prediction_inputs.get('LeadsGenerated', 50)), step=5)
        pred_type = st.selectbox("Campaign Type", options=type_options, index=default_type_index)

        submitted_success = st.form_submit_button("Predict Success (ROI > 1)")

        if submitted_success:
            logger.info("Campaign success prediction form submitted.")
            # Store inputs
            st.session_state.campaign_success_prediction_inputs = {
                 'Budget': pred_budget, 'LeadsGenerated': pred_leads, 'Type': pred_type
            }

            try:
                encoded_type = le_campaign_encoder.transform([pred_type])[0]
                input_data = pd.DataFrame({
                    "Budget": [pred_budget],
                    "LeadsGenerated": [pred_leads],
                    "CampaignTypeEncoded": [encoded_type]
                })
                input_data = input_data.reindex(columns=success_features, fill_value=0)

                # Predict probability for better insight (optional)
                # prediction_proba = success_model.predict_proba(input_data)[0][1] # Probability of class 1 (Success)
                prediction = success_model.predict(input_data)[0] # Direct prediction (0 or 1)
                prediction_text = "Yes" if prediction == 1 else "No"

                # --- Store results ---
                st.session_state.campaign_success_prediction_result = prediction_text
                # st.session_state.campaign_success_prediction_proba = prediction_proba # Optional
                st.session_state.campaign_success_prediction_context = (
                    "INPUTS USED FOR PREDICTION:\n"
                    f"- Campaign Budget: ${pred_budget:,.2f}\n"
                    f"- Expected Leads: {pred_leads}\n"
                    f"- Campaign Type: {pred_type}\n\n"
                    f"PREDICTED SUCCESS (ROI > 1): {prediction_text}\n"
                    # f"(Predicted Probability of Success: {prediction_proba:.1%})\n" # Optional
                    f"\n(Historical Avg ROI in current filter: {avg_roi:.2f})\n"
                    "Note: Prediction based on historical patterns. Actual ROI depends on execution quality, market conditions, and lead conversion rates."
                )
                logger.info(f"Campaign success prediction successful: {prediction_text}")

            except Exception as e:
                st.error(f"Error during campaign success prediction: {e}")
                logger.error("Campaign success prediction failed.", exc_info=True)
                st.session_state.campaign_success_prediction_result = None
                st.session_state.campaign_success_prediction_context = None
            # Rerun implicitly

    # --- Display results OUTSIDE the form ---
    if st.session_state.campaign_success_prediction_result is not None:
        st.success(f"**Predicted Success (ROI > 1): {st.session_state.campaign_success_prediction_result}**")
        if st.session_state.campaign_success_prediction_context and client:
             render_ai_explanation(
                 "AI Explanation of Campaign Success Prediction",
                 client,
                 VISUAL_EXPLANATION_PROMPT,
                 st.session_state.campaign_success_prediction_context
             )
        elif not client:
             st.warning("AI Client unavailable, cannot generate explanation.")
        if st.button("Clear Prediction Result", key="clear_campaign_pred"):
             st.session_state.campaign_success_prediction_result = None
             st.session_state.campaign_success_prediction_context = None
             st.rerun()


st.divider()

# --- Section: Chat Feature ---
st.header("💬 Ask Questions About Marketing Data") # Updated header

# Prepare context for the chat (reuse mktg_summary_context if available)
if 'mktg_summary_context' not in locals():
     mktg_summary_context = f"Marketing data analysis for {len(marketing_campaigns_final)} filtered campaigns."
     logger.warning("Using fallback chat context as mktg_summary_context was not defined.")

chat_base_context = mktg_summary_context # Use the determined context

# Define filter_details_dict matching the structure
filter_details_dict = {
    'start_date': start_date.strftime('%Y-%m-%d') if start_date else 'N/A',
    'end_date': end_date.strftime('%Y-%m-%d') if end_date else 'N/A',
    'city': city,
    'project_type': project_type # Key name used in filter_utils and expected by template
}

# Call render_chat_interface matching the structure
render_chat_interface(
    client=client,
    chat_prompt_template=CHAT_PROMPT_TEMPLATE,
    base_context=chat_base_context, # Use the page-specific context variable
    filter_details=filter_details_dict,
    page_key="marketing", # Unique key for this page
    placeholder="Ask about campaigns, ROI, leads, or acquisition..." # Updated placeholder
)
# --- End of File ---