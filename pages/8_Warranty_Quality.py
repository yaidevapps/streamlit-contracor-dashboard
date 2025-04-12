# pages/8_Warranty_Quality.py
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

st.title("Warranty & Quality Dashboard")

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
# Note: Filters might have limited effect if claims aren't directly linked to city
filtered_data = apply_filters(data_orig, start_date, end_date, city, project_type)
if not filtered_data:
    st.error("Error applying filters or no data matches filters. Cannot display filtered data.")
    st.stop()

# Recalculate costs (standard step, minimal effect unless claims linked to projects)
filtered_data = recalculate_costs_after_filtering(filtered_data, employee_rates)

# --- Extract Filtered DataFrames for this Page ---
warranty_claims_final = filtered_data.get('warranty_claims', pd.DataFrame())
projects_final = filtered_data.get('projects', pd.DataFrame())

# --- Calculate Warranty KPIs (Using Filtered Data) ---
claim_rate = 0.0
avg_resolution_time = 0.0
total_cost_to_fix = 0.0
total_customer_cost = 0.0
customer_cost_pct = 0.0
completed_projects_count = len(projects_final[projects_final["Status"] == "Completed"]) if not projects_final.empty else 0

if not warranty_claims_final.empty:
    # Claim Rate (claims / completed projects in filtered period)
    if completed_projects_count > 0:
        claim_rate = (len(warranty_claims_final) / completed_projects_count) * 100

    # Average Resolution Time
    if 'ResolutionTimeDays' in warranty_claims_final.columns:
         warranty_claims_final['ResolutionTimeDays'] = pd.to_numeric(warranty_claims_final['ResolutionTimeDays'], errors='coerce')
         avg_resolution_time = warranty_claims_final['ResolutionTimeDays'].mean()
         if pd.isna(avg_resolution_time): avg_resolution_time = 0.0

    # Costs
    if 'CostToFix' in warranty_claims_final.columns:
        warranty_claims_final['CostToFix'] = pd.to_numeric(warranty_claims_final['CostToFix'], errors='coerce').fillna(0)
        total_cost_to_fix = warranty_claims_final['CostToFix'].sum()

    if 'CustomerCost' in warranty_claims_final.columns:
        warranty_claims_final['CustomerCost'] = pd.to_numeric(warranty_claims_final['CustomerCost'], errors='coerce').fillna(0)
        total_customer_cost = warranty_claims_final['CustomerCost'].sum()

    # Customer Cost Percentage
    if total_cost_to_fix > 0: # Avoid division by zero
        customer_cost_pct = (total_customer_cost / total_cost_to_fix) * 100


# --- Render Warranty & Quality Sections ---

# Warranty Summary
st.header("📋 Warranty & Quality Summary")
st.markdown("Key warranty metrics based on the selected global filters.")
col1, col2, col3 = st.columns(3)
col1.metric("Warranty Claims (Filtered)", len(warranty_claims_final))
col1.metric("Claim Rate vs Completed Projects", f"{claim_rate:.1f}%")
col2.metric("Avg. Resolution Time (Days)", f"{avg_resolution_time:.1f}")
col2.metric("Total Cost to Fix", f"${total_cost_to_fix:,.0f}")
col3.metric("Total Customer Cost", f"${total_customer_cost:,.0f}")
col3.metric("Customer Cost Percentage", f"{customer_cost_pct:.1f}%")


# AI Warranty Advisor
st.subheader("💡 AI Quality Advisor")
wq_summary_context = (
    f"Warranty & Quality Snapshot (Filtered Period: {start_date.strftime('%Y-%m-%d') if start_date else 'N/A'} to {end_date.strftime('%Y-%m-%d') if end_date else 'N/A'}, City: {city}, Type: {project_type}):\n"
    f"- Warranty Claims Analyzed: {len(warranty_claims_final)}\n"
    f"- Claim Rate vs Completed Projects (Filtered): {claim_rate:.1f}%\n"
    f"- Average Resolution Time: {avg_resolution_time:.1f} days\n"
    f"- Total Cost to Fix (Filtered): ${total_cost_to_fix:,.0f}\n"
    f"- Customer Cost as % of Total Cost: {customer_cost_pct:.1f}%\n"
)
render_ai_explanation(
    "Get AI-Powered Warranty & Quality Advice",
    client,
    SUMMARY_ADVICE_PROMPT, # Using the general advice prompt
    wq_summary_context,
    additional_format_args={'current_date': CURRENT_DATE.strftime('%Y-%m-%d')}
)

st.divider()

# --- Section: Claims by Issue Type ---
st.header("📊 Claims Analysis by Issue Type")
st.markdown("Distribution and cost associated with different warranty issue types based on filtered claims.")

if warranty_claims_final.empty:
    st.warning("No warranty claim data available for the selected filters.")
else:
    try:
        # Group by Issue Type
        claims_by_type = warranty_claims_final.groupby('IssueType').agg(
            ClaimCount=('ClaimID', 'count'),
            TotalCostToFix=('CostToFix', 'sum'),
            AvgCostToFix=('CostToFix', 'mean'),
            AvgResolutionTime=('ResolutionTimeDays', 'mean')
        ).reset_index().sort_values('ClaimCount', ascending=False)

        if not claims_by_type.empty:
            # Bar chart for counts
            fig_counts = px.bar(
                claims_by_type,
                x='IssueType',
                y='ClaimCount',
                title='Warranty Claim Count by Issue Type (Filtered)',
                labels={'IssueType': 'Issue Type', 'ClaimCount': 'Number of Claims'}
            )
            # Bar chart for costs
            fig_costs = px.bar(
                claims_by_type.sort_values('TotalCostToFix', ascending=False),
                x='IssueType',
                y='TotalCostToFix',
                title='Total Warranty Cost by Issue Type (Filtered)',
                labels={'IssueType': 'Issue Type', 'TotalCostToFix': 'Total Cost to Fix ($)'}
            )

            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                 st.plotly_chart(fig_counts, use_container_width=True)
            with col_chart2:
                 st.plotly_chart(fig_costs, use_container_width=True)

            # AI Explanation for Issue Types
            top_issue_by_count = claims_by_type.iloc[0]['IssueType'] if not claims_by_type.empty else 'N/A'
            top_issue_by_cost = claims_by_type.sort_values('TotalCostToFix', ascending=False).iloc[0]['IssueType'] if not claims_by_type.empty else 'N/A'
            claims_context = (
                f"Analysis of {len(warranty_claims_final)} filtered warranty claims by issue type.\n"
                f"- Total unique issue types: {warranty_claims_final['IssueType'].nunique()}\n"
                f"- Most frequent issue type: '{top_issue_by_count}' ({claims_by_type.iloc[0]['ClaimCount']} claims).\n"
                f"- Most costly issue type: '{top_issue_by_cost}' (Total Cost: ${claims_by_type.sort_values('TotalCostToFix', ascending=False).iloc[0]['TotalCostToFix']:,.0f}).\n"
            )
            render_ai_explanation("AI Analysis of Claims by Issue Type", client, VISUAL_EXPLANATION_PROMPT, claims_context)
        else:
            st.info("No claims data to group by issue type for the current filters.")

    except Exception as e:
        logger.error(f"Error generating claims by issue type visuals: {e}", exc_info=True)
        st.error("Could not generate claims by issue type visuals.")

st.divider()

# --- Section: Resolution Trends ---
st.header("📈 Warranty Cost & Resolution Time Trends")
st.markdown("Tracking total warranty costs and average resolution times per month based on filtered claims.")

trend_df = pd.DataFrame()
if not warranty_claims_final.empty and 'ClaimDate' in warranty_claims_final.columns:
    try:
        trend_calc_df = warranty_claims_final.copy()
        trend_calc_df['ClaimDate'] = pd.to_datetime(trend_calc_df['ClaimDate'], errors='coerce')
        trend_calc_df = trend_calc_df.dropna(subset=['ClaimDate'])

        if not trend_calc_df.empty:
            trend_df = trend_calc_df.set_index('ClaimDate').resample('ME').agg(
                MonthlyCostToFix=('CostToFix', 'sum'),
                AvgMonthlyResolution=('ResolutionTimeDays', 'mean')
            ).reset_index()
            # Fill potential missing months if needed for smoother line chart
            # full_date_range = pd.date_range(start=trend_df['ClaimDate'].min(), end=trend_df['ClaimDate'].max(), freq='ME')
            # trend_df = trend_df.set_index('ClaimDate').reindex(full_date_range).fillna(0).reset_index().rename(columns={'index':'ClaimDate'})

            logger.info(f"Calculated warranty trends over {len(trend_df)} months.")
        else:
            logger.info("No valid claim data with dates for trend calculation.")

    except Exception as e:
        logger.error(f"Error calculating warranty trends: {e}", exc_info=True)
        st.error("Could not calculate warranty trends.")
        trend_df = pd.DataFrame()

if trend_df.empty:
    st.warning("No warranty trend data available for the selected filters.")
else:
    fig_trend_cost = px.line(
        trend_df, x="ClaimDate", y="MonthlyCostToFix",
        title="Total Warranty Cost to Fix Over Time (Monthly)", markers=True,
        labels={'ClaimDate': 'Month', 'MonthlyCostToFix': 'Total Cost to Fix ($)'}
    )
    fig_trend_time = px.line(
        trend_df, x="ClaimDate", y="AvgMonthlyResolution",
        title="Average Warranty Resolution Time Over Time (Monthly)", markers=True,
        labels={'ClaimDate': 'Month', 'AvgMonthlyResolution': 'Avg. Resolution Time (Days)'}
        )

    st.plotly_chart(fig_trend_cost, use_container_width=True)
    st.plotly_chart(fig_trend_time, use_container_width=True)

    # AI Explanation for Trends
    trend_context = (
        f"Warranty trends based on {len(warranty_claims_final)} filtered claims.\n"
        f"- Data spans from {trend_df['ClaimDate'].min().strftime('%Y-%m')} to {trend_df['ClaimDate'].max().strftime('%Y-%m')}.\n"
        f"- Total Cost to Fix in period: ${trend_df['MonthlyCostToFix'].sum():,.0f}.\n"
        f"- Average Monthly Resolution Time: {trend_df['AvgMonthlyResolution'].mean():.1f} days (if available).\n"
    )
    render_ai_explanation("AI Analysis of Warranty Trends", client, VISUAL_EXPLANATION_PROMPT, trend_context)

st.divider()

# --- Section: Predict Claim Likelihood ---
st.header("🔮 Predict Claim Likelihood")
st.markdown("Estimate the likelihood of a warranty claim for a completed project based on historical data.")

# --- Model Training (using original data, cached resource) ---
@st.cache_resource
def train_claim_model():
    """Loads data, preprocesses completed projects, and trains the claim likelihood model."""
    logger.info("Attempting to train warranty claim likelihood model...")
    data_for_train = load_all_data()
    if data_for_train is None or 'projects' not in data_for_train or 'warranty_claims' not in data_for_train:
        logger.error("Cannot train claim model: Project or warranty data unavailable.")
        return None, None, None # Model, features, encoder

    projects_orig = data_for_train['projects'].copy()
    warranty_claims_orig = data_for_train['warranty_claims'].copy()

    if projects_orig.empty:
        logger.warning("Original project data is empty. Cannot train claim model.")
        return None, None, None

    # --- Preprocessing ---
    # Use only completed projects
    df_ml = projects_orig[projects_orig["Status"] == "Completed"].copy()

    # Create target variable: 1 if claim exists, 0 otherwise
    claimed_project_ids = set(warranty_claims_orig['ProjectID'].astype(str).unique())
    df_ml['ProjectID'] = df_ml['ProjectID'].astype(str)
    df_ml['HasClaim'] = df_ml['ProjectID'].isin(claimed_project_ids).astype(int)

    # Feature Engineering: Project Age at time of analysis (or potential claim)
    df_ml['CompletionDateStr'] = pd.to_datetime(df_ml['CompletionDateStr'], errors='coerce')
    # Use CURRENT_DATE for age calculation consistency
    df_ml['ProjectAgeDays'] = (CURRENT_DATE - df_ml['CompletionDateStr']).dt.days
    # Handle potential NaNs or future completion dates if data is inconsistent
    df_ml['ProjectAgeDays'] = df_ml['ProjectAgeDays'].fillna(0).clip(lower=0)

    # Add other relevant features (ensure they exist and are numeric/encoded)
    numeric_features_claim = ['FinalAmount', 'ActualHours', 'SquareFootage', 'CostOverrunAmount'] # Example features
    for col in numeric_features_claim:
         if col in df_ml.columns:
              df_ml[col] = pd.to_numeric(df_ml[col], errors='coerce').fillna(0)
         else:
              logger.warning(f"Claim predictor training: Feature '{col}' missing. Assigning 0.")
              df_ml[col] = 0.0

    # Encode categorical features
    encoders_claim = {}
    cat_features_claim = ['ProjectType', 'ConstructionType']
    final_features_claim = ['ProjectAgeDays'] + numeric_features_claim # Start with numeric

    for col in cat_features_claim:
        if col in df_ml.columns:
            df_ml[col] = df_ml[col].astype(str).fillna('Unknown')
            if df_ml[col].nunique() > 1:
                le = LabelEncoder()
                try:
                    df_ml[f'{col}Encoded'] = le.fit_transform(df_ml[col])
                    final_features_claim.append(f'{col}Encoded')
                    encoders_claim[col] = le
                except Exception as e_enc:
                    logger.error(f"Error encoding feature '{col}' for claim model: {e_enc}")
            else:
                logger.warning(f"Skipping encoding for '{col}' (claim model) (<= 1 unique value).")
        else:
             logger.warning(f"Categorical feature '{col}' not found for claim model.")

    target_claim = 'HasClaim'

    # Drop rows where target or features are NaN (target should be defined, features handled above)
    df_ml = df_ml.dropna(subset=[target_claim])
    df_ml = df_ml.dropna(subset=final_features_claim)

    if df_ml.empty or len(df_ml) < 20 or df_ml[target_claim].nunique() < 2: # Need sufficient data and both classes
        logger.warning(f"Insufficient valid data ({len(df_ml)} rows) or classes ({df_ml[target_claim].nunique()}) for claim model.")
        return None, None, None

    X = df_ml[final_features_claim]
    y = df_ml[target_claim]

    if X.isnull().values.any() or y.isnull().values.any():
        logger.error("NaN values detected in final training data (X or y) for claim model.")
        return None, None, None

    # --- Training ---
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced') # Classifier for binary target
        model.fit(X, y)
        logger.info(f"Warranty claim likelihood model trained successfully with features: {final_features_claim}")
        return model, final_features_claim, encoders_claim
    except Exception as e:
        logger.error(f"Error training claim likelihood model: {e}", exc_info=True)
        return None, None, None

# Train the model (runs once and caches)
claim_model, claim_features, claim_encoders = train_claim_model()

# --- Prediction Form ---
if claim_model is None or not claim_features or claim_encoders is None:
    st.warning("Warranty claim likelihood prediction model is unavailable due to training issues.")
else:
    st.markdown("**Enter Completed Project Details:**")
    # Initialize session state
    if 'warranty_claim_prediction_result' not in st.session_state:
        st.session_state.warranty_claim_prediction_result = None
    if 'warranty_claim_prediction_context' not in st.session_state:
        st.session_state.warranty_claim_prediction_context = None
    if 'warranty_claim_prediction_inputs' not in st.session_state:
        st.session_state.warranty_claim_prediction_inputs = {
             'ProjectAgeDays': 365, 'FinalAmount': 100000, 'ActualHours': 150, 'SquareFootage': 5000, 'CostOverrunAmount': 0,
             **{cat: enc.classes_[0] if len(enc.classes_)>0 else 'Unknown' for cat, enc in claim_encoders.items()}
        }

    with st.form("warranty_claim_prediction_form"):
        # --- Feature Inputs (match training features) ---
        pred_age_days = st.number_input("Project Age (Days since completion)", min_value=0, value=int(st.session_state.warranty_claim_prediction_inputs.get('ProjectAgeDays', 365)), step=30)
        pred_final_amount = st.number_input("Project Final Amount ($)", min_value=0.0, value=float(st.session_state.warranty_claim_prediction_inputs.get('FinalAmount', 100000.0)), step=5000.0)
        pred_actual_hours = st.number_input("Project Actual Hours", min_value=0, value=int(st.session_state.warranty_claim_prediction_inputs.get('ActualHours', 150)), step=10)
        pred_sqft = st.number_input("Project Square Footage", min_value=0, value=int(st.session_state.warranty_claim_prediction_inputs.get('SquareFootage', 5000)), step=100)
        pred_cost_overrun = st.number_input("Project Cost Overrun ($)", value=float(st.session_state.warranty_claim_prediction_inputs.get('CostOverrunAmount', 0.0)), step=100.0) # Allow negative

        pred_inputs_cat_claim = {}
        for col, encoder in claim_encoders.items():
            encoded_col_name = f'{col}Encoded'
            if encoded_col_name in claim_features:
                options = sorted(list(encoder.classes_))
                default_cat = st.session_state.warranty_claim_prediction_inputs.get(col, options[0] if options else 'Unknown')
                default_idx = options.index(default_cat) if default_cat in options else 0
                selectbox_key = f"pred_claim_{col.lower().replace(' ', '_')}"
                pred_inputs_cat_claim[col] = st.selectbox(f"{col.replace('Type', ' Type')}", options=options, key=selectbox_key, index=default_idx)

        submitted_claim = st.form_submit_button("Predict Claim Likelihood")

        if submitted_claim:
            logger.info("Warranty claim prediction form submitted.")
            # Store inputs
            current_inputs_claim = {
                'ProjectAgeDays': pred_age_days, 'FinalAmount': pred_final_amount, 'ActualHours': pred_actual_hours,
                'SquareFootage': pred_sqft, 'CostOverrunAmount': pred_cost_overrun,
                **pred_inputs_cat_claim
            }
            st.session_state.warranty_claim_prediction_inputs = current_inputs_claim

            try:
                input_dict_claim_df = {
                     'ProjectAgeDays': pred_age_days, 'FinalAmount': pred_final_amount, 'ActualHours': pred_actual_hours,
                     'SquareFootage': pred_sqft, 'CostOverrunAmount': pred_cost_overrun
                }
                # Encode categoricals
                for col, selected_val in pred_inputs_cat_claim.items():
                    encoded_col_name = f'{col}Encoded'
                    if encoded_col_name in claim_features:
                        encoder = claim_encoders.get(col)
                        if selected_val not in encoder.classes_:
                            selected_val_for_transform = 'Unknown' if 'Unknown' in encoder.classes_ else None
                            if selected_val_for_transform is None: raise ValueError(f"Unseen value '{selected_val}' for {col}")
                            st.warning(f"Value '{selected_val}' for {col} not seen during training. Using 'Unknown'.")
                        else:
                            selected_val_for_transform = selected_val
                        input_dict_claim_df[encoded_col_name] = encoder.transform([selected_val_for_transform])[0]

                # Create DataFrame and reindex
                input_df_claim = pd.DataFrame([input_dict_claim_df])
                input_df_claim = input_df_claim.reindex(columns=claim_features, fill_value=0)

                # Predict Probability of Claim (class 1)
                prediction_proba = claim_model.predict_proba(input_df_claim)[0][1]

                # --- Store results ---
                st.session_state.warranty_claim_prediction_result = prediction_proba
                st.session_state.warranty_claim_prediction_context = (
                    "INPUTS USED FOR PREDICTION:\n"
                    f"- Project Age (Days): {pred_age_days}\n"
                    f"- Final Amount: ${pred_final_amount:,.2f}\n"
                    f"- Actual Hours: {pred_actual_hours}\n"
                    f"- Square Footage: {pred_sqft}\n"
                    f"- Cost Overrun: ${pred_cost_overrun:,.2f}\n"
                    + "".join([f"- {col.replace('Type', ' Type')}: {val}\n" for col, val in pred_inputs_cat_claim.items()]) +
                    f"\nPREDICTED CLAIM LIKELIHOOD: {prediction_proba:.1%}\n\n"
                    f"(Historical Claim Rate in current filter: {claim_rate:.1f}%)\n"
                    "Note: Prediction based on historical patterns. Installation quality, specific components used, and customer usage significantly impact actual claims."
                )
                logger.info(f"Warranty claim prediction successful: {prediction_proba:.1%}")

            except ValueError as ve:
                 st.error(f"Error processing prediction input: {ve}")
                 logger.error(f"Warranty claim prediction failed due to input error: {ve}", exc_info=True)
                 st.session_state.warranty_claim_prediction_result = None
                 st.session_state.warranty_claim_prediction_context = None
            except Exception as e:
                st.error(f"Error during claim likelihood prediction: {e}")
                logger.error("Claim likelihood prediction failed.", exc_info=True)
                st.session_state.warranty_claim_prediction_result = None
                st.session_state.warranty_claim_prediction_context = None
            # Rerun implicitly

    # --- Display results OUTSIDE the form ---
    if st.session_state.warranty_claim_prediction_result is not None:
        st.success(f"**Predicted Claim Likelihood: {st.session_state.warranty_claim_prediction_result:.1%}**")
        if st.session_state.warranty_claim_prediction_context and client:
             render_ai_explanation(
                 "AI Explanation of Claim Likelihood",
                 client,
                 VISUAL_EXPLANATION_PROMPT,
                 st.session_state.warranty_claim_prediction_context
             )
        elif not client:
             st.warning("AI Client unavailable, cannot generate explanation.")
        if st.button("Clear Prediction Result", key="clear_claim_pred"):
             st.session_state.warranty_claim_prediction_result = None
             st.session_state.warranty_claim_prediction_context = None
             st.rerun()

st.divider()

# --- Section: Chat Feature ---
st.header("💬 Ask Questions About Warranty & Quality") # Updated header

# Prepare context for the chat
if 'wq_summary_context' not in locals():
     wq_summary_context = f"Warranty & Quality analysis for {len(warranty_claims_final)} filtered claims."
     logger.warning("Using fallback chat context as wq_summary_context was not defined.")

chat_base_context = wq_summary_context

# Define filter_details_dict matching the structure
filter_details_dict = {
    'start_date': start_date.strftime('%Y-%m-%d') if start_date else 'N/A',
    'end_date': end_date.strftime('%Y-%m-%d') if end_date else 'N/A',
    'city': city,
    'project_type': project_type
}

# Call render_chat_interface matching the structure
render_chat_interface(
    client=client,
    chat_prompt_template=CHAT_PROMPT_TEMPLATE,
    base_context=chat_base_context,
    filter_details=filter_details_dict,
    page_key="warranty_quality", # Unique key for this page
    placeholder="Ask about claim rates, costs, issue types, or resolutions..." # Updated placeholder
)