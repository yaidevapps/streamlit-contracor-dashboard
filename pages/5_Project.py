# pages/5_Project.py
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import logging
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

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

st.title("Project Dashboard")

# --- AI Client Initialization ---
client = get_genai_client()
if not client:
    st.error("AI Client failed to initialize. AI features will be unavailable.")

# --- Data Loading ---
data_orig = load_all_data()
if data_orig is None:
    st.error("Fatal error loading data. Dashboard cannot be displayed.")
    st.stop()

# Prepare employee rates (standard pattern, needed for cost recalc, though less direct impact here)
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
filtered_data = apply_filters(data_orig, start_date, end_date, city, project_type)
if not filtered_data:
    st.error("Error applying filters or no data matches filters. Cannot display filtered data.")
    st.stop()

# Recalculate costs based on filtered time entries (standard step, recalculates project costs/margins)
filtered_data = recalculate_costs_after_filtering(filtered_data, employee_rates)

# --- Extract Filtered DataFrames for this Page ---
projects_final = filtered_data.get('projects', pd.DataFrame())
bids_final = filtered_data.get('bids', pd.DataFrame())
timelines_final = filtered_data.get('project_timelines', pd.DataFrame())
change_orders_final = filtered_data.get('change_orders', pd.DataFrame())
warranty_claims_final = filtered_data.get('warranty_claims', pd.DataFrame())

# --- Calculate Project KPIs (Using Filtered & Recalculated Data) ---
avg_profit = 0.0
bid_win_rate = 0.0
avg_variance = 0.0
completed_projects_count = 0

# Average Profit Margin (on completed projects in filtered data)
completed_projects_filtered = projects_final[projects_final["Status"] == "Completed"] if not projects_final.empty else pd.DataFrame()
completed_projects_count = len(completed_projects_filtered)
if not completed_projects_filtered.empty and 'ProfitMargin' in completed_projects_filtered.columns:
    # ProfitMargin should be numeric after recalculate_costs_after_filtering
    avg_profit = completed_projects_filtered['ProfitMargin'].mean()
    if pd.isna(avg_profit): avg_profit = 0.0

# Bid Win Rate (using filtered bids)
if not bids_final.empty and 'Status' in bids_final.columns:
    won_bids = (bids_final["Status"] == "Won").sum()
    lost_bids = (bids_final["Status"] == "Lost").sum()
    total_submitted = won_bids + lost_bids
    if total_submitted > 0:
        bid_win_rate = (won_bids / total_submitted) * 100

# Average Timeline Variance (using filtered timelines)
if not timelines_final.empty and 'EstimatedDate' in timelines_final.columns and 'ActualDate' in timelines_final.columns:
    # Ensure dates are datetime (should be from data_utils)
    timelines_calc = timelines_final.copy()
    timelines_calc["EstimatedDate"] = pd.to_datetime(timelines_calc["EstimatedDate"], errors='coerce')
    timelines_calc["ActualDate"] = pd.to_datetime(timelines_calc["ActualDate"], errors='coerce')
    timelines_calc = timelines_calc.dropna(subset=["EstimatedDate", "ActualDate"]) # Only use entries with both dates
    if not timelines_calc.empty:
        timelines_calc["Variance"] = (timelines_calc["ActualDate"] - timelines_calc["EstimatedDate"]).dt.days
        avg_variance = timelines_calc["Variance"].mean()
        if pd.isna(avg_variance): avg_variance = 0.0

# --- Render Project Sections ---

# Project Summary
st.header("📋 Project Summary")
st.markdown("Key project metrics based on the selected global filters.")
col1, col2, col3 = st.columns(3)
col1.metric("Filtered Projects", len(projects_final))
col1.metric("Completed (Filtered)", completed_projects_count)
col2.metric("Avg. Profit Margin (Completed)", f"{avg_profit:.2%}")
col2.metric("Avg. Timeline Variance (Days)", f"{avg_variance:.1f}")
col3.metric("Filtered Bids", len(bids_final))
col3.metric("Bid Win Rate (Filtered)", f"{bid_win_rate:.1f}%")

# AI Project Advisor
st.subheader("💡 AI Project Advisor")
proj_summary_context = (
    f"Project Data Snapshot (Filtered Period: {start_date.strftime('%Y-%m-%d') if start_date else 'N/A'} to {end_date.strftime('%Y-%m-%d') if end_date else 'N/A'}, City: {city}, Type: {project_type}):\n"
    f"- Projects Analyzed: {len(projects_final)} (Completed: {completed_projects_count})\n"
    f"- Average Profit Margin (Completed Projects): {avg_profit:.1%}\n"
    f"- Bids Analyzed: {len(bids_final)}. Bid Win Rate: {bid_win_rate:.1f}%\n"
    f"- Average Timeline Variance (Completed Milestones): {avg_variance:.1f} days\n"
    f"- Change Orders (Filtered): {len(change_orders_final)}\n"
    f"- Warranty Claims (Filtered): {len(warranty_claims_final)}\n"
    f"Note: Profit Margins recalculated based on filtered time entries and expenses."
)
render_ai_explanation(
    "Get AI-Powered Project Advice",
    client,
    SUMMARY_ADVICE_PROMPT, # Using the general advice prompt
    proj_summary_context,
    additional_format_args={'current_date': CURRENT_DATE.strftime('%Y-%m-%d')}
)

st.divider()

# --- Section: Project Pipeline ---
st.header("📊 Project Pipeline Trends")
st.markdown("Tracking project starts and status distribution over time based on filtered data.")

pipeline_df = pd.DataFrame()
if not projects_final.empty and 'StartDateStr' in projects_final.columns:
    try:
        pipeline_calc_df = projects_final.copy()
        # Ensure StartDateStr is datetime (should be from data_utils)
        pipeline_calc_df["StartDateStr"] = pd.to_datetime(pipeline_calc_df["StartDateStr"], errors='coerce')
        pipeline_calc_df = pipeline_calc_df.dropna(subset=["StartDateStr", "Status"])

        if not pipeline_calc_df.empty:
            pipeline_df = pipeline_calc_df.groupby([
                pd.Grouper(key="StartDateStr", freq="ME"), # Group by month end
                "Status"
            ]).size().reset_index(name="Count")
            logger.info(f"Calculated pipeline trends for {len(pipeline_df)} month/status combinations.")
        else:
            logger.info("No valid project data with dates/status for pipeline trend.")

    except Exception as e:
        logger.error(f"Error generating project pipeline data: {e}", exc_info=True)
        st.error("Could not process project pipeline data.")
        pipeline_df = pd.DataFrame()


if pipeline_df.empty:
    st.warning("No project pipeline data available for the selected filters.")
else:
    fig_pipeline = px.line(
        pipeline_df,
        x="StartDateStr",
        y="Count",
        color="Status",
        title="Project Pipeline Trends by Month (Filtered)",
        markers=True,
        labels={'StartDateStr': 'Month', 'Count': 'Number of Projects'}
    )
    st.plotly_chart(fig_pipeline, use_container_width=True)

    # AI Explanation for Pipeline
    pipeline_context = (
        f"Project pipeline trends based on {len(projects_final)} filtered projects.\n"
        f"- Data spans from {pipeline_df['StartDateStr'].min().strftime('%Y-%m')} to {pipeline_df['StartDateStr'].max().strftime('%Y-%m')}.\n"
        f"- Status distribution summary (total counts in period): {pipeline_df.groupby('Status')['Count'].sum().to_dict()}\n"
    )
    render_ai_explanation("AI Analysis of Project Pipeline", client, VISUAL_EXPLANATION_PROMPT, pipeline_context)

st.divider()

# --- Section: Bid Analysis ---
st.header("📈 Bid Analysis")
st.markdown("Funnel visualization and lost bid reason analysis based on filtered bids.")

if bids_final.empty:
    st.warning("No bid data available for the selected filters.")
else:
    try:
        # Ensure Status column exists
        if 'Status' not in bids_final.columns:
             raise ValueError("'Status' column missing from bids data.")

        # Create Funnel Chart Data (Count occurrences of each status)
        funnel_data = bids_final['Status'].value_counts().reset_index()
        funnel_data.columns = ['Status', 'Count'] # Rename columns
        # Define order for funnel stages if possible (example order)
        status_order = ['Submitted', 'Pending', 'Won', 'Lost', 'Withdrawn'] # Adjust as needed
        funnel_data['Status'] = pd.Categorical(funnel_data['Status'], categories=status_order, ordered=True)
        funnel_data = funnel_data.sort_values('Status')

        fig_funnel = px.funnel(funnel_data, x="Count", y="Status", title="Bid Status Funnel (Filtered)")
        st.plotly_chart(fig_funnel, use_container_width=True)

        # Lost Bid Reasons Pie Chart
        lost_bids_df = bids_final[bids_final["Status"] == "Lost"]
        if not lost_bids_df.empty and 'DeclineReason' in lost_bids_df.columns:
            lost_reasons_counts = lost_bids_df['DeclineReason'].value_counts().reset_index()
            lost_reasons_counts.columns = ['DeclineReason', 'Count']
            if not lost_reasons_counts.empty:
                 fig_reasons = px.pie(lost_reasons_counts, names="DeclineReason", values="Count", title="Reasons for Lost Bids (Filtered)")
                 st.plotly_chart(fig_reasons, use_container_width=True)
                 top_lost_reason_str = f" Top lost reason: '{lost_reasons_counts.iloc[0]['DeclineReason']}' ({lost_reasons_counts.iloc[0]['Count']} bids)."
            else:
                 top_lost_reason_str = " No specific reasons recorded for lost bids."
                 st.info("No specific reasons recorded for lost bids in the filtered data.")
        else:
             top_lost_reason_str = ""
             st.info("No 'Lost' bids with 'DeclineReason' found in the filtered data.")


        # AI Explanation for Bids
        bid_analysis_context = (
            f"Bid analysis based on {len(bids_final)} filtered bids.\n"
            f"- Bid Win Rate (Won / [Won+Lost]): {bid_win_rate:.1f}%\n"
            f"- Funnel shows distribution across statuses: {funnel_data.set_index('Status')['Count'].to_dict()}\n"
            f"{top_lost_reason_str}"
        )
        render_ai_explanation("AI Analysis of Bid Performance", client, VISUAL_EXPLANATION_PROMPT, bid_analysis_context)

    except Exception as e:
        logger.error(f"Error generating bid analysis visuals: {e}", exc_info=True)
        st.error("Could not generate bid analysis visuals.")


st.divider()

# --- Section: Timeline Variance ---
st.header("⏳ Timeline Variance Analysis")
st.markdown("Scatter plot showing milestone completion variance against estimated dates based on filtered data.")

variance_plot_df = pd.DataFrame()
critical_milestones_count = 0
if not timelines_final.empty and not projects_final.empty: # Need projects for hover data potentially
    try:
        variance_calc_df = timelines_final.copy()
        # Ensure dates are datetime and Variance exists (calculated earlier)
        variance_calc_df["EstimatedDate"] = pd.to_datetime(variance_calc_df["EstimatedDate"], errors='coerce')
        variance_calc_df["ActualDate"] = pd.to_datetime(variance_calc_df["ActualDate"], errors='coerce')
        variance_calc_df = variance_calc_df.dropna(subset=["EstimatedDate", "ActualDate"])

        if not variance_calc_df.empty:
            variance_calc_df["Variance"] = (variance_calc_df["ActualDate"] - variance_calc_df["EstimatedDate"]).dt.days
            # Define 'Critical' variance (e.g., > 7 days late)
            variance_threshold = 7
            variance_calc_df["Critical"] = variance_calc_df["Variance"] > variance_threshold
            critical_milestones_count = variance_calc_df["Critical"].sum()

            # Merge with projects for more hover info if needed
            variance_plot_df = variance_calc_df.merge(
                 projects_final[['ProjectID', 'ProjectType']], # Select specific columns
                 on='ProjectID',
                 how='left'
            )

            logger.info(f"Timeline variance calculated. Avg: {avg_variance:.1f} days. Critical (> {variance_threshold} days): {critical_milestones_count}")

        else:
             logger.info("No timeline entries with valid estimated and actual dates.")

    except Exception as e:
        logger.error(f"Error processing timeline variance data: {e}", exc_info=True)
        st.error("Could not process timeline variance data.")
        variance_plot_df = pd.DataFrame()


if variance_plot_df.empty:
    st.warning("No timeline variance data available to display for the selected filters.")
else:
    fig_variance = px.scatter(
        variance_plot_df,
        x="EstimatedDate",
        y="Variance",
        color="Critical",
        title="Project Milestone Timeline Variance (Filtered)",
        hover_data=["ProjectID", "Milestone", "ProjectType", "ActualDate"],
        labels={'EstimatedDate': 'Estimated Completion Date', 'Variance': 'Variance (Days)', 'Critical': f'Delayed > {variance_threshold} Days'},
        color_discrete_map={True: "red", False: "blue"} # Explicit colors
    )
    st.plotly_chart(fig_variance, use_container_width=True)

    # AI Explanation for Variance
    variance_context = (
        f"Timeline variance analysis based on {len(variance_plot_df)} completed milestones from filtered projects.\n"
        f"- Average Variance: {avg_variance:.1f} days.\n"
        f"- Critical Delays (> {variance_threshold} days): {critical_milestones_count} milestones.\n"
        f"- Variance Range: {variance_plot_df['Variance'].min()} to {variance_plot_df['Variance'].max()} days.\n"
    )
    render_ai_explanation("AI Analysis of Timeline Variance", client, VISUAL_EXPLANATION_PROMPT, variance_context)


st.divider()

# --- Section: Predict Project Profitability ---
st.header("🔮 Predict Project Profitability")
st.markdown("Estimate profit margin for a new project based on historical data (trained on all completed projects).")

# --- Model Training (using original data, cached resource) ---
@st.cache_resource
def train_profit_model():
    """Loads data, preprocesses completed projects, and trains the profit model."""
    logger.info("Attempting to train project profit prediction model...")
    data_for_train = load_all_data()
    if data_for_train is None or 'projects' not in data_for_train or 'change_orders' not in data_for_train:
        logger.error("Cannot train profit model: Project or change order data unavailable.")
        return None, None, None # Model, features, encoders

    projects_orig = data_for_train['projects'].copy()
    change_orders_orig = data_for_train['change_orders'].copy()

    if projects_orig.empty:
         logger.warning("Original project data is empty. Cannot train profit model.")
         return None, None, None

    # --- Preprocessing ---
    # Use only completed projects with valid profit margin for training
    df_ml = projects_orig[projects_orig["Status"] == "Completed"].copy()
    # ProfitMargin should be calculated & numeric from initial load/recalc step
    # Ensure key numeric features exist and handle potential errors/NaNs
    numeric_features = ['BidAmount', 'SquareFootage', 'EstimatedHours', 'ActualHours', 'TotalExpenses', 'ProfitMargin']
    for col in numeric_features:
        if col in df_ml.columns:
            df_ml[col] = pd.to_numeric(df_ml[col], errors='coerce')
        else:
            logger.warning(f"Profit predictor training: Column '{col}' missing. Assigning 0.")
            df_ml[col] = 0.0
    # Fill NaNs for features used in model (use 0 or median/mean as appropriate)
    fill_zero_cols = ['BidAmount', 'SquareFootage', 'EstimatedHours', 'ActualHours', 'TotalExpenses']
    for col in fill_zero_cols:
         df_ml[col] = df_ml[col].fillna(0)

    # Process Change Orders (ensure required columns exist)
    change_totals = pd.DataFrame()
    if not change_orders_orig.empty and all(c in change_orders_orig.columns for c in ['ProjectID', 'Amount', 'Status']):
         try:
              co_calc = change_orders_orig.copy()
              co_calc['ProjectID'] = co_calc['ProjectID'].astype(str)
              co_calc['Amount'] = pd.to_numeric(co_calc['Amount'], errors='coerce').fillna(0)
              approved_co = co_calc[co_calc['Status'] == 'Approved']
              if not approved_co.empty:
                   change_totals = approved_co.groupby('ProjectID')['Amount'].sum().reset_index()
                   change_totals.rename(columns={'Amount': 'TotalChangeOrderAmount'}, inplace=True)
                   change_totals['ProjectID'] = change_totals['ProjectID'].astype(str) # For merge
         except Exception as e:
              logger.error(f"Error processing change orders for profit predictor training: {e}")
              change_totals = pd.DataFrame() # Ensure it's empty on error
    else:
         logger.warning("Change order data missing or lacks required columns for profit model training.")

    # Merge change order totals
    df_ml['ProjectID'] = df_ml['ProjectID'].astype(str)
    if not change_totals.empty:
         df_ml = df_ml.merge(change_totals, on='ProjectID', how='left')
    # Ensure column exists even if merge failed or no COs
    if 'TotalChangeOrderAmount' not in df_ml.columns:
        df_ml['TotalChangeOrderAmount'] = 0.0
    df_ml['TotalChangeOrderAmount'] = df_ml['TotalChangeOrderAmount'].fillna(0)

    # Feature Engineering & Encoding
    encoders = {}
    cat_features = ['ProjectType', 'ConstructionType'] # Add others if relevant e.g., 'City'
    final_features = ['BidAmount', 'SquareFootage', 'EstimatedHours', 'TotalChangeOrderAmount'] # Base numeric features

    for col in cat_features:
        if col in df_ml.columns:
            df_ml[col] = df_ml[col].astype(str).fillna('Unknown')
            if df_ml[col].nunique() > 1:
                le = LabelEncoder()
                try:
                    df_ml[f'{col}Encoded'] = le.fit_transform(df_ml[col])
                    final_features.append(f'{col}Encoded')
                    encoders[col] = le
                except Exception as e_enc:
                    logger.error(f"Error encoding categorical feature '{col}' for profit model: {e_enc}")
            else:
                 logger.warning(f"Skipping encoding for '{col}' (<= 1 unique value).")
        else:
             logger.warning(f"Categorical feature '{col}' not found in project data.")

    # Target variable
    target = 'ProfitMargin'
    # Drop rows where target is NaN (essential for training)
    df_ml = df_ml.dropna(subset=[target])

    # Ensure all final features exist before final dropna
    final_features = [f for f in final_features if f in df_ml.columns]
    if not final_features:
         logger.error("No valid features found for profit model training after preprocessing.")
         return None, None, None
    df_ml = df_ml.dropna(subset=final_features) # Drop rows with NaN in features

    if df_ml.empty or len(df_ml) < 10:
        logger.warning(f"Insufficient valid data ({len(df_ml)} rows) after preprocessing for profit model.")
        return None, None, None

    X = df_ml[final_features]
    y = df_ml[target]

    # Final check for NaNs/Infs
    if X.isnull().values.any() or y.isnull().values.any() or np.isinf(X.values).any() or np.isinf(y.values).any():
        logger.error("NaN or Inf values detected in final training data (X or y) for profit model.")
        return None, None, None

    # --- Training ---
    try:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10, min_samples_split=5)
        model.fit(X, y)
        logger.info(f"Project profit prediction model trained successfully with features: {final_features}")
        return model, final_features, encoders # Return model, features used, and encoders
    except Exception as e:
        logger.error(f"Error training project profit model: {e}", exc_info=True)
        return None, None, None

# Train the model (runs once and caches)
profit_model, profit_features, profit_encoders = train_profit_model()

# --- Prediction Form ---
if profit_model is None or not profit_features or profit_encoders is None:
    st.warning("Project profit prediction model is unavailable due to training issues.")
else:
    st.markdown("**Enter New Project Details:**")
    # Initialize session state
    if 'proj_profit_prediction_result' not in st.session_state:
        st.session_state.proj_profit_prediction_result = None
    if 'proj_profit_prediction_context' not in st.session_state:
        st.session_state.proj_profit_prediction_context = None
    if 'proj_profit_prediction_inputs' not in st.session_state:
        # Initialize with sensible defaults, using encoder classes where possible
        st.session_state.proj_profit_prediction_inputs = {
            'BidAmount': 100000.0, 'SquareFootage': 5000, 'EstimatedHours': 150, 'TotalChangeOrderAmount': 0.0,
            # Initialize categorical defaults from encoders
            **{cat: enc.classes_[0] if len(enc.classes_)>0 else 'Unknown' for cat, enc in profit_encoders.items()}
        }

    with st.form("project_profit_prediction_form"):
        # Numeric Inputs
        pred_bid_amount = st.number_input("Estimated Bid Amount ($)", min_value=1000.0, value=float(st.session_state.proj_profit_prediction_inputs.get('BidAmount', 100000.0)), step=1000.0)
        pred_sq_footage = st.number_input("Square Footage", min_value=500, value=int(st.session_state.proj_profit_prediction_inputs.get('SquareFootage', 5000)), step=100)
        pred_est_hours = st.number_input("Estimated Hours", min_value=10, value=int(st.session_state.proj_profit_prediction_inputs.get('EstimatedHours', 150)), step=10)
        pred_co_amount = st.number_input("Anticipated Change Order Amount ($)", min_value=0.0, value=float(st.session_state.proj_profit_prediction_inputs.get('TotalChangeOrderAmount', 0.0)), step=500.0)

        # Categorical Inputs (Dynamically created based on trained encoders)
        pred_inputs_cat = {}
        for col, encoder in profit_encoders.items():
            encoded_col_name = f'{col}Encoded'
            if encoded_col_name in profit_features: # Check if model uses this feature
                options = sorted(list(encoder.classes_))
                default_cat_value = st.session_state.proj_profit_prediction_inputs.get(col, options[0] if options else 'Unknown')
                default_index = options.index(default_cat_value) if default_cat_value in options else 0
                selectbox_key = f"pred_proj_{col.lower().replace(' ', '_')}"
                pred_inputs_cat[col] = st.selectbox(f"{col.replace('Type', ' Type')}", options=options, key=selectbox_key, index=default_index)

        submitted = st.form_submit_button("Predict Profit Margin")

        if submitted:
            logger.info("Project profit prediction form submitted.")
            # Store current inputs
            current_inputs_dict = {
                'BidAmount': pred_bid_amount, 'SquareFootage': pred_sq_footage,
                'EstimatedHours': pred_est_hours, 'TotalChangeOrderAmount': pred_co_amount,
                **pred_inputs_cat # Add selected categorical values
            }
            st.session_state.proj_profit_prediction_inputs = current_inputs_dict # Update session state

            try:
                # Prepare input for prediction model
                input_dict_for_df = {
                     'BidAmount': pred_bid_amount, 'SquareFootage': pred_sq_footage,
                     'EstimatedHours': pred_est_hours, 'TotalChangeOrderAmount': pred_co_amount
                }
                # Encode categorical inputs
                for col, selected_value in pred_inputs_cat.items():
                    encoded_col_name = f'{col}Encoded'
                    if encoded_col_name in profit_features: # Check if model expects it
                        encoder = profit_encoders.get(col)
                        # Handle unseen values if necessary (optional, could error instead)
                        if selected_value not in encoder.classes_:
                             # Option 1: Raise error or use default like 'Unknown' if trained
                             if 'Unknown' in encoder.classes_:
                                  selected_value_for_transform = 'Unknown'
                                  st.warning(f"Value '{selected_value}' for {col} not seen during training. Using 'Unknown'.")
                             else:
                                  raise ValueError(f"Cannot encode unseen value '{selected_value}' for {col}. Available: {encoder.classes_}")
                        else:
                            selected_value_for_transform = selected_value

                        input_dict_for_df[encoded_col_name] = encoder.transform([selected_value_for_transform])[0]

                # Create DataFrame and reindex to match training features
                input_df = pd.DataFrame([input_dict_for_df])
                input_df = input_df.reindex(columns=profit_features, fill_value=0) # Crucial step

                # Predict
                prediction = profit_model.predict(input_df)[0]
                # Optional: Clip prediction to realistic bounds (e.g., -50% to +70%)
                prediction = np.clip(prediction, -0.5, 0.7)

                # --- Store results ---
                st.session_state.proj_profit_prediction_result = prediction
                st.session_state.proj_profit_prediction_context = (
                    "INPUTS USED FOR PREDICTION:\n"
                    f"- Bid Amount: ${pred_bid_amount:,.2f}\n"
                    f"- Square Footage: {pred_sq_footage}\n"
                    f"- Estimated Hours: {pred_est_hours}\n"
                    f"- Anticipated Change Order Amount: ${pred_co_amount:,.2f}\n"
                    + "".join([f"- {col.replace('Type', ' Type')}: {val}\n" for col, val in pred_inputs_cat.items()]) + # Show selected categories
                    f"\nPREDICTED PROFIT MARGIN: {prediction:.2%}\n\n"
                    f"(Historical Avg Completed Project Margin in current filter: {avg_profit:.1%})\n" # Show filtered avg
                    "Note: Prediction based on historical patterns in all completed projects. Actual results depend heavily on execution, cost control, and unforeseen events."
                )
                logger.info(f"Project profit prediction successful: {prediction:.2%}")

            except ValueError as ve:
                 st.error(f"Error processing prediction input: {ve}")
                 logger.error(f"Project profit prediction failed due to input error: {ve}", exc_info=True)
                 st.session_state.proj_profit_prediction_result = None
                 st.session_state.proj_profit_prediction_context = None
            except Exception as e:
                st.error(f"Error during project profit prediction calculation: {e}")
                logger.error("Project profit prediction failed.", exc_info=True)
                st.session_state.proj_profit_prediction_result = None
                st.session_state.proj_profit_prediction_context = None
            # Rerun implicitly

    # --- Display results OUTSIDE the form ---
    if st.session_state.proj_profit_prediction_result is not None:
        st.success(f"**Predicted Profit Margin: {st.session_state.proj_profit_prediction_result:.2%}**")
        if st.session_state.proj_profit_prediction_context and client:
             render_ai_explanation(
                 "AI Explanation of Profit Prediction",
                 client,
                 VISUAL_EXPLANATION_PROMPT, # Can use generic
                 st.session_state.proj_profit_prediction_context
             )
        elif not client:
             st.warning("AI Client unavailable, cannot generate explanation.")
        if st.button("Clear Prediction Result", key="clear_proj_profit_pred"):
             st.session_state.proj_profit_prediction_result = None
             st.session_state.proj_profit_prediction_context = None
             st.rerun()

st.divider()

# --- Section: Chat Feature ---
st.header("💬 Dig Into The Details") # Header matching the requested structure

# Prepare context for the chat (reuse proj_summary_context if available)
# Ensure proj_summary_context is defined earlier in the script
if 'proj_summary_context' not in locals():
     proj_summary_context = f"Project data analysis for {len(projects_final)} filtered projects."
     logger.warning("Using fallback chat context as proj_summary_context was not defined.")

chat_base_context = proj_summary_context # Use the determined context

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
    page_key="project", # Unique key for this page
    placeholder="Ask about pipeline, bids, timelines, or profitability..." # Page-specific placeholder
)