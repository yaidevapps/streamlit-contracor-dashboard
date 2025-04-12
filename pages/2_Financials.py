# pages/2_Financials.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import logging
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor # For Profit Prediction
from sklearn.preprocessing import LabelEncoder   # For Profit Prediction

# Import shared utilities
from utils.data_utils import load_all_data
from utils.filter_utils import render_global_filters, apply_filters, recalculate_costs_after_filtering
from utils.ai_utils import get_genai_client, render_ai_explanation, render_chat_interface, SUMMARY_ADVICE_PROMPT, VISUAL_EXPLANATION_PROMPT, CHAT_PROMPT_TEMPLATE
from config import CSS_FILE, CURRENT_DATE # Import necessary config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Page Setup and Styling ---
st.set_page_config(layout="wide")
try:
    with open(CSS_FILE) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("CSS file not found.")

st.title("Financials Dashboard")

# --- AI Client Initialization ---
client = get_genai_client()
if not client:
    st.error("AI Client failed to initialize. AI features will be unavailable.")

# --- Data Loading ---
# data_orig contains the full, initially processed dataset
data_orig = load_all_data()
if data_orig is None:
    st.error("Fatal error loading data. Dashboard cannot be displayed.")
    st.stop()

# Prepare employee rates (used in recalc) from original data
employee_rates = pd.DataFrame()
if "employees" in data_orig and not data_orig["employees"].empty:
     emp_df_rates = data_orig["employees"].copy()
     emp_df_rates['EmployeeID'] = emp_df_rates['EmployeeID'].astype(str)
     # Ensure rate columns exist before indexing
     rate_cols = [col for col in ['HourlyRate', 'OvertimeRate'] if col in emp_df_rates.columns]
     if rate_cols:
          employee_rates = emp_df_rates.drop_duplicates(subset=['EmployeeID']).set_index('EmployeeID')[rate_cols]
     else:
          logger.error("HourlyRate or OvertimeRate missing from employees data.")


# --- Global Filters ---
start_date, end_date, city, project_type = render_global_filters(data_orig)

# --- Apply Filters and Recalculate Costs ---
# filtered_data contains data relevant to the current filter selection
filtered_data = apply_filters(data_orig, start_date, end_date, city, project_type)
if not filtered_data:
    st.error("Error applying filters or no data matches filters. Cannot display filtered data.")
    # Provide option to reset filters or display message
    st.stop()

# Recalculate costs based on filtered time entries
filtered_data = recalculate_costs_after_filtering(filtered_data, employee_rates)

# --- Extract Filtered DataFrames for Display ---
projects_final = filtered_data.get('projects', pd.DataFrame())
service_calls_final = filtered_data.get('service_calls', pd.DataFrame())
invoices_final = filtered_data.get('invoices', pd.DataFrame())
expenses_final = filtered_data.get('expenses', pd.DataFrame())
bids_final = filtered_data.get('bids', pd.DataFrame())
time_entries_final = filtered_data.get('time_entries', pd.DataFrame())

# --- Calculate Financial Metrics (Using Filtered & Recalculated Data) ---
total_revenue_paid = invoices_final[invoices_final["Status"] == "Paid"]["Amount"].sum() if not invoices_final.empty else 0

total_project_labor_cost = projects_final['ProjectLaborCost'].sum() if not projects_final.empty and 'ProjectLaborCost' in projects_final.columns else 0
total_project_other_expenses = projects_final['OtherExpenses'].sum() if not projects_final.empty and 'OtherExpenses' in projects_final.columns else 0
total_service_call_labor_cost = service_calls_final['ServiceLaborCost'].sum() if not service_calls_final.empty and 'ServiceLaborCost' in service_calls_final.columns else 0
total_service_call_materials = service_calls_final['MaterialsCost'].sum() if not service_calls_final.empty and 'MaterialsCost' in service_calls_final.columns else 0
overhead_expenses = expenses_final[expenses_final['RelatedProjectID'].fillna('None') == 'None']['Amount'].sum() if not expenses_final.empty else 0

total_combined_expenses = total_project_labor_cost + total_project_other_expenses + \
                         total_service_call_labor_cost + total_service_call_materials + \
                         overhead_expenses

completed_projects_filtered = projects_final[projects_final["Status"] == "Completed"] if not projects_final.empty else pd.DataFrame()
# Ensure ProfitMargin column exists before calculating mean
avg_project_profit_margin = 0.0
if not completed_projects_filtered.empty and 'ProfitMargin' in completed_projects_filtered.columns:
     # Make sure it's numeric before calculating mean
     completed_projects_filtered['ProfitMargin'] = pd.to_numeric(completed_projects_filtered['ProfitMargin'], errors='coerce')
     avg_project_profit_margin = completed_projects_filtered['ProfitMargin'].mean()
     if pd.isna(avg_project_profit_margin): # Handle case where all margins are NaN
          avg_project_profit_margin = 0.0


# --- Render Financials Sections ---

# Financial Summary
st.header("📋 Financial Summary")
# ... (Summary rendering code remains the same) ...
st.markdown("Key metrics based on the selected filters.")
col1, col2, col3 = st.columns(3)
col1.metric("Total Revenue (Paid Invoices)", f"${total_revenue_paid:,.2f}")
col2.metric("Total Expenses (All Costs)", f"${total_combined_expenses:,.2f}")
col3.metric("Avg. Project Profit Margin (Completed)", f"{avg_project_profit_margin:.2%}")

# Reconciliation Check
expected_revenue_projects = completed_projects_filtered["FinalAmount"].sum() if not completed_projects_filtered.empty else 0
expected_revenue_services = service_calls_final["BilledAmount"].sum() if not service_calls_final.empty else 0
total_expected_revenue = expected_revenue_projects + expected_revenue_services
revenue_diff = total_revenue_paid - total_expected_revenue
if total_expected_revenue > 0 and abs(revenue_diff) > 0.15 * total_expected_revenue: # 15% tolerance
    st.warning(
        f"Revenue Check: Paid Invoices (${total_revenue_paid:,.2f}) vs. "
        f"Expected Revenue (Completed Projects + Billed Services = ${total_expected_revenue:,.2f}). "
        f"Difference: ${revenue_diff:,.2f}. "
        "Note: Differences expected due to payment timing, outstanding invoices, filtering effects."
    )

# AI Financial Insights
st.subheader("💡 AI Financial Insights")
# ... (AI context generation and rendering remain the same) ...
fin_summary_context = (
    f"Financial snapshot (Filtered Period: {start_date.strftime('%Y-%m-%d') if start_date else 'N/A'} to {end_date.strftime('%Y-%m-%d') if end_date else 'N/A'}, City: {city}, Type: {project_type}):\n"
    f"- Revenue (Paid): ${total_revenue_paid:,.0f}\n"
    f"- Total Expenses: ${total_combined_expenses:,.0f} (Proj Labor: ${total_project_labor_cost:,.0f}, Proj Other: ${total_project_other_expenses:,.0f}, Serv Labor: ${total_service_call_labor_cost:,.0f}, Serv Mat: ${total_service_call_materials:,.0f}, Overhead: ${overhead_expenses:,.0f})\n"
    f"- Projects: {len(projects_final)} filtered, {len(completed_projects_filtered)} completed.\n"
    f"- Avg. Completed Project Profit Margin: {avg_project_profit_margin:.1%}.\n"
    f"Note: Costs/Margins recalculated based on filtered data. Check raw data if margins seem unrealistic."
)
# --- MODIFIED CALL ---
render_ai_explanation(
    "Get AI-Powered Financial Advice",
    client,
    SUMMARY_ADVICE_PROMPT, # The template containing {context} and {current_date}
    fin_summary_context,   # The data context string for {context}
    additional_format_args={'current_date': CURRENT_DATE.strftime('%Y-%m-%d')} # Pass current_date separately
)


st.divider()

# Revenue Trends
st.header("📈 Revenue Trends")
# ... (Revenue trends rendering code remains the same) ...
if not invoices_final.empty and 'InvoiceDate' in invoices_final.columns:
    invoices_final['InvoiceDate'] = pd.to_datetime(invoices_final['InvoiceDate'], errors='coerce')
    monthly_revenue = invoices_final[invoices_final['Status']=='Paid'].set_index('InvoiceDate')['Amount'].resample('ME').sum().reset_index()
    if not monthly_revenue.empty and monthly_revenue['Amount'].sum() > 0:
        fig_trends = px.line(monthly_revenue, x="InvoiceDate", y="Amount", title="Paid Revenue Trends by Month (Filtered)", markers=True)
        fig_trends.update_layout(xaxis_title="Month", yaxis_title="Paid Revenue ($)")
        st.plotly_chart(fig_trends, use_container_width=True)
        # AI Explanation
        try:
            peak_month = monthly_revenue.loc[monthly_revenue['Amount'].idxmax()]
            low_month = monthly_revenue.loc[monthly_revenue['Amount'].idxmin()]
            revenue_trends_context = (
                f"Chart shows monthly paid revenue (filtered data) from {monthly_revenue['InvoiceDate'].min().strftime('%Y-%m')} to {monthly_revenue['InvoiceDate'].max().strftime('%Y-%m')}. "
                f"Total paid revenue in period: ${monthly_revenue['Amount'].sum():,.0f}. Peak: {peak_month['InvoiceDate'].strftime('%Y-%m')} (${peak_month['Amount']:,.0f}). Low: {low_month['InvoiceDate'].strftime('%Y-%m')} (${low_month['Amount']:,.0f})."
            )
            render_ai_explanation("AI Explanation of Revenue Trends", client, VISUAL_EXPLANATION_PROMPT, revenue_trends_context)
        except Exception as e:
            logger.error(f"Error generating context for revenue trends AI: {e}")
            st.warning("Could not generate AI explanation for revenue trends.")
    else: st.warning("No paid invoice data available for the selected filters.")
else: st.warning("No invoice data available for the selected filters.")


st.divider()

# Bid Performance
st.header("🎯 Bid Performance")
# ... (Bid performance rendering code remains the same) ...
if not bids_final.empty:
    bid_status_counts = bids_final["Status"].value_counts()
    won_bids = bid_status_counts.get("Won", 0)
    lost_bids = bid_status_counts.get("Lost", 0)
    pending_bids = bid_status_counts.get("Pending", 0)
    other_bids = len(bids_final) - (won_bids + lost_bids + pending_bids)

    total_submitted_bids = won_bids + lost_bids
    bid_to_win_ratio = won_bids / total_submitted_bids if total_submitted_bids > 0 else 0

    col_b1, col_b2, col_b3, col_b4 = st.columns(4)
    col_b1.metric("Bid Win Rate", f"{bid_to_win_ratio:.1%}")
    col_b2.metric("Bids Won", f"{won_bids}")
    col_b3.metric("Bids Lost", f"{lost_bids}")
    col_b4.metric("Bids Pending/Other", f"{pending_bids + other_bids}")
    st.caption(f"Win rate calculated as Won / (Won + Lost) within the filtered period.")

    # Status Distribution Chart
    bid_status_df = bid_status_counts.reset_index()
    # Ensure correct column names if needed after reset_index
    bid_status_df.columns = ['Status', 'Count'] if len(bid_status_df.columns) == 2 else bid_status_df.columns

    if not bid_status_df.empty and 'Status' in bid_status_df.columns and 'Count' in bid_status_df.columns:
         fig_bid_status = px.bar(bid_status_df, x='Status', y='Count', title='Bid Status Distribution (Filtered)', text_auto=True)
         st.plotly_chart(fig_bid_status, use_container_width=True)

    # Lost Reasons Analysis
    lost_reasons_df = bids_final[bids_final["Status"] == "Lost"]
    if not lost_reasons_df.empty and 'DeclineReason' in lost_reasons_df.columns:
         lost_reasons_counts = lost_reasons_df['DeclineReason'].value_counts().reset_index()
         # Ensure column names after value_counts().reset_index()
         lost_reasons_counts.columns = ['DeclineReason', 'count'] if len(lost_reasons_counts.columns) == 2 else lost_reasons_counts.columns

         if not lost_reasons_counts.empty and 'DeclineReason' in lost_reasons_counts.columns and 'count' in lost_reasons_counts.columns:
             with st.expander("Analysis of Lost Bid Reasons (Filtered)"):
                  fig_lost_reasons = px.pie(lost_reasons_counts, values='count', names='DeclineReason', title='Lost Bid Reasons')
                  st.plotly_chart(fig_lost_reasons, use_container_width=True)
                  top_lost_reason_str = f" Top lost reason: '{lost_reasons_counts.iloc[0]['DeclineReason']}' ({lost_reasons_counts.iloc[0]['count']} bids)."
         else:
             top_lost_reason_str = ""
    else:
         top_lost_reason_str = ""


    # AI Explanation
    bid_performance_context = (
        f"Bid performance (filtered) shows {len(bids_final)} total bids. "
        f"Counts: Won: {won_bids}, Lost: {lost_bids}, Pending/Other: {pending_bids + other_bids}. "
        f"Win rate (Won / (Won + Lost)): {bid_to_win_ratio:.1%}."
        f"{top_lost_reason_str}"
    )
    render_ai_explanation("AI Explanation of Bid Performance", client, VISUAL_EXPLANATION_PROMPT, bid_performance_context)

else:
    st.warning("No bid data available for the selected filters.")


st.divider()

# Expense Flow
st.header("💸 Expense Flow Breakdown")
# ... (Expense flow rendering code remains the same) ...
st.markdown("Visualizing the flow of all expenses within the filtered period.")
sankey_nodes = ["Project Labor", "Project Other Costs", "Service Call Labor", "Service Call Materials", "Overhead", "Total Expenses"]
sankey_values = [
    max(0, total_project_labor_cost), max(0, total_project_other_expenses),
    max(0, total_service_call_labor_cost), max(0, total_service_call_materials),
    max(0, overhead_expenses)
]

valid_indices = [i for i, v in enumerate(sankey_values) if v > 0.1]
if valid_indices:
    final_nodes_labels = [sankey_nodes[i] for i in valid_indices] + [sankey_nodes[5]] # Source nodes + Target node
    node_map = {name: i for i, name in enumerate(final_nodes_labels)}

    final_sources_indices = [node_map[sankey_nodes[i]] for i in valid_indices]
    final_targets_indices = [node_map[sankey_nodes[5]]] * len(final_sources_indices)
    final_values = [sankey_values[i] for i in valid_indices]

    fig_expense = go.Figure(go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=final_nodes_labels, color="#87CEEB"),
        link=dict(source=final_sources_indices, target=final_targets_indices, value=final_values, color="#d3d3d3")
    ))
    fig_expense.update_layout(title_text="Expense Categories Flowing to Total Expenses (Filtered)", font_size=12)
    st.plotly_chart(fig_expense, use_container_width=True)

    # AI Explanation
    expense_context = (
        f"Total Expenses (${total_combined_expenses:,.0f}) breakdown (filtered period):\n"
        f"- Project Labor: ${total_project_labor_cost:,.0f}\n"
        f"- Project Other Costs: ${total_project_other_expenses:,.0f}\n"
        f"- Service Call Labor: ${total_service_call_labor_cost:,.0f}\n"
        f"- Service Call Materials: ${total_service_call_materials:,.0f}\n"
        f"- Overhead: ${overhead_expenses:,.0f}"
    )
    render_ai_explanation("AI Explanation of Expense Flow", client, VISUAL_EXPLANATION_PROMPT, expense_context)
else:
    st.warning("No significant expense data available for the selected filters.")

st.divider()

# --- Project Profit Margin Predictor ---
st.header("🔮 Project Profit Margin Predictor")
st.markdown("Estimate the potential profit margin for a new project based on historical data (trained on all completed projects).")

# --- Model Training Section (using data_orig) ---
profit_model = None
profit_features = []
profit_encoders = {} # Dictionary to store encoders for categorical features

# Define preprocessing function within the page or import if moved to utils
def preprocess_for_profit_prediction(df_projects_orig, df_change_orders_orig):
    """Prepares data for profit margin prediction training."""
    if df_projects_orig is None or df_projects_orig.empty:
         logger.error("Project data is empty or None for profit predictor preprocessing.")
         return pd.DataFrame(), [], {}, {} # Return empty results

    # Use only completed projects for training
    df_ml = df_projects_orig[df_projects_orig["Status"] == "Completed"].copy()
    if df_ml.empty:
         logger.warning("No 'Completed' projects found in original data for training profit model.")
         return pd.DataFrame(), [], {}, {}

    # --- Data Cleaning & Feature Engineering ---
    # Ensure necessary numeric columns exist and are numeric
    # Note: TotalExpenses, ProjectLaborCost, OtherExpenses should be present from initial load in data_utils
    required_numeric = ['FinalAmount', 'BidAmount', 'SquareFootage', 'EstimatedHours', 'TotalExpenses', 'ProjectLaborCost', 'OtherExpenses', 'ActualHours']
    for col in required_numeric:
        if col in df_ml.columns:
            df_ml[col] = pd.to_numeric(df_ml[col], errors='coerce')
        else:
             logger.warning(f"Predictor preprocessing: Column '{col}' missing in projects. Assigning 0.")
             df_ml[col] = 0.0 # Add column as 0 if missing

    # Fill NaNs in key numeric features used for modeling or calculation
    fill_zero_cols = ['FinalAmount', 'BidAmount', 'SquareFootage', 'EstimatedHours', 'ActualHours', 'TotalExpenses', 'ProjectLaborCost', 'OtherExpenses']
    for col in fill_zero_cols:
         # Check column exists before filling
         if col in df_ml.columns:
            df_ml[col] = df_ml[col].fillna(0)
         else:
            df_ml[col] = 0.0 # Ensure it exists

    # --- Handle Change Orders ---
    change_orders_amount = pd.DataFrame() # Default empty
    # Check if change order data is valid and has required columns
    if df_change_orders_orig is not None and not df_change_orders_orig.empty \
       and all(c in df_change_orders_orig.columns for c in ['ProjectID', 'Amount', 'Status']):
        try:
            df_co_copy = df_change_orders_orig.copy()
            df_co_copy['ProjectID'] = df_co_copy['ProjectID'].astype(str)
            df_co_copy['Amount'] = pd.to_numeric(df_co_copy['Amount'], errors='coerce').fillna(0)
            approved_co = df_co_copy[df_co_copy['Status'] == 'Approved']
            if not approved_co.empty:
                change_orders_amount = approved_co.groupby('ProjectID')['Amount'].sum().reset_index()
                change_orders_amount.rename(columns={'Amount': 'TotalChangeOrderAmount'}, inplace=True)
                change_orders_amount['ProjectID'] = change_orders_amount['ProjectID'].astype(str) # Ensure type for merge
            else:
                logger.info("No 'Approved' change orders found for predictor.")
        except Exception as e:
            logger.error(f"Error processing change orders for predictor: {e}")
            change_orders_amount = pd.DataFrame() # Reset on error
    else:
         logger.warning("Change order data missing, empty, or lacks required columns for predictor.")

    # Merge Change Order Amount
    df_ml['ProjectID'] = df_ml['ProjectID'].astype(str) # Ensure key type match before merge
    if not change_orders_amount.empty:
         df_ml = df_ml.merge(change_orders_amount, on='ProjectID', how='left')

    # Ensure column exists after merge and fill NaNs
    if 'TotalChangeOrderAmount' not in df_ml.columns:
        df_ml['TotalChangeOrderAmount'] = 0.0
    df_ml['TotalChangeOrderAmount'] = df_ml['TotalChangeOrderAmount'].fillna(0)

    # --- Recalculate Profit Margin for Training Data ---
    df_ml['ProfitMarginCalc'] = 0.0
    # Ensure TotalExpenses exists (should be calculated during initial load)
    if 'TotalExpenses' not in df_ml.columns:
         logger.error("Cannot calculate ProfitMarginCalc for training: TotalExpenses missing.")
         return pd.DataFrame(), [], {}, {} # Cannot proceed

    revenue_mask = df_ml['FinalAmount'].notna() & (df_ml['FinalAmount'].abs() > 1e-9)
    df_ml.loc[revenue_mask, 'ProfitMarginCalc'] = (df_ml['FinalAmount'] - df_ml['TotalExpenses']) / df_ml['FinalAmount']
    df_ml['ProfitMarginCalc'] = df_ml['ProfitMarginCalc'].replace([np.inf, -np.inf], 0).fillna(0)

    # --- Feature Selection & Encoding ---
    # Use features likely known before project start/completion
    features = ['BidAmount', 'SquareFootage', 'EstimatedHours', 'TotalChangeOrderAmount']
    target = 'ProfitMarginCalc' # Target is the calculated profit margin

    encoders = {} # Store fitted encoders
    cat_features = ['ProjectType', 'ConstructionType'] # Define categorical features
    for col in cat_features:
        if col in df_ml.columns:
            df_ml[col] = df_ml[col].astype(str).fillna('Unknown') # Handle NaNs before encoding
            if df_ml[col].nunique() > 1: # Only encode if more than one category
                le = LabelEncoder()
                try:
                    df_ml[f'{col}Encoded'] = le.fit_transform(df_ml[col])
                    features.append(f'{col}Encoded')
                    encoders[col] = le # Store the fitted encoder
                    logger.info(f"Encoded feature '{col}' for profit predictor. Classes: {le.classes_}")
                except Exception as e:
                    logger.error(f"Error encoding column '{col}' for profit predictor: {e}")
            else:
                logger.warning(f"Skipping encoding for '{col}' in profit predictor (<= 1 unique value).")
        else:
            logger.warning(f"Categorical feature '{col}' not found for profit predictor.")

    # --- Final Data Check ---
    # Ensure all selected features actually exist in df_ml before dropna
    final_features_check = [f for f in features if f in df_ml.columns]
    if target not in df_ml.columns:
        logger.error(f"Target column '{target}' for profit prediction not found after preprocessing.")
        return pd.DataFrame(), [], {}, {}

    # Drop rows with NaN in final selected features or the calculated target
    initial_rows = len(df_ml)
    df_ml = df_ml.dropna(subset=final_features_check + [target])
    dropped_rows = initial_rows - len(df_ml)
    if dropped_rows > 0:
         logger.info(f"Dropped {dropped_rows} rows with NaNs in features/target before training profit model.")

    if df_ml.empty:
        logger.warning("No valid data remaining after preprocessing for profit model training.")
        return pd.DataFrame(), [], {}, {}

    logger.info(f"Profit predictor preprocessing complete. Training data shape: {df_ml.shape}, Features: {final_features_check}")
    # Return only the features actually used/created
    return df_ml, final_features_check, target, encoders

@st.cache_resource # Cache the trained model resource
def train_profit_model(_df_ml, features, target):
    """Trains the profit prediction model."""
    # Basic validation
    if _df_ml is None or _df_ml.empty or not features or target not in _df_ml.columns:
        logger.error("Insufficient data or configuration for training profit model.")
        return None
    if len(_df_ml) < 10: # Minimum samples check
        logger.warning(f"Only {_df_ml.shape[0]} samples available for profit model training. Model may be unreliable.")
        # return None # Optionally stop if too few samples

    # Ensure features exist and select data
    features = [f for f in features if f in _df_ml.columns] # Use only existing features
    if not features:
        logger.error("No valid features remain for training profit model.")
        return None

    X = _df_ml[features]
    y = _df_ml[target]

    # Final checks before training
    if X.isnull().values.any() or y.isnull().values.any() or np.isinf(X.values).any() or np.isinf(y.values).any():
         logger.error("NaN or Inf values detected in profit model training data X or y. Aborting training.")
         return None

    try:
        # Define and train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10, min_samples_split=5)
        model.fit(X, y)
        logger.info(f"Profit prediction model trained successfully with features: {features}")
        return model
    except Exception as e:
        logger.error(f"Error training profit prediction model: {e}", exc_info=True)
        return None

# --- Train the model using original data (run only once per session due to caching) ---
# Extract necessary original dataframes safely
projects_orig_train = data_orig.get('projects')
change_orders_orig_train = data_orig.get('change_orders')

# Check if data loading was successful before proceeding
if projects_orig_train is not None and not projects_orig_train.empty:
    # Ensure required columns for preprocessing exist in the original data
    required_preprocess_cols = ['Status', 'ProjectID', 'ProjectType', 'ConstructionType'] # Base requirements
    if all(col in projects_orig_train.columns for col in required_preprocess_cols):
        ml_data_profit, profit_features, profit_target, profit_encoders = preprocess_for_profit_prediction(projects_orig_train, change_orders_orig_train)
        # Check if preprocessing was successful
        if not ml_data_profit.empty and profit_features:
            profit_model = train_profit_model(ml_data_profit, profit_features, profit_target)
        else:
            logger.warning("Preprocessing for profit model failed or yielded no valid features/data.")
            profit_model = None
    else:
        logger.error("Original project data missing required columns for profit model preprocessing.")
        profit_model = None
else:
     logger.warning("Original project data missing or empty, cannot train profit model.")
     profit_model = None


# --- Prediction Form and Result Display ---
if profit_model is None or not profit_features:
    # Display message if model training failed
    st.warning("Profit prediction model could not be trained due to data issues. Prediction unavailable.")
else:
    st.markdown("**Enter New Project Details:**")

    # Initialize session state variables if they don't exist for prediction persistence
    if 'profit_prediction_result' not in st.session_state:
        st.session_state.profit_prediction_result = None
    if 'profit_prediction_context' not in st.session_state:
        st.session_state.profit_prediction_context = None
    if 'profit_prediction_inputs' not in st.session_state:
         # Initialize with default values matching the form defaults below
         st.session_state.profit_prediction_inputs = {
              'BidAmount': 50000.0,
              'SquareFootage': 5000,
              'EstimatedHours': 150,
              'TotalChangeOrderAmount': 0.0
              # Categorical defaults will be handled by selectbox index=0 ('Unknown')
         }

    with st.form("profit_prediction_form"):
        # --- Form Inputs ---
        # Use default values from session state to repopulate form after rerun
        pred_bid_amount = st.number_input("Estimated Bid Amount ($)", min_value=1000.0, value=float(st.session_state.profit_prediction_inputs.get('BidAmount', 50000.0)), step=1000.0)
        pred_sq_footage = st.number_input("Square Footage", min_value=1000, value=int(st.session_state.profit_prediction_inputs.get('SquareFootage', 5000)), step=100)
        pred_est_hours = st.number_input("Estimated Hours", min_value=10, value=int(st.session_state.profit_prediction_inputs.get('EstimatedHours', 150)), step=10)
        pred_co_amount = st.number_input("Anticipated Change Order Amount ($)", min_value=0.0, value=float(st.session_state.profit_prediction_inputs.get('TotalChangeOrderAmount', 0.0)), step=500.0)

        pred_inputs_cat = {}
        valid_form = True
        # Dynamically create selectboxes based on encoders used during training
        for col, encoder in profit_encoders.items():
            encoded_col_name = f'{col}Encoded'
            if encoded_col_name in profit_features: # Check if this encoded feature was actually used by the trained model
                options = sorted(list(encoder.classes_))
                # Ensure 'Unknown' is an option if it was learned or add it otherwise
                if 'Unknown' not in options:
                     options = ['Unknown'] + options

                try:
                     selectbox_key = f"pred_{col.lower().replace(' ', '_')}" # Unique key for widget
                     # Use session state value if available for this category, default to 'Unknown'
                     default_cat_value = st.session_state.profit_prediction_inputs.get(col, 'Unknown')
                     # Find index of default value, default to index of 'Unknown' or 0 if 'Unknown' isn't present
                     default_index = options.index(default_cat_value) if default_cat_value in options else (options.index('Unknown') if 'Unknown' in options else 0)
                     pred_inputs_cat[col] = st.selectbox(f"{col.replace('Type', ' Type')}", options=options, key=selectbox_key, index=default_index)
                except Exception as e:
                     logger.error(f"Error creating selectbox for {col}: {e}")
                     st.error(f"Error creating filter options for {col}. Please check data.")
                     valid_form = False
            else:
                 logger.debug(f"Skipping selectbox for '{col}' as '{encoded_col_name}' not in trained features: {profit_features}")


        # --- Form Submission ---
        submitted = st.form_submit_button("Predict Margin")

        if submitted and valid_form:
            logger.info("Profit prediction form submitted.")
            # Prepare input dictionary based on this submission's values
            current_input_data_dict = {
                'BidAmount': pred_bid_amount,
                'SquareFootage': pred_sq_footage,
                'EstimatedHours': pred_est_hours,
                'TotalChangeOrderAmount': pred_co_amount
            }
            # Store inputs used for *this* submission in session state for potential form repopulation
            st.session_state.profit_prediction_inputs = {
                'BidAmount': pred_bid_amount,
                'SquareFootage': pred_sq_footage,
                'EstimatedHours': pred_est_hours,
                'TotalChangeOrderAmount': pred_co_amount
            }

            valid_pred_input = True
            # Add encoded categorical features from this submission
            for col, selected_value in pred_inputs_cat.items():
                 encoded_col_name = f'{col}Encoded'
                 if encoded_col_name in profit_features: # Check if feature is expected by model
                     encoder = profit_encoders.get(col)
                     if encoder:
                         st.session_state.profit_prediction_inputs[col] = selected_value # Store selected category
                         try:
                             # Handle potential unseen values during prediction
                             if selected_value in encoder.classes_:
                                 encoded_val = encoder.transform([selected_value])[0]
                             elif 'Unknown' in encoder.classes_:
                                 encoded_val = encoder.transform(['Unknown'])[0]
                                 logger.warning(f"Value '{selected_value}' for {col} not seen in training. Using 'Unknown' encoding.")
                                 # Optionally show warning to user: st.warning(...)
                             else:
                                 logger.error(f"Cannot encode '{selected_value}' for {col} and 'Unknown' category not available. Prediction might be inaccurate.")
                                 st.error(f"Cannot process value '{selected_value}' for {col}. Please select a trained value or ensure 'Unknown' is handled.")
                                 encoded_val = -1 # Assign a default/error value
                                 valid_pred_input = False
                             current_input_data_dict[encoded_col_name] = encoded_val
                         except ValueError as ve:
                             logger.error(f"Error encoding input '{selected_value}' for {col}: {ve}. Check if value is in encoder classes: {encoder.classes_}")
                             st.error(f"Error processing input for {col}. Please select a valid option.")
                             valid_pred_input = False
                         except Exception as e:
                             logger.error(f"Unexpected error encoding input for {col}: {e}", exc_info=True)
                             st.error(f"Internal error processing input for {col}.")
                             valid_pred_input = False
                     else:
                          logger.error(f"Encoder not found for categorical column '{col}' during prediction.")
                          valid_pred_input = False
                 # else: Feature not needed by model, skip adding to dict

            if valid_pred_input:
                 try:
                     # Create DataFrame from current submission's data
                     input_df = pd.DataFrame([current_input_data_dict])
                     # Reindex to ensure all features model was trained on are present and in correct order
                     # Use only the features the model was actually trained with (profit_features)
                     input_df = input_df.reindex(columns=profit_features, fill_value=0)

                     logger.info(f"Input data for prediction model (reindexed):\n{input_df.to_string()}")

                     # Make prediction
                     prediction = profit_model.predict(input_df)[0]

                     # --- Store results in session state ---
                     st.session_state.profit_prediction_result = prediction
                     # Prepare context string for the explanation
                     st.session_state.profit_prediction_context = (
                         "INPUTS USED FOR PREDICTION:\n"
                         f"- Estimated Bid Amount: ${pred_bid_amount:,.2f}\n"
                         f"- Square Footage: {pred_sq_footage}\n"
                         f"- Estimated Hours: {pred_est_hours}\n"
                         f"- Anticipated Change Order Amount: ${pred_co_amount:,.2f}\n"
                         # Use the actual selections from pred_inputs_cat for clarity
                         + "".join([f"- {col.replace('Type', ' Type')}: {val}\n" for col, val in pred_inputs_cat.items()]) +
                         f"\nPREDICTED PROFIT MARGIN: {prediction:.2%}\n"
                         f"(Historical Avg Completed Project Margin in current filter: {avg_project_profit_margin:.1%})\n" # Show filtered avg for comparison
                         "Note: Prediction based on historical patterns in *all* completed projects' data available at training time. Actual results depend heavily on project execution, accurate cost tracking, and unforeseen circumstances."
                     )
                     # --- End storing results ---
                     logger.info(f"Prediction successful: {prediction:.2%}")


                 except Exception as e:
                     st.error(f"Error during profit prediction calculation: {e}")
                     logger.error(f"Profit prediction failed.", exc_info=True)
                     # Clear potentially stale results from session state on error
                     st.session_state.profit_prediction_result = None
                     st.session_state.profit_prediction_context = None
            else:
                 logger.warning("Prediction skipped due to invalid input.")
                 # Clear results if input was invalid
                 st.session_state.profit_prediction_result = None
                 st.session_state.profit_prediction_context = None
            # Form submission processed, trigger rerun implicitly by ending 'with st.form'

    # --- Display Results *OUTSIDE* the Form, using session state ---
    # This block runs on every script rerun, including the one triggered by form submission
    # and the one triggered by clicking the explanation button
    if st.session_state.profit_prediction_result is not None:
        # Display the stored prediction result
        st.success(f"**Predicted Profit Margin: {st.session_state.profit_prediction_result:.2%}**")

        # Display the AI explanation using the stored context
        if st.session_state.profit_prediction_context and client:
            render_ai_explanation(
                "AI Explanation of Prediction", # Title also used for session_state key inside render func
                client,
                VISUAL_EXPLANATION_PROMPT, # Or a more specific template if desired
                st.session_state.profit_prediction_context # Use stored context
                )
        elif not client:
            st.warning("AI Client unavailable, cannot generate explanation.")
        # Optionally add a button to clear the prediction/explanation
        if st.button("Clear Prediction Result", key="clear_profit_pred"):
             st.session_state.profit_prediction_result = None
             st.session_state.profit_prediction_context = None
             st.session_state.profit_prediction_inputs = {} # Reset inputs too
             st.rerun()


st.divider()

# --- Cash Flow Forecast ---
st.header("📅 Cash Flow Projection (Next 3 Months)")
st.markdown("Simplified projection based on outstanding invoices (due in period) and known expenses (dated in period). Uses filtered data.")

# Use final filtered dataframes
if not invoices_final.empty and not expenses_final.empty:
    try:
        today_dt = CURRENT_DATE # Use consistent 'today' from config
        future_3m = today_dt + timedelta(days=90)
        logger.info(f"Generating cash flow forecast from {today_dt.date()} to {future_3m.date()}")

        # Ensure date columns are datetime
        invoices_final['DueDate'] = pd.to_datetime(invoices_final['DueDate'], errors='coerce')
        expenses_final['Date'] = pd.to_datetime(expenses_final['Date'], errors='coerce')

        # Filter for outstanding/upcoming invoices within the next 3 months
        # Include Overdue as they might be paid soon
        inflow_statuses = ['Outstanding', 'Upcoming', 'Overdue']
        outstanding_invoices = invoices_final[
            (invoices_final['Status'].isin(inflow_statuses)) &
            (invoices_final['DueDate'].notna()) &
            # Using DueDate to estimate *when* inflow might occur
            (invoices_final['DueDate'] >= today_dt) &
            (invoices_final['DueDate'] <= future_3m)
        ].copy()
        logger.info(f"Found {len(outstanding_invoices)} invoices due in forecast period.")

        # Estimate inflow based on due date month
        monthly_inflows = pd.Series(dtype=float) # Ensure initialization
        if not outstanding_invoices.empty:
            outstanding_invoices['DueMonth'] = outstanding_invoices['DueDate'].dt.to_period('M')
            monthly_inflows = outstanding_invoices.groupby('DueMonth')['Amount'].sum()
        else:
            logger.info("No outstanding/upcoming invoices found in the forecast period.")


        # Filter for future known expenses within the next 3 months (using expense date)
        # Could also filter based on ExpenseStatus == 'Unpaid', but date is simpler projection
        future_expenses = expenses_final[
            (expenses_final['Date'].notna()) &
            (expenses_final['Date'] >= today_dt) &
            (expenses_final['Date'] <= future_3m)
        ].copy()
        logger.info(f"Found {len(future_expenses)} expenses dated in forecast period.")

        monthly_outflows = pd.Series(dtype=float) # Ensure initialization
        if not future_expenses.empty:
            future_expenses['ExpenseMonth'] = future_expenses['Date'].dt.to_period('M')
            monthly_outflows = future_expenses.groupby('ExpenseMonth')['Amount'].sum()
        else:
             logger.info("No expenses found dated in the forecast period.")


        # Combine into a forecast dataframe
        start_period = pd.Timestamp(today_dt).to_period('M')
        end_period = pd.Timestamp(future_3m).to_period('M')
        # Handle cases where start/end might be invalid
        if pd.isna(start_period) or pd.isna(end_period):
             raise ValueError("Could not determine valid start/end period for forecast.")

        forecast_index = pd.period_range(start=start_period, end=end_period, freq='M')

        if not forecast_index.empty:
            forecast_df = pd.DataFrame(index=forecast_index)
            # Align series with the forecast index before assigning
            forecast_df['Projected Inflows (AR Due)'] = monthly_inflows.reindex(forecast_index, fill_value=0)
            forecast_df['Projected Outflows (Known Exp.)'] = monthly_outflows.reindex(forecast_index, fill_value=0)
            forecast_df['Net Cash Flow'] = forecast_df['Projected Inflows (AR Due)'] - forecast_df['Projected Outflows (Known Exp.)']
            # Assuming starting cash balance is 0 for this simple projection
            forecast_df['Cumulative Cash Flow'] = forecast_df['Net Cash Flow'].cumsum()
            forecast_df.index = forecast_df.index.strftime('%Y-%m') # Format index for display

            st.dataframe(forecast_df.style.format("${:,.0f}"))

            # Plot Net Cash Flow
            # Convert index back to something plottable if needed, or use the string index
            plot_df_forecast = forecast_df.reset_index().rename(columns={'index': 'Month'})
            fig_forecast = px.bar(plot_df_forecast, x='Month', y='Net Cash Flow', title="Projected Monthly Net Cash Flow (Next 3 Months - Filtered)",
                                labels={'Net Cash Flow': 'Net Cash Flow ($)', 'Month': 'Month'}, text_auto='.0f')
            fig_forecast.update_traces(textposition='outside')
            st.plotly_chart(fig_forecast, use_container_width=True)

            # AI Explanation
            cash_flow_context = (
                f"3-Month Cash Flow Projection Summary (based on filtered data):\n"
                f"- Total Projected Inflows (AR Due): ${forecast_df['Projected Inflows (AR Due)'].sum():,.0f}\n"
                f"- Total Projected Outflows (Known Exp.): ${forecast_df['Projected Outflows (Known Exp.)'].sum():,.0f}\n"
                f"- Overall Projected Net Cash Flow: ${forecast_df['Net Cash Flow'].sum():,.0f}\n"
                f"Monthly Net Flows:\n{forecast_df['Net Cash Flow'].to_string(float_format='${:,.0f}'.format)}\n"
                f"Note: Projection is simplified based only on invoices due and expenses dated in the period."
            )
            render_ai_explanation("AI Explanation of Cash Flow Projection", client, VISUAL_EXPLANATION_PROMPT, cash_flow_context)

        else:
             st.warning("Could not generate forecast period index (start/end dates might be invalid).")
    except Exception as e:
        logger.error(f"Error creating cash flow forecast: {e}", exc_info=True)
        st.error(f"Could not generate cash flow forecast: {e}")

else:
    st.warning("Insufficient invoice or expense data available for the selected filters to create a cash flow forecast.")


st.divider()

# Chat Feature
st.header("💬 Ask Questions About Your Financial Data")
filter_details_dict = {
    'start_date': start_date.strftime('%Y-%m-%d') if start_date else 'N/A',
    'end_date': end_date.strftime('%Y-%m-%d') if end_date else 'N/A',
    'city': city,
    'project_type': project_type
}
render_chat_interface(
    client,
    CHAT_PROMPT_TEMPLATE,
    fin_summary_context,
    filter_details=filter_details_dict,
    page_key="financials", # <--- ADD UNIQUE KEY
    placeholder="Ask about the filtered financial data..."
)