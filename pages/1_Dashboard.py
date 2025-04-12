# pages/1_Dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import logging
from datetime import datetime, timedelta

# Import shared utilities
from utils.data_utils import load_all_data
from utils.filter_utils import render_global_filters, apply_filters, recalculate_costs_after_filtering
from utils.ai_utils import get_genai_client, render_ai_explanation, render_chat_interface, SUMMARY_ADVICE_PROMPT, VISUAL_EXPLANATION_PROMPT, CHAT_PROMPT_TEMPLATE
from config import ( # <--- MODIFY THIS IMPORT SECTION
    CSS_FILE,
    CURRENT_DATE,
    BILLABLE_ROLES,
    HOURS_PER_EMPLOYEE_PER_YEAR,
    DEFAULT_FILTER_START_DATE, # <-- ADD
    DEFAULT_FILTER_END_DATE    # <-- ADD
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Page Setup and Styling ---
st.set_page_config(layout="wide") # Good practice to set on each page too
try:
    with open(CSS_FILE) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("CSS file not found.")

st.title("Executive Dashboard")

# --- AI Client Initialization ---
client = get_genai_client()
if not client:
    st.error("AI Client failed to initialize. AI features will be unavailable.")

# --- Data Loading ---
# Use the cached function from data_utils
data_orig = load_all_data()
if data_orig is None:
    st.error("Fatal error loading data. Dashboard cannot be displayed.")
    st.stop()

# Prepare employee rates (used in recalc) - ensure EmployeeID is index and string
employee_rates = pd.DataFrame()
if "employees" in data_orig and not data_orig["employees"].empty:
     emp_df_rates = data_orig["employees"].copy()
     emp_df_rates['EmployeeID'] = emp_df_rates['EmployeeID'].astype(str)
     employee_rates = emp_df_rates.drop_duplicates(subset=['EmployeeID']).set_index('EmployeeID')[['HourlyRate', 'OvertimeRate']]


# --- Global Filters ---
start_date, end_date, city, project_type = render_global_filters(data_orig)

# --- Apply Filters and Recalculate Costs ---
filtered_data = apply_filters(data_orig, start_date, end_date, city, project_type)
if not filtered_data:
    st.error("Error applying filters. Cannot display filtered data.")
    st.stop()

# IMPORTANT: Recalculate costs based on the *filtered* time entries
filtered_data = recalculate_costs_after_filtering(filtered_data, employee_rates)

# --- Extract Filtered DataFrames ---
# Use .get() with default empty DataFrame for safety
projects_df = filtered_data.get('projects', pd.DataFrame())
service_calls_df = filtered_data.get('service_calls', pd.DataFrame())
expenses_df = filtered_data.get('expenses', pd.DataFrame())
time_entries_df = filtered_data.get('time_entries', pd.DataFrame())
invoices_df = filtered_data.get('invoices', pd.DataFrame())
employees_df = filtered_data.get('employees', pd.DataFrame()) # Usually use original for counts, but filter might affect some analyses
bids_df = filtered_data.get('bids', pd.DataFrame())

# --- Calculate KPIs (Using Filtered & Recalculated Data) ---
total_paid_revenue = invoices_df[invoices_df['Status'] == 'Paid']['Amount'].sum() if not invoices_df.empty else 0
projects_completed = len(projects_df[projects_df["Status"] == "Completed"]) if not projects_df.empty else 0

on_time_rate = 0
if not projects_df.empty and 'DelayImpactHours' in projects_df.columns:
    # Ensure column is numeric
    projects_df['DelayImpactHours'] = pd.to_numeric(projects_df['DelayImpactHours'], errors='coerce').fillna(0)
    on_time_projects = projects_df[(projects_df["DelayImpactHours"] == 0) & (projects_df["Status"] == "Completed")] # Consider only completed
    total_completed_for_rate = len(projects_df[projects_df["Status"] == "Completed"])
    if total_completed_for_rate > 0:
        on_time_rate = len(on_time_projects) / total_completed_for_rate * 100

# Utilization (using filtered time entries and potentially filtered employees if relevant)
utilization = 0
total_hours_logged_filt = time_entries_df['TotalHours'].sum() if not time_entries_df.empty else 0
num_billable_employees = 0
total_capacity_hours = 0

# Base capacity on original employee list unless filtering employees is desired
employees_for_capacity = data_orig.get('employees', pd.DataFrame()) # Use original usually
if not employees_for_capacity.empty and 'Role' in employees_for_capacity.columns:
    employees_billable = employees_for_capacity[employees_for_capacity['Role'].isin(BILLABLE_ROLES)]
    num_billable_employees = len(employees_billable)

    # Calculate analysis duration based on filtered dates if available, else full range
    if start_date and end_date:
        analysis_duration_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    elif not time_entries_df.empty and 'Date' in time_entries_df: # Fallback to filtered time range
        min_date_filt = time_entries_df['Date'].min()
        max_date_filt = time_entries_df['Date'].max()
        if pd.notna(min_date_filt) and pd.notna(max_date_filt):
             analysis_duration_days = (max_date_filt - min_date_filt).days
        else: analysis_duration_days = 365 # Default if range unknown
    else: # Fallback to default filter range or 1 year
         analysis_duration_days = (DEFAULT_FILTER_END_DATE - DEFAULT_FILTER_START_DATE).days

    analysis_duration_years = max(1, analysis_duration_days) / 365.25 # Ensure positive duration, use 1 day min
    total_capacity_hours = num_billable_employees * HOURS_PER_EMPLOYEE_PER_YEAR * analysis_duration_years

    if total_capacity_hours > 0:
        utilization = (total_hours_logged_filt / total_capacity_hours) * 100
        logger.info(f"Utilization calculated: {total_hours_logged_filt:.0f} filtered hrs / {total_capacity_hours:.0f} capacity hrs = {utilization:.1f}%")
    else:
        logger.warning("Could not calculate utilization: Zero capacity hours.")
else:
    logger.warning("Could not calculate utilization: Employees data missing or insufficient.")

# AP/AR (from filtered expenses/invoices)
ap = expenses_df[expenses_df['ExpenseStatus'] == 'Unpaid']['Amount'].sum() if not expenses_df.empty and 'ExpenseStatus' in expenses_df.columns else 0
ar = invoices_df[invoices_df['Status'].isin(['Outstanding', 'Overdue'])]['Amount'].sum() if not invoices_df.empty else 0

# --- Render Dashboard Sections ---

# Summary Section
st.header("📋 Summary")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Filtered Data Counts")
    st.write(f"- Projects: {len(projects_df)}")
    st.write(f"- Service Calls: {len(service_calls_df)}")
    st.write(f"- Expenses: {len(expenses_df)}")
    st.write(f"- Time Entries: {len(time_entries_df)}")
    st.write(f"- Invoices: {len(invoices_df)}")
    st.write(f"- Bids: {len(bids_df)}")
with col2:
    st.subheader("Key Performance Indicators (Filtered)")
    st.metric("Total Paid Revenue", f"${total_paid_revenue:,.0f}")
    st.metric("Projects Completed", f"{projects_completed}")
    st.metric("On-Time Completion Rate (Completed)", f"{on_time_rate:.1f}%")
    st.metric("Technician Utilization (Est.)", f"{min(utilization, 100):.1f}%")
    st.metric("Accounts Receivable (Current)", f"${ar:,.0f}")
    st.metric("Accounts Payable (Unpaid)", f"${ap:,.0f}")

# AI Business Advisor
st.subheader("💡 AI Business Advisor")
summary_context = (
    f"Data Overview (Filtered Period: {start_date.strftime('%Y-%m-%d') if start_date else 'N/A'} to {end_date.strftime('%Y-%m-%d') if end_date else 'N/A'}, City: {city}, Type: {project_type}):\n"
    f"- Projects: {len(projects_df)} records, {projects_completed} completed.\n"
    f"- KPIs: Paid Revenue ${total_paid_revenue:,.0f}, On-Time Rate {on_time_rate:.1f}%, Est. Utilization {min(utilization, 100):.1f}% (based on {num_billable_employees} billable roles), AR ${ar:,.0f}, AP ${ap:,.0f}.\n"
    f"Note: Utilization is an estimate. On-Time Rate based on DelayImpactHours==0 for completed projects."
)
# --- MODIFIED CALL ---
render_ai_explanation(
    "Get High-Level Business Advice",
    client,
    SUMMARY_ADVICE_PROMPT, # The template containing {context} and {current_date}
    summary_context,       # The data context string for {context}
    additional_format_args={'current_date': CURRENT_DATE.strftime('%Y-%m-%d')} # Pass current_date separately
)


st.divider()

# KPI Overview
st.header("📊 KPI Overview")
col_k1, col_k2, col_k3 = st.columns(3)
col_k1.metric("Paid Revenue (Filtered)", f"${total_paid_revenue:,.0f}")
col_k2.metric("On-Time Rate (Completed)", f"{on_time_rate:.1f}%")
col_k3.metric("Utilization (Est.)", f"{min(utilization, 100):.1f}%")

# Project Starts Plot
if not projects_df.empty and 'StartDateStr' in projects_df.columns:
     # Ensure datetime before resampling
     projects_df['StartDateStr'] = pd.to_datetime(projects_df['StartDateStr'], errors='coerce')
     project_starts = projects_df.dropna(subset=['StartDateStr']).set_index('StartDateStr').resample('ME').size().reset_index(name='Project Starts')
     if not project_starts.empty:
        fig_kpi = px.line(project_starts, x='StartDateStr', y='Project Starts', title="Project Starts Per Month (Filtered)", markers=True)
        fig_kpi.update_layout(xaxis_title="Month", yaxis_title="Number of Projects Started")
        st.plotly_chart(fig_kpi, use_container_width=True)
        kpi_plot_context = ( f"Chart shows monthly project starts from {project_starts['StartDateStr'].min().strftime('%Y-%m')} to {project_starts['StartDateStr'].max().strftime('%Y-%m')}. "
                             f"Total projects started in period: {project_starts['Project Starts'].sum()}. Peak month: {project_starts.loc[project_starts['Project Starts'].idxmax()]['StartDateStr'].strftime('%Y-%m')} ({project_starts['Project Starts'].max()} starts)." )
        render_ai_explanation("AI Analysis of Project Starts", client, VISUAL_EXPLANATION_PROMPT, kpi_plot_context)
     else:
          st.warning("No project start data available within the selected filters.")
else:
     st.warning("Project data or StartDateStr column missing for KPI plot.")

st.divider()

# Cash Position
st.header("💰 Cash Position")
st.metric("AR (Outstanding/Overdue)", f"${ar:,.2f}")
st.metric("AP (Unpaid Expenses)", f"${ap:,.2f}")

if ar > 0 or ap > 0:
    cash_pos_df = pd.DataFrame({'Type': ['AR (Current)', 'AP (Unpaid)'], 'Amount': [ar, ap]})
    fig_cash = px.bar(cash_pos_df, x='Type', y='Amount', title="AR vs AP (Filtered)", text_auto=True)
    fig_cash.update_traces(texttemplate='$%{y:,.0f}')
    st.plotly_chart(fig_cash, use_container_width=True)
    cash_context = (f"Cash Position (Filtered): AR (Outstanding/Overdue): ${ar:,.0f}, AP (Unpaid Expenses): ${ap:,.0f}. "
                    f"AR/AP Ratio: {ar / ap:.2f}" if ap > 0 else "AR/AP Ratio: N/A (AP is zero)")
    render_ai_explanation("AI Analysis of Cash Position", client, VISUAL_EXPLANATION_PROMPT, cash_context)
else:
    st.warning("No AR or AP data available within the selected filters.")


st.divider()

# Predict Operational Efficiency - Train on ORIGINAL data, predict based on trend
st.header("🔮 Predict Operational Efficiency Trend")
st.caption("Predicting future On-Time Rate based on historical trends (using DelayImpactHours == 0 on Completed Projects). Model trained on all available data.")

efficiency_predictions = []
model_trained = False
fig_eff = None # Initialize figure variable

# Use original data for training
projects_orig_eff = data_orig.get('projects', pd.DataFrame())
if not projects_orig_eff.empty and 'StartDateStr' in projects_orig_eff.columns and 'DelayImpactHours' in projects_orig_eff.columns and 'Status' in projects_orig_eff.columns:
    try:
        projects_orig_eff["StartDateStr_dt"] = pd.to_datetime(projects_orig_eff["StartDateStr"], errors='coerce')
        projects_orig_eff["DelayImpactHours"] = pd.to_numeric(projects_orig_eff["DelayImpactHours"], errors='coerce')

        eff_data = projects_orig_eff[projects_orig_eff['Status']=='Completed'].dropna(subset=["StartDateStr_dt", "DelayImpactHours"]).copy()

        if not eff_data.empty:
            eff_data["YearMonth"] = eff_data["StartDateStr_dt"].dt.to_period("M")
            eff_df = eff_data.groupby("YearMonth")["DelayImpactHours"].apply(
                lambda x: (x == 0).sum() / len(x) * 100 if len(x) > 0 else 0 # Percentage
            ).reset_index(name="Efficiency")
            eff_df = eff_df.sort_values('YearMonth') # Ensure sorted by time

            if len(eff_df) > 5: # Need some history
                eff_df["TimeIndex"] = np.arange(len(eff_df))
                X = eff_df[["TimeIndex"]]
                y = eff_df["Efficiency"]

                # Simple model for trend prediction (consider ARIMA/Prophet for more serious forecasting)
                from sklearn.linear_model import LinearRegression # Or use GBR as before
                model = LinearRegression()
                model.fit(X, y)
                model_trained = True

                # Predict next 3 months
                last_index = eff_df["TimeIndex"].max()
                future_indices = pd.DataFrame({"TimeIndex": [last_index + i for i in range(1, 4)]})
                predictions = model.predict(future_indices)
                efficiency_predictions = np.clip(predictions, 0, 100) # Clip predictions between 0 and 100%

                st.write("Predicted On-Time Rate Trend (Next 3 Months):")
                for i, pred in enumerate(efficiency_predictions):
                    st.write(f"- Month +{i+1}: {pred:.1f}%")

                # Plot historical and predicted efficiency
                last_period = eff_df["YearMonth"].max()
                future_periods = pd.period_range(start=last_period + 1, periods=3, freq='M')
                future_df = pd.DataFrame({
                     'YearMonth': future_periods, 'Efficiency': efficiency_predictions, 'Type': 'Prediction'
                })
                eff_df['Type'] = 'Historical'
                # Convert Period to Timestamp for plotting
                eff_df['DatePlot'] = eff_df['YearMonth'].dt.to_timestamp()
                future_df['DatePlot'] = future_df['YearMonth'].dt.to_timestamp()

                plot_df = pd.concat([eff_df[['DatePlot', 'Efficiency', 'Type']], future_df], ignore_index=True)

                fig_eff = px.line(plot_df, x='DatePlot', y='Efficiency', color='Type', title="Historical and Predicted On-Time Rate Trend", markers=True)
                fig_eff.update_layout(yaxis_tickformat=".0f", yaxis_title="On-Time Rate (%)", xaxis_title="Month", yaxis_range=[0, 105]) # Set y-axis range
                st.plotly_chart(fig_eff, use_container_width=True)

            else: st.warning("Not enough historical monthly data (Completed Projects) to train prediction model.")
        else: st.warning("No valid 'Completed' project data with dates/delay info for efficiency calculation.")
    except Exception as e:
        logger.error(f"Error during efficiency prediction: {e}", exc_info=True)
        st.error(f"Could not generate efficiency prediction: {e}")
else:
    st.warning("Original Project data missing required columns for efficiency prediction.")

# AI Analysis for Efficiency Prediction
if model_trained and len(efficiency_predictions) > 0:
     eff_context = (
         f"Efficiency Prediction: Based on historical On-Time Rate trend (Completed Projects, DelayImpactHours==0). "
         f"Predicted On-Time Rate for Next 3 Months: {efficiency_predictions[0]:.1f}%, {efficiency_predictions[1]:.1f}%, {efficiency_predictions[2]:.1f}%. "
         f"Current Filtered On-Time Rate (Completed Projects): {on_time_rate:.1f}%. "
         f"Current Filtered Est. Technician Utilization: {min(utilization, 100):.1f}%. "
         f"Note: Prediction is a simple trend extrapolation."
     )
     # Using VISUAL_EXPLANATION_PROMPT for consistency, but could use a more specific one
     render_ai_explanation("AI Analysis of Efficiency Prediction", client, VISUAL_EXPLANATION_PROMPT, eff_context)


st.divider()

# Chat Feature
st.header("💬 Dig Into The Details")
filter_details_dict = {
    'start_date': start_date.strftime('%Y-%m-%d') if start_date else 'N/A',
    'end_date': end_date.strftime('%Y-%m-%d') if end_date else 'N/A',
    'city': city,
    'project_type': project_type
}
render_chat_interface(
    client,
    CHAT_PROMPT_TEMPLATE,
    summary_context,
    filter_details=filter_details_dict,
    page_key="dashboard", # <--- ADD UNIQUE KEY
    placeholder="Ask about the dashboard..."
)