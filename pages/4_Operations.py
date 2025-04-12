# pages/4_Operations.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go # Keep just in case, though not used in final version below
import numpy as np
import logging
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor # For Productivity Prediction
from sklearn.preprocessing import LabelEncoder   # For Productivity Prediction

# Import shared utilities
from utils.data_utils import load_all_data
from utils.filter_utils import render_global_filters, apply_filters, recalculate_costs_after_filtering
from utils.ai_utils import (
    get_genai_client, render_ai_explanation, render_chat_interface,
    SUMMARY_ADVICE_PROMPT, VISUAL_EXPLANATION_PROMPT, CHAT_PROMPT_TEMPLATE
)
from config import (
    CSS_FILE, CURRENT_DATE, BILLABLE_ROLES, HOURS_PER_EMPLOYEE_PER_YEAR,
    DEFAULT_FILTER_START_DATE, DEFAULT_FILTER_END_DATE # Import date defaults for utilization calc fallback
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Custom Prompt for Resource Allocation (Specific to this page) ---
# Keep this custom prompt as it fulfills a unique requirement
RESOURCE_ALLOCATION_PROMPT = """
You are an Operations Director with 25+ years of experience optimizing technician deployments for luxury residential low voltage contractors in the Seattle and Greater Eastside markets.

CONTEXT:
{context}

TASK:
Provide optimal technician assignment strategies that balance workload, skill matching, and service quality for this luxury residential contractor. Include both strategic principles and a practical example based on the context.

REQUIREMENTS:
- Begin with 2-3 sentences on resource optimization principles for high-end residential services.
- Provide 3 specific resource allocation strategies with expected operational benefits, grounded in the context provided.
- Include a sample technician assignment table showing optimal matching of skills to common service types mentioned or implied in the context.
- Base all recommendations strictly on the provided context data.

RESPONSE FORMAT:
[2-3 sentences on resource optimization principles]

Strategic Allocation Approaches:
1. [Strategy 1 with expected operational benefit based on context]
2. [Strategy 2 with expected operational benefit based on context]
3. [Strategy 3 with expected operational benefit based on context]

Sample Optimal Assignment Matrix (Based on Context):
| Service Type        | Recommended Technician Profile | Key Benefit Rationale              |
| :------------------ | :--------------------------- | :--------------------------------- |
| [Common Service Type 1 from Context] | [Optimal profile based on context] | [Specific benefit, e.g., FTF rate] |
| [Common Service Type 2 from Context] | [Optimal profile based on context] | [Specific benefit, e.g., Speed]    |
| [Common Service Type 3 from Context] | [Optimal profile based on context] | [Specific benefit, e.g., Upsell]   |
"""

# --- Page Setup and Styling ---
st.set_page_config(layout="wide")
try:
    with open(CSS_FILE) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    logger.info("CSS loaded successfully.")
except FileNotFoundError:
    st.warning("CSS file not found. Using default styles.")

st.title("Operations Dashboard")

# --- AI Client Initialization ---
client = get_genai_client()
if not client:
    st.error("AI Client failed to initialize. AI features will be unavailable.")

# --- Data Loading ---
data_orig = load_all_data()
if data_orig is None:
    st.error("Fatal error loading data. Dashboard cannot be displayed.")
    st.stop()

# Prepare employee rates (standard pattern)
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

# Recalculate costs based on filtered time entries (standard step)
filtered_data = recalculate_costs_after_filtering(filtered_data, employee_rates)

# --- Extract Filtered DataFrames for this Page ---
service_calls_final = filtered_data.get('service_calls', pd.DataFrame())
employees_final = filtered_data.get('employees', pd.DataFrame())
inventory_final = filtered_data.get('inventory', pd.DataFrame())
equipment_final = filtered_data.get('equipment', pd.DataFrame()) # Added equipment
time_entries_final = filtered_data.get('time_entries', pd.DataFrame())
# customers_final = filtered_data.get('customers', pd.DataFrame()) # Uncomment if needed

# --- Calculate Operational KPIs (Using Filtered Data) ---
total_labor_hours = 0
avg_labor_hours_per_call = 0
first_time_fix_rate = 0
items_below_reorder = 0
technician_utilization = 0

# Calculate Total Labor Hours from filtered time entries
if not time_entries_final.empty and 'TotalHours' in time_entries_final.columns:
    time_entries_final['TotalHours'] = pd.to_numeric(time_entries_final['TotalHours'], errors='coerce').fillna(0)
    total_labor_hours = time_entries_final['TotalHours'].sum()

# Calculate Avg Labor Hours per Service Call from filtered service calls
if not service_calls_final.empty and 'LaborHours' in service_calls_final.columns:
    service_calls_final['LaborHours'] = pd.to_numeric(service_calls_final['LaborHours'], errors='coerce')
    avg_labor_hours_per_call = service_calls_final['LaborHours'].mean()
    if pd.isna(avg_labor_hours_per_call): avg_labor_hours_per_call = 0

# Calculate First-Time Fix Rate from filtered service calls
if not service_calls_final.empty and 'FirstTimeFixStatus' in service_calls_final.columns:
    ftf_yes = service_calls_final['FirstTimeFixStatus'] == 'Yes'
    if len(service_calls_final) > 0:
        first_time_fix_rate = (ftf_yes.sum() / len(service_calls_final)) * 100
    else:
        first_time_fix_rate = 0

# Calculate Low Inventory Items from filtered inventory
if not inventory_final.empty and 'CurrentStock' in inventory_final.columns and 'ReorderLevel' in inventory_final.columns:
    inventory_final['CurrentStock'] = pd.to_numeric(inventory_final['CurrentStock'], errors='coerce').fillna(0)
    inventory_final['ReorderLevel'] = pd.to_numeric(inventory_final['ReorderLevel'], errors='coerce').fillna(0)
    items_below_reorder = (inventory_final['CurrentStock'] <= inventory_final['ReorderLevel']).sum()

# Calculate Technician Utilization (similar to Dashboard)
num_billable_employees = 0
total_capacity_hours = 0
# Use original employee list for capacity calculation unless filtering employees is desired
employees_for_capacity = data_orig.get('employees', pd.DataFrame())
if not employees_for_capacity.empty and 'Role' in employees_for_capacity.columns:
    employees_billable = employees_for_capacity[employees_for_capacity['Role'].isin(BILLABLE_ROLES)]
    num_billable_employees = len(employees_billable)

    # Calculate analysis duration based on filtered dates if available
    if start_date and end_date:
        analysis_duration_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    else: # Fallback to default filter range or 1 year
         analysis_duration_days = (DEFAULT_FILTER_END_DATE - DEFAULT_FILTER_START_DATE).days

    analysis_duration_years = max(1, analysis_duration_days) / 365.25 # Min 1 day duration
    total_capacity_hours = num_billable_employees * HOURS_PER_EMPLOYEE_PER_YEAR * analysis_duration_years

    if total_capacity_hours > 0:
        technician_utilization = (total_labor_hours / total_capacity_hours) * 100
        logger.info(f"Ops Util calc: {total_labor_hours:.1f} filt hrs / {total_capacity_hours:.1f} capacity hrs = {technician_utilization:.1f}%")
    else:
        logger.warning("Could not calculate ops utilization: Zero capacity hours.")
else:
    logger.warning("Could not calculate ops utilization: Employees data missing or roles insufficient.")
technician_utilization = min(technician_utilization, 100) # Cap at 100%


# --- Render Operations Sections ---

# Operations Summary
st.header("📋 Operations Summary")
st.markdown("Key operational metrics based on the selected global filters.")
col1, col2, col3 = st.columns(3)
col1.metric("Total Labor Hours Logged", f"{total_labor_hours:,.1f}")
col1.metric("First-Time Fix Rate", f"{first_time_fix_rate:.1f}%")
col2.metric("Avg. Labor Hours / Call", f"{avg_labor_hours_per_call:.1f}")
col2.metric("Items Below Reorder Level", f"{items_below_reorder}")
col3.metric("Technician Utilization (Est.)", f"{technician_utilization:.1f}%")
# Add more relevant KPIs if needed

# AI Operations Advisor
st.subheader("💡 AI Operations Advisor")
ops_summary_context = (
    f"Operational Snapshot (Filtered Period: {start_date.strftime('%Y-%m-%d') if start_date else 'N/A'} to {end_date.strftime('%Y-%m-%d') if end_date else 'N/A'}, City: {city}, Type: {project_type}):\n"
    f"- Service Calls Analyzed: {len(service_calls_final)}\n"
    f"- Total Labor Hours Logged (Filtered): {total_labor_hours:,.1f}\n"
    f"- Average Labor Hours per Service Call: {avg_labor_hours_per_call:.1f}\n"
    f"- First-Time Fix Rate: {first_time_fix_rate:.1f}%\n"
    f"- Inventory Items Below Reorder Level: {items_below_reorder}\n"
    f"- Estimated Technician Utilization: {technician_utilization:.1f}% (based on {num_billable_employees} billable roles over {analysis_duration_days} days)\n"
)
render_ai_explanation(
    "Get AI-Powered Operational Advice",
    client,
    SUMMARY_ADVICE_PROMPT, # Using the general advice prompt
    ops_summary_context,
    additional_format_args={'current_date': CURRENT_DATE.strftime('%Y-%m-%d')}
)

st.divider()

# --- Section: Job Schedule ---
st.header("📅 Job Schedule Analysis")
st.markdown("Timeline of service calls for technicians based on the selected global filters.")

schedule_df = pd.DataFrame()
conflicts_count = 0
if not service_calls_final.empty and not employees_final.empty:
    try:
        # Prepare data for timeline
        schedule_df = service_calls_final.copy()
        # Convert time strings to timedeltas, then add to DateStr for proper datetime
        schedule_df['Start'] = pd.to_datetime(schedule_df['DateStr'], errors='coerce') + \
                               pd.to_timedelta(schedule_df['StartTime'] + ':00', errors='coerce') # Add seconds for parsing
        schedule_df['End'] = pd.to_datetime(schedule_df['DateStr'], errors='coerce') + \
                             pd.to_timedelta(schedule_df['EndTime'] + ':00', errors='coerce') # Add seconds for parsing

        # Map Technician Names (ensure EmployeeID types match)
        employees_final['EmployeeID'] = employees_final['EmployeeID'].astype(str)
        schedule_df['TechnicianID'] = schedule_df['TechnicianID'].astype(str)
        tech_name_map = employees_final.set_index('EmployeeID')['Name']
        schedule_df['TechnicianName'] = schedule_df['TechnicianID'].map(tech_name_map).fillna('Unknown Tech')

        # Filter out entries where start or end time couldn't be parsed
        schedule_df = schedule_df.dropna(subset=['Start', 'End'])

        if not schedule_df.empty:
            fig_schedule = px.timeline(
                schedule_df,
                x_start="Start",
                x_end="End",
                y="TechnicianName",
                color="Priority", # Color by priority
                hover_data=['ServiceID', 'ServiceType', 'IssueCategory'],
                title="Service Call Schedule by Technician"
            )
            fig_schedule.update_yaxes(categoryorder="total ascending") # Sort techs by activity
            st.plotly_chart(fig_schedule, use_container_width=True)

            # Basic conflict detection (simple check per technician)
            def detect_overlap(group):
                group = group.sort_values('Start')
                return (group['Start'] < group['End'].shift()).any()

            conflicts = schedule_df.groupby('TechnicianName').filter(lambda x: len(x) > 1).groupby('TechnicianName').apply(detect_overlap)
            conflicts_count = conflicts.sum()
            if conflicts_count > 0:
                st.warning(f"Detected potential scheduling conflicts/overlaps for {conflicts_count} technician(s) in this view.")

        else:
            st.info("No valid schedule data to display for the selected filters (check start/end times).")

    except Exception as e:
        logger.error(f"Error generating job schedule: {e}", exc_info=True)
        st.error("Could not generate the job schedule visualization.")

else:
    st.warning("Service call or employee data is empty for the selected filters. Cannot display schedule.")

# AI Explanation for Schedule
if not schedule_df.empty:
    avg_duration_hours = (schedule_df['End'] - schedule_df['Start']).mean().total_seconds() / 3600 if not schedule_df.empty else 0
    schedule_context = (
        f"Service call schedule timeline for {len(schedule_df)} calls across {schedule_df['TechnicianName'].nunique()} technicians.\n"
        f"- Priority Distribution: {schedule_df['Priority'].value_counts().to_dict()}\n"
        f"- Average Call Duration (calculated): {avg_duration_hours:.1f} hours.\n"
        f"- Potential Scheduling Conflicts Detected: {conflicts_count}\n"
        f"Note: Visualization based on parsed start/end times from filtered data."
    )
    render_ai_explanation("AI Analysis of Job Schedule", client, VISUAL_EXPLANATION_PROMPT, schedule_context)

st.divider()

# --- Section: Technician Productivity ---
st.header("📊 Technician Productivity")
st.markdown("Analysis of hours logged and estimated utilization per technician based on filtered data.")

tech_prod_final = pd.DataFrame()
if not time_entries_final.empty and not employees_final.empty:
    try:
        # Aggregate hours from filtered time entries
        tech_hours_agg = time_entries_final.groupby('EmployeeID')['TotalHours'].sum().reset_index()

        # Merge with filtered employee data (ensure IDs are strings)
        employees_final['EmployeeID'] = employees_final['EmployeeID'].astype(str)
        tech_hours_agg['EmployeeID'] = tech_hours_agg['EmployeeID'].astype(str)
        tech_prod_final = tech_hours_agg.merge(employees_final, on='EmployeeID', how='inner') # Inner join to keep only relevant techs

        # Calculate Utilization (using same method as summary KPI for consistency)
        tech_prod_final['UtilizationRate'] = 0.0 # Initialize
        if total_capacity_hours > 0 and num_billable_employees > 0: # Use overall capacity for rate calc consistency
             # Calculate capacity per *individual* employee over the period
             individual_capacity_hours = HOURS_PER_EMPLOYEE_PER_YEAR * analysis_duration_years
             if individual_capacity_hours > 0:
                  # Calculate utilization only for billable roles
                  billable_mask = tech_prod_final['Role'].isin(BILLABLE_ROLES)
                  tech_prod_final.loc[billable_mask, 'UtilizationRate'] = \
                      (tech_prod_final['TotalHours'] / individual_capacity_hours) * 100
                  tech_prod_final['UtilizationRate'] = tech_prod_final['UtilizationRate'].clip(upper=150) # Cap display, but check data if > 100

        # Add Certification Count
        tech_prod_final['NumCertifications'] = tech_prod_final['Certifications'].apply(
            lambda x: len(x.split('|')) if isinstance(x, str) and x else 0
        )

        logger.info(f"Calculated productivity for {len(tech_prod_final)} technicians.")

    except Exception as e:
        logger.error(f"Error calculating technician productivity: {e}", exc_info=True)
        st.error("Could not calculate technician productivity.")
        tech_prod_final = pd.DataFrame()

if tech_prod_final.empty:
    st.warning("No technician productivity data available for the selected filters.")
else:
    # Sort by Total Hours for better visualization
    tech_prod_final = tech_prod_final.sort_values('TotalHours', ascending=False)

    fig_prod = px.bar(
        tech_prod_final,
        x="Name",
        y="TotalHours",
        color="Role",
        title="Technician Hours Logged (Filtered Period)",
        hover_data=['Role', 'NumCertifications', 'UtilizationRate'],
        labels={'Name': 'Technician', 'TotalHours': 'Total Hours Logged'}
    )
    # Add utilization rate as text on bars (optional)
    # fig_prod.update_traces(texttemplate='%{customdata[2]:.1f}%', textposition='outside', customdata=tech_prod_final[['Role', 'NumCertifications', 'UtilizationRate']].values)
    st.plotly_chart(fig_prod, use_container_width=True)

    # AI Explanation for Productivity
    prod_context = (
        f"Technician productivity analysis for {len(tech_prod_final)} technicians based on filtered time entries.\n"
        f"- Total Hours Logged Range: {tech_prod_final['TotalHours'].min():.1f} to {tech_prod_final['TotalHours'].max():.1f}\n"
        f"- Average Hours Logged: {tech_prod_final['TotalHours'].mean():.1f}\n"
        f"- Roles included: {', '.join(tech_prod_final['Role'].unique())}\n"
        f"- Average Estimated Utilization (Billable Roles): {tech_prod_final[tech_prod_final['Role'].isin(BILLABLE_ROLES)]['UtilizationRate'].mean():.1f}%\n"
    )
    render_ai_explanation("AI Analysis of Technician Productivity", client, VISUAL_EXPLANATION_PROMPT, prod_context)

st.divider()

# --- Section: Inventory Management ---
st.header("📦 Inventory Management")
st.markdown("Overview of inventory levels and items needing attention based on filtered data.")

inventory_analysis_df = pd.DataFrame()
critical_inventory_df = pd.DataFrame()
total_restock_cost = 0

if not inventory_final.empty:
    try:
        inventory_analysis_df = inventory_final.copy()
        # Ensure numeric types
        num_cols = ['CurrentStock', 'ReorderLevel', 'UnitCost']
        for col in num_cols:
            inventory_analysis_df[col] = pd.to_numeric(inventory_analysis_df[col], errors='coerce').fillna(0)

        # Determine Stock Status
        inventory_analysis_df['StockStatus'] = np.where(
            inventory_analysis_df['CurrentStock'] <= inventory_analysis_df['ReorderLevel'],
            "Below Reorder", "Adequate"
        )

        # Calculate Potential Restock Cost for items below reorder level
        inventory_analysis_df['RestockCost'] = np.where(
            inventory_analysis_df['StockStatus'] == 'Below Reorder',
            (inventory_analysis_df['ReorderLevel'] - inventory_analysis_df['CurrentStock']) * inventory_analysis_df['UnitCost'],
            0
        )
        # Ensure restock cost is non-negative
        inventory_analysis_df['RestockCost'] = inventory_analysis_df['RestockCost'].clip(lower=0)


        critical_inventory_df = inventory_analysis_df[inventory_analysis_df['StockStatus'] == "Below Reorder"].copy()
        total_restock_cost = critical_inventory_df['RestockCost'].sum()
        logger.info(f"Inventory Analysis: {len(critical_inventory_df)} items below reorder level. Total restock cost: ${total_restock_cost:,.2f}")

    except Exception as e:
        logger.error(f"Error analyzing inventory data: {e}", exc_info=True)
        st.error("Could not analyze inventory data.")
        inventory_analysis_df = pd.DataFrame()
        critical_inventory_df = pd.DataFrame()

if critical_inventory_df.empty:
    st.info("No inventory items are currently below reorder level based on the selected filters.")
    # Optionally show a table of all inventory if needed: st.dataframe(inventory_analysis_df)
else:
    st.warning(f"Found {len(critical_inventory_df)} items below reorder level. Estimated restock cost: ${total_restock_cost:,.2f}")
    # Sort by restock cost for visualization
    critical_inventory_df = critical_inventory_df.sort_values('RestockCost', ascending=False)

    fig_inventory = px.bar(
        critical_inventory_df.head(20), # Limit displayed items for clarity
        x="Description",
        y="CurrentStock",
        color="Category",
        title="Top 20 Critical Inventory Items (Below Reorder Level)",
        hover_data=['Manufacturer', 'ReorderLevel', 'UnitCost', 'RestockCost'],
        labels={'Description': 'Item Description', 'CurrentStock': 'Current Stock Level'}
    )
    fig_inventory.update_traces(texttemplate='$%{customdata[3]:,.0f}', textposition='outside', customdata=critical_inventory_df.head(20)[['Manufacturer', 'ReorderLevel', 'UnitCost', 'RestockCost']].values)
    fig_inventory.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_inventory, use_container_width=True)

    # AI Explanation for Inventory
    most_critical_cat = critical_inventory_df['Category'].mode()[0] if not critical_inventory_df.empty else "N/A"
    inventory_context = (
        f"Inventory analysis based on {len(inventory_analysis_df)} filtered items.\n"
        f"- Items Below Reorder Level: {len(critical_inventory_df)}\n"
        f"- Estimated Total Restock Cost (for critical items): ${total_restock_cost:,.2f}\n"
        f"- Category with most critical items: {most_critical_cat}\n"
        f"Chart shows the top 20 items needing attention by estimated restock cost."
    )
    render_ai_explanation("AI Analysis of Critical Inventory", client, VISUAL_EXPLANATION_PROMPT, inventory_context)

st.divider()

# --- Section: Predict Technician Productivity ---
st.header("🔮 Predict Technician Productivity (Annual Hours)")
st.markdown("Estimate annual hours for a technician profile based on historical data (trained on all time entries).")

# --- Model Training (using original data, cached resource) ---
@st.cache_resource
def train_productivity_model():
    """Loads data, preprocesses, and trains the productivity model."""
    logger.info("Attempting to train technician productivity model...")
    data_for_train = load_all_data()
    if data_for_train is None or 'time_entries' not in data_for_train or 'employees' not in data_for_train:
        logger.error("Cannot train productivity model: Time entry or employee data unavailable.")
        return None, None # Return None for model and encoder

    time_entries_orig = data_for_train['time_entries'].copy()
    employees_orig = data_for_train['employees'].copy()

    if time_entries_orig.empty or employees_orig.empty:
        logger.warning("Time entry or employee data is empty. Cannot train productivity model.")
        return None, None

    # --- Preprocessing ---
    required_cols_emp = ['EmployeeID', 'Role', 'Certifications', 'HireDate']
    required_cols_time = ['EmployeeID', 'TotalHours', 'Date']
    if not all(col in employees_orig.columns for col in required_cols_emp) or \
       not all(col in time_entries_orig.columns for col in required_cols_time):
        logger.error("Cannot train productivity model: Missing required columns.")
        return None, None

    # Aggregate total hours per employee
    time_entries_orig['TotalHours'] = pd.to_numeric(time_entries_orig['TotalHours'], errors='coerce').fillna(0)
    # Determine time range for aggregation (e.g., per year)
    time_entries_orig['Date'] = pd.to_datetime(time_entries_orig['Date'], errors='coerce')
    time_entries_orig = time_entries_orig.dropna(subset=['Date'])
    if time_entries_orig.empty:
        logger.warning("No valid time entries with dates found for productivity model.")
        return None, None
    time_entries_orig['Year'] = time_entries_orig['Date'].dt.year
    # Aggregate hours per employee *per year*
    tech_hours_agg = time_entries_orig.groupby(['EmployeeID', 'Year'])['TotalHours'].sum().reset_index()
    # Use the most recent full year or average? Let's average annual hours for simplicity here.
    tech_avg_annual_hours = tech_hours_agg.groupby('EmployeeID')['TotalHours'].mean().reset_index()
    tech_avg_annual_hours.rename(columns={'TotalHours': 'AvgAnnualHours'}, inplace=True)


    # Merge with employee data
    employees_orig['EmployeeID'] = employees_orig['EmployeeID'].astype(str)
    tech_avg_annual_hours['EmployeeID'] = tech_avg_annual_hours['EmployeeID'].astype(str)
    df_ml = tech_avg_annual_hours.merge(employees_orig, on='EmployeeID', how='inner')

    # Feature Engineering
    df_ml['NumCertifications'] = df_ml['Certifications'].apply(lambda x: len(x.split('|')) if isinstance(x, str) and x else 0)
    # Calculate Years of Service (using CURRENT_DATE from config)
    df_ml['HireDate'] = pd.to_datetime(df_ml['HireDate'], errors='coerce')
    df_ml['YearsOfService'] = (CURRENT_DATE - df_ml['HireDate']).dt.days / 365.25
    df_ml['YearsOfService'] = df_ml['YearsOfService'].fillna(0).clip(lower=0) # Handle NaNs and negative values

    # Encode Role
    df_ml['Role'] = df_ml['Role'].astype(str).fillna('Unknown')
    le_role = LabelEncoder()
    df_ml['RoleEncoded'] = le_role.fit_transform(df_ml['Role'])

    # Define features and target
    features = ['NumCertifications', 'RoleEncoded', 'YearsOfService']
    target = 'AvgAnnualHours'

    # Drop rows with NaN target after merge/calculation
    df_ml = df_ml.dropna(subset=[target])

    if df_ml.empty or len(df_ml) < 10:
        logger.warning(f"Insufficient valid data ({len(df_ml)} rows) after preprocessing for productivity model.")
        return None, None

    X = df_ml[features]
    y = df_ml[target]

    # Final check for NaNs/Infs
    if X.isnull().values.any() or y.isnull().values.any() or np.isinf(X.values).any() or np.isinf(y.values).any():
        logger.error("NaN or Inf values detected in final training data (X or y) for productivity model.")
        return None, None

    # --- Training ---
    try:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf=3)
        model.fit(X, y)
        logger.info(f"Productivity prediction model trained successfully with features: {features}")
        return model, le_role # Return model and role encoder
    except Exception as e:
        logger.error(f"Error training productivity model: {e}", exc_info=True)
        return None, None

# Train the model (runs once and caches)
productivity_model, le_role_encoder = train_productivity_model()

# --- Prediction Form ---
if productivity_model is None or le_role_encoder is None:
    st.warning("Technician productivity prediction model is unavailable due to training issues.")
else:
    st.markdown("**Enter Technician Profile:**")
    # Initialize session state
    if 'productivity_prediction_result' not in st.session_state:
        st.session_state.productivity_prediction_result = None
    if 'productivity_prediction_context' not in st.session_state:
        st.session_state.productivity_prediction_context = None
    if 'productivity_prediction_inputs' not in st.session_state:
        st.session_state.productivity_prediction_inputs = {
            'NumCertifications': 3,
            'YearsOfService': 5.0,
            'Role': le_role_encoder.classes_[0] if len(le_role_encoder.classes_) > 0 else 'Unknown'
        }

    with st.form("productivity_prediction_form"):
        role_options = sorted(list(le_role_encoder.classes_))
        default_role = st.session_state.productivity_prediction_inputs.get('Role', role_options[0] if role_options else 'Unknown')
        default_role_index = role_options.index(default_role) if default_role in role_options else 0

        pred_num_certs = st.number_input("Number of Certifications", min_value=0, max_value=15, value=int(st.session_state.productivity_prediction_inputs.get('NumCertifications', 3)), step=1)
        pred_years_service = st.number_input("Years of Service", min_value=0.0, max_value=40.0, value=float(st.session_state.productivity_prediction_inputs.get('YearsOfService', 5.0)), step=0.5)
        pred_role = st.selectbox("Technician Role", options=role_options, index=default_role_index)

        submitted = st.form_submit_button("Predict Annual Hours")

        if submitted:
            logger.info("Productivity prediction form submitted.")
            # Store inputs
            st.session_state.productivity_prediction_inputs = {
                 'NumCertifications': pred_num_certs,
                 'YearsOfService': pred_years_service,
                 'Role': pred_role
            }

            try:
                encoded_role = le_role_encoder.transform([pred_role])[0]
                input_data = pd.DataFrame({
                    "NumCertifications": [pred_num_certs],
                    "RoleEncoded": [encoded_role],
                    "YearsOfService": [pred_years_service]
                })
                # Ensure correct feature order
                input_data = input_data[['NumCertifications', 'RoleEncoded', 'YearsOfService']]

                prediction = productivity_model.predict(input_data)[0]
                prediction = max(0, prediction) # Ensure non-negative prediction

                # --- Store results (CORRECTED CONTEXT STRING) ---
                st.session_state.productivity_prediction_result = prediction
                st.session_state.productivity_prediction_context = (
                    "INPUTS USED FOR PREDICTION:\n"
                    f"- Number of Certifications: {pred_num_certs}\n"
                    f"- Years of Service: {pred_years_service:.1f}\n"
                    f"- Technician Role: {pred_role}\n\n"
                    f"PREDICTED AVERAGE ANNUAL HOURS: {prediction:,.0f}\n\n"
                    # --- REMOVED THE LINE CAUSING THE KeyError ---
                    # f"(Historical Avg Annual Hours for {pred_role} (all data): {data_orig['employees'][data_orig['employees']['Role'] == pred_role]['AvgAnnualHours'].mean():,.0f} - requires precalculating this average)\n"
                    "Note: Prediction based on historical averages derived from time entries. Actual hours vary based on workload, project complexity, and efficiency."
                )
                # --- END CORRECTION ---
                logger.info(f"Productivity prediction successful: {prediction:,.0f} hours")

            except Exception as e:
                st.error(f"Error during productivity prediction calculation: {e}")
                logger.error("Productivity prediction failed.", exc_info=True)
                st.session_state.productivity_prediction_result = None
                st.session_state.productivity_prediction_context = None
            # Rerun implicitly

    # --- Display results OUTSIDE the form ---
    if st.session_state.productivity_prediction_result is not None:
        st.success(f"**Predicted Average Annual Hours: {st.session_state.productivity_prediction_result:,.0f}**")
        if st.session_state.productivity_prediction_context and client:
             render_ai_explanation(
                 "AI Explanation of Productivity Prediction",
                 client,
                 VISUAL_EXPLANATION_PROMPT, # Can use generic
                 st.session_state.productivity_prediction_context
             )
        elif not client:
             st.warning("AI Client unavailable, cannot generate explanation.")
        if st.button("Clear Prediction Result", key="clear_prod_pred"):
             st.session_state.productivity_prediction_result = None
             st.session_state.productivity_prediction_context = None
             st.rerun()

st.divider()

# --- Section: Resource Allocation Optimization ---
st.header("⚙️ Resource Allocation Optimization")
st.markdown("AI-driven suggestions for optimizing technician assignments based on filtered operational data.")

# Construct context for the custom Resource Allocation prompt
try:
    # Get service types and roles from the *filtered* data for context relevance
    common_service_types = service_calls_final['ServiceType'].value_counts().head(5).index.tolist() if not service_calls_final.empty else ['N/A']
    available_roles = employees_final['Role'].unique().tolist() if not employees_final.empty else ['N/A']
    # Calculate avg certs per role from filtered data
    avg_certs_per_role = tech_prod_final.groupby('Role')['NumCertifications'].mean().to_dict() if not tech_prod_final.empty else {}
    avg_certs_str = ", ".join([f"{role}: {avg:.1f}" for role, avg in avg_certs_per_role.items()])

    resource_context = (
        f"Operational Context (Filtered):\n"
        f"- Service Calls: {len(service_calls_final)}. Common Types: {', '.join(common_service_types)}.\n"
        f"- Technicians Available: {len(employees_final)}. Roles: {', '.join(available_roles)}.\n"
        f"- Key KPIs: FTF Rate {first_time_fix_rate:.1f}%, Est. Utilization {technician_utilization:.1f}%.\n"
        f"- Avg Certifications per Role: {avg_certs_str if avg_certs_str else 'N/A'}.\n"
    )
    render_ai_explanation(
        "Get AI Resource Allocation Strategies",
        client,
        RESOURCE_ALLOCATION_PROMPT, # Use the custom prompt defined at the top
        resource_context
    )
except Exception as e:
     logger.error(f"Error preparing context or calling AI for resource allocation: {e}", exc_info=True)
     st.error("Could not generate resource allocation suggestions due to an error.")


st.divider()

# --- Section: Chat Feature ---
st.header("💬 Dig Into The Details") # Standardized Header

# Add suggestions for the Operations page - THESE WILL BE DISPLAYED BY ai_utils.py
# st.markdown("""
# Try asking:
# - "How can we improve our first-time fix rate for 'Network Troubleshooting' calls?"
# - "Which technicians have the highest utilization but lowest satisfaction scores?"
# - "What's the optimal reorder strategy for high-cost inventory items?"
# """) # Commented out as ai_utils handles this

# Prepare context for the chat (reuse ops_summary_context if available)
if 'ops_summary_context' not in locals():
     ops_summary_context = f"Operational data analysis for {len(service_calls_final)} service calls and {len(employees_final)} employees."
     logger.warning("Using fallback chat context as ops_summary_context was not defined.")

chat_base_context = ops_summary_context # Use the determined context

# Prepare the dictionary with current filter values
filter_details_dict = {
    'start_date': start_date.strftime('%Y-%m-%d') if start_date else 'N/A',
    'end_date': end_date.strftime('%Y-%m-%d') if end_date else 'N/A',
    'city': city,
    'project_type': project_type # Key name used in filter_utils and expected by template
}

# Render the standard chat interface - this function will handle displaying suggestions
render_chat_interface(
    client=client,
    chat_prompt_template=CHAT_PROMPT_TEMPLATE,
    base_context=chat_base_context,
    filter_details=filter_details_dict,
    page_key="operations", # Unique key for this page
    placeholder="Ask about schedules, productivity, inventory, or resources..."
)