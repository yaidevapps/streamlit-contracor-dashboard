# pages/6_Employee.py
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
from config import ( # Import necessary config items
    CSS_FILE, CURRENT_DATE, BILLABLE_ROLES, HOURS_PER_EMPLOYEE_PER_YEAR,
    DEFAULT_FILTER_START_DATE, DEFAULT_FILTER_END_DATE
)

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

st.title("Employee Dashboard")

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
employees_final = filtered_data.get('employees', pd.DataFrame())
service_calls_final = filtered_data.get('service_calls', pd.DataFrame()) # For avg labor hours calc
time_entries_final = filtered_data.get('time_entries', pd.DataFrame())

# --- Calculate Employee KPIs (Using Filtered Data) ---
total_certs = 0
avg_labor_hours_per_call = 0
technician_utilization = 0
expiring_certs_count = 0
total_employees_filtered = len(employees_final)

# Total Certifications (on filtered employees)
if not employees_final.empty and 'Certifications' in employees_final.columns:
    employees_final['NumCertifications'] = employees_final['Certifications'].apply(lambda x: len(x.split('|')) if isinstance(x, str) and x else 0)
    total_certs = employees_final['NumCertifications'].sum()

# Average Labor Hours per Call (using filtered service calls)
if not service_calls_final.empty and 'LaborHours' in service_calls_final.columns:
    service_calls_final['LaborHours'] = pd.to_numeric(service_calls_final['LaborHours'], errors='coerce')
    avg_labor_hours_per_call = service_calls_final['LaborHours'].mean()
    if pd.isna(avg_labor_hours_per_call): avg_labor_hours_per_call = 0

# Technician Utilization (using filtered time entries and config)
total_hours_logged_filt = 0
if not time_entries_final.empty and 'TotalHours' in time_entries_final.columns:
     time_entries_final['TotalHours'] = pd.to_numeric(time_entries_final['TotalHours'], errors='coerce').fillna(0)
     total_hours_logged_filt = time_entries_final['TotalHours'].sum()

num_billable_employees = 0
total_capacity_hours = 0
# Use original employee list for capacity unless filtering employees is desired
employees_for_capacity = data_orig.get('employees', pd.DataFrame())
if not employees_for_capacity.empty and 'Role' in employees_for_capacity.columns:
    employees_billable = employees_for_capacity[employees_for_capacity['Role'].isin(BILLABLE_ROLES)]
    num_billable_employees = len(employees_billable)

    # Calculate analysis duration from filters
    if start_date and end_date:
        analysis_duration_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    else: # Fallback
         analysis_duration_days = (DEFAULT_FILTER_END_DATE - DEFAULT_FILTER_START_DATE).days

    analysis_duration_years = max(1, analysis_duration_days) / 365.25
    total_capacity_hours = num_billable_employees * HOURS_PER_EMPLOYEE_PER_YEAR * analysis_duration_years

    if total_capacity_hours > 0:
        technician_utilization = (total_hours_logged_filt / total_capacity_hours) * 100
    else:
        logger.warning("Could not calculate employee utilization: Zero capacity hours.")
else:
    logger.warning("Could not calculate employee utilization: Employees data missing or insufficient.")
technician_utilization = min(technician_utilization, 100)

# Certifications Expiring Soon (relative to CURRENT_DATE)
if not employees_final.empty and 'LicenseExpirations' in employees_final.columns:
    expiry_date_threshold = CURRENT_DATE + timedelta(days=365) # Example: expiring within next year
    employees_final['LicenseExpirations'] = pd.to_datetime(employees_final['LicenseExpirations'], errors='coerce')
    expiring_certs_count = (employees_final['LicenseExpirations'] < expiry_date_threshold).sum()


# --- Render Employee Sections ---

# Employee Summary
st.header("📋 Employee Summary")
st.markdown("Key workforce metrics based on the selected global filters.")
col1, col2, col3 = st.columns(3)
col1.metric("Total Employees (Filtered)", total_employees_filtered)
col1.metric("Total Certifications", f"{total_certs:,}")
col2.metric("Avg. Labor Hours / Service Call", f"{avg_labor_hours_per_call:.1f}")
col2.metric("Certifications Expiring (Next Year)", expiring_certs_count)
col3.metric("Technician Utilization (Est.)", f"{technician_utilization:.1f}%")


# AI Employee Advisor
st.subheader("💡 AI Talent Management Advisor")
emp_summary_context = (
    f"Employee Data Snapshot (Filtered Period: {start_date.strftime('%Y-%m-%d') if start_date else 'N/A'} to {end_date.strftime('%Y-%m-%d') if end_date else 'N/A'}, City: {city}, Type: {project_type}):\n"
    f"- Employees Analyzed: {total_employees_filtered}\n"
    f"- Total Certifications Held (Filtered): {total_certs}\n"
    f"- Average Labor Hours per Service Call (Filtered): {avg_labor_hours_per_call:.1f}\n"
    f"- Estimated Technician Utilization (All Billable Staff): {technician_utilization:.1f}%\n"
    f"- Certifications Expiring Within Next Year: {expiring_certs_count}\n"
)
render_ai_explanation(
    "Get AI-Powered Talent Management Advice",
    client,
    SUMMARY_ADVICE_PROMPT, # Using the general advice prompt
    emp_summary_context,
    additional_format_args={'current_date': CURRENT_DATE.strftime('%Y-%m-%d')}
)

st.divider()

# --- Section: Skills Matrix ---
st.header("📊 Skills Matrix (Certifications by Role)")
st.markdown("Distribution of the number of certifications held by employees across different roles (based on filtered employees).")

if employees_final.empty or 'NumCertifications' not in employees_final.columns or 'Role' not in employees_final.columns:
    st.warning("No employee data with roles and certifications available for the selected filters.")
else:
    try:
        # Use crosstab for heatmap-like visualization
        skills_crosstab = pd.crosstab(employees_final["Role"], employees_final["NumCertifications"])
        if not skills_crosstab.empty:
            fig_skills = px.imshow(
                skills_crosstab,
                title="Number of Employees by Role and Certification Count",
                text_auto=True, # Show counts on cells
                labels={'x': 'Number of Certifications', 'y': 'Employee Role', 'color': 'Employee Count'},
                color_continuous_scale=px.colors.sequential.Blues # Example color scale
            )
            st.plotly_chart(fig_skills, use_container_width=True)

            # AI Explanation for Skills Matrix
            skills_context = (
                f"Skills matrix visualization for {len(employees_final)} filtered employees.\n"
                f"- Shows counts of employees by Role and Number of Certifications held.\n"
                f"- Roles included: {', '.join(employees_final['Role'].unique())}.\n"
                f"- Certification counts range from {employees_final['NumCertifications'].min()} to {employees_final['NumCertifications'].max()}.\n"
                # Add specific observation e.g. role with highest avg certs
                # f"- Role with highest average certifications: {employees_final.groupby('Role')['NumCertifications'].mean().idxmax()}"
            )
            render_ai_explanation("AI Analysis of Skills Matrix", client, VISUAL_EXPLANATION_PROMPT, skills_context)
        else:
            st.info("No data to display in the skills matrix for the current filters.")

    except Exception as e:
        logger.error(f"Error generating skills matrix: {e}", exc_info=True)
        st.error("Could not generate the skills matrix visualization.")

st.divider()

# --- Section: Performance Metrics (Hours & Overtime Proxy) ---
st.header("📈 Performance Metrics")
st.markdown("Analysis of total hours logged per technician based on filtered time entries.")

perf_df = pd.DataFrame()
if not time_entries_final.empty and not employees_final.empty:
    try:
        # Aggregate hours from filtered time entries
        tech_hours_agg = time_entries_final.groupby('EmployeeID')['TotalHours'].sum().reset_index()

        # Merge with filtered employee data
        employees_final['EmployeeID'] = employees_final['EmployeeID'].astype(str)
        tech_hours_agg['EmployeeID'] = tech_hours_agg['EmployeeID'].astype(str)
        perf_df = tech_hours_agg.merge(employees_final[['EmployeeID', 'Name', 'Role']], on='EmployeeID', how='inner')

        # Calculate Overtime Hours Proxy (Requires OvertimeHours column from time_entries)
        if 'OvertimeHours' in time_entries_final.columns:
             time_entries_final['OvertimeHours'] = pd.to_numeric(time_entries_final['OvertimeHours'], errors='coerce').fillna(0)
             overtime_agg = time_entries_final.groupby('EmployeeID')['OvertimeHours'].sum().reset_index()
             overtime_agg['EmployeeID'] = overtime_agg['EmployeeID'].astype(str) # Ensure consistent type
             perf_df = perf_df.merge(overtime_agg, on='EmployeeID', how='left')
             perf_df['OvertimeHours'] = perf_df['OvertimeHours'].fillna(0)
             perf_df['OvertimeRatio'] = (perf_df['OvertimeHours'] / perf_df['TotalHours'] * 100).fillna(0)
        else:
             logger.warning("OvertimeHours column not found in time_entries. Cannot calculate overtime ratio.")
             perf_df['OvertimeHours'] = 0
             perf_df['OvertimeRatio'] = 0.0 # Default if column missing

        logger.info(f"Calculated performance metrics for {len(perf_df)} technicians.")

    except Exception as e:
        logger.error(f"Error calculating performance metrics: {e}", exc_info=True)
        st.error("Could not calculate performance metrics.")
        perf_df = pd.DataFrame()

if perf_df.empty:
    st.warning("No performance data available for the selected filters.")
else:
    perf_df = perf_df.sort_values('TotalHours', ascending=False)
    fig_perf = px.bar(
        perf_df,
        x="Name",
        y="TotalHours",
        color="Role", # Color by Role
        title="Technician Hours Logged (Filtered Period)",
        hover_data=['Role', 'OvertimeHours', 'OvertimeRatio'],
        labels={'Name': 'Technician', 'TotalHours': 'Total Hours Logged'}
    )
    # Optionally add text labels, e.g., overtime ratio
    # fig_perf.update_traces(texttemplate='%{customdata[2]:.1f}% OT', textposition='outside', customdata=perf_df[['Role', 'OvertimeHours', 'OvertimeRatio']].values)
    st.plotly_chart(fig_perf, use_container_width=True)

    # AI Explanation for Performance
    perf_context = (
        f"Technician performance metrics based on {len(perf_df)} filtered employees and their time entries.\n"
        f"- Total Hours Logged Range: {perf_df['TotalHours'].min():.1f} to {perf_df['TotalHours'].max():.1f}\n"
        f"- Average Total Hours: {perf_df['TotalHours'].mean():.1f}\n"
        f"- Average Overtime Hours (if available): {perf_df['OvertimeHours'].mean():.1f}\n"
        f"- Average Overtime Ratio (if available): {perf_df['OvertimeRatio'].mean():.1f}%\n"
    )
    render_ai_explanation("AI Analysis of Technician Performance", client, VISUAL_EXPLANATION_PROMPT, perf_context)

st.divider()

# --- Section: Certification Tracker ---
st.header("📅 Certification Tracker")
st.markdown("Timeline showing upcoming license/certification expirations for filtered employees.")

certs_plot_df = pd.DataFrame()
if not employees_final.empty and 'LicenseExpirations' in employees_final.columns and 'HireDate' in employees_final.columns:
    try:
        certs_plot_df = employees_final.copy()
        certs_plot_df['LicenseExpirations'] = pd.to_datetime(certs_plot_df['LicenseExpirations'], errors='coerce')
        # Use HireDate as start, but handle NaNs
        certs_plot_df['HireDate'] = pd.to_datetime(certs_plot_df['HireDate'], errors='coerce')
        # Fallback start date if HireDate is missing
        fallback_start = CURRENT_DATE - timedelta(days=5*365) # Assume max 5 years if hire date missing
        certs_plot_df['StartDateForPlot'] = certs_plot_df['HireDate'].fillna(fallback_start)


        # Drop entries where expiration date is missing for plotting
        certs_plot_df = certs_plot_df.dropna(subset=['LicenseExpirations', 'StartDateForPlot', 'Name'])

        if not certs_plot_df.empty:
            # Define Urgency based on proximity to CURRENT_DATE
            today_plus_90d = CURRENT_DATE + timedelta(days=90)
            today_plus_1y = CURRENT_DATE + timedelta(days=365)

            certs_plot_df['Urgency'] = np.select(
                [
                    certs_plot_df['LicenseExpirations'] < CURRENT_DATE, # Expired
                    certs_plot_df['LicenseExpirations'] < today_plus_90d, # Urgent (within 90 days)
                    certs_plot_df['LicenseExpirations'] < today_plus_1y, # Soon (within 1 year)
                ],
                [
                    "Expired",
                    "Urgent (90 Days)",
                    "Soon (1 Year)",
                ],
                default="OK (>1 Year)" # Default
            )
            urgency_order = ["Expired", "Urgent (90 Days)", "Soon (1 Year)", "OK (>1 Year)"]
            certs_plot_df['Urgency'] = pd.Categorical(certs_plot_df['Urgency'], categories=urgency_order, ordered=True)
            certs_plot_df = certs_plot_df.sort_values(['Urgency', 'LicenseExpirations']) # Sort by urgency then date

            fig_certs = px.timeline(
                certs_plot_df,
                x_start="StartDateForPlot", # Using the calculated start date
                x_end="LicenseExpirations",
                y="Name",
                color="Urgency",
                title="Certification/License Expiration Timeline (Filtered)",
                hover_data=['Role', 'LicenseExpirations'],
                labels={'StartDateForPlot': 'Start (Hire Date or Fallback)', 'LicenseExpirations': 'Expiration Date'},
                 color_discrete_map={ # Custom colors for urgency
                    "Expired": "darkred",
                    "Urgent (90 Days)": "red",
                    "Soon (1 Year)": "orange",
                    "OK (>1 Year)": "green"
                }
            )
            fig_certs.update_yaxes(categoryorder="array", categoryarray=certs_plot_df['Name'].tolist()) # Keep sorted order
            st.plotly_chart(fig_certs, use_container_width=True)

            # AI Explanation for Certifications
            cert_context = (
                f"Certification expiration timeline for {len(certs_plot_df)} filtered employees.\n"
                f"- Urgency breakdown: {certs_plot_df['Urgency'].value_counts().to_dict()}\n"
                f"- Number expiring within next year (including urgent/expired): {expiring_certs_count}\n"
                f"Timeline shows expiration dates colored by urgency relative to {CURRENT_DATE.strftime('%Y-%m-%d')}."
            )
            render_ai_explanation("AI Analysis of Certification Tracker", client, VISUAL_EXPLANATION_PROMPT, cert_context)

        else:
            st.info("No employees with valid license expiration dates found for the selected filters.")

    except Exception as e:
        logger.error(f"Error generating certification tracker: {e}", exc_info=True)
        st.error("Could not generate the certification tracker visualization.")
else:
    st.warning("Employee data is missing or lacks 'LicenseExpirations'/'HireDate' columns for the selected filters.")


st.divider()

# --- Section: Predict Employee Performance ---
st.header("🔮 Predict Employee Performance (Annual Hours)")
st.markdown("Estimate annual hours based on employee profile using historical data (trained on all data).")

# --- Model Training (using original data, cached resource) ---
@st.cache_resource
def train_performance_model():
    """Loads data, preprocesses, and trains the performance model."""
    logger.info("Attempting to train employee performance model...")
    data_for_train = load_all_data()
    if data_for_train is None or 'time_entries' not in data_for_train or 'employees' not in data_for_train:
        logger.error("Cannot train performance model: Time entry or employee data unavailable.")
        return None, None, None # Model, features, encoder

    time_entries_orig = data_for_train['time_entries'].copy()
    employees_orig = data_for_train['employees'].copy()

    if time_entries_orig.empty or employees_orig.empty:
        logger.warning("Time entry or employee data is empty. Cannot train performance model.")
        return None, None, None

    # --- Preprocessing --- (Same logic as Operations predictor)
    required_cols_emp = ['EmployeeID', 'Role', 'Certifications', 'HireDate']
    required_cols_time = ['EmployeeID', 'TotalHours', 'Date']
    if not all(col in employees_orig.columns for col in required_cols_emp) or \
       not all(col in time_entries_orig.columns for col in required_cols_time):
        logger.error("Cannot train performance model: Missing required columns.")
        return None, None, None

    time_entries_orig['TotalHours'] = pd.to_numeric(time_entries_orig['TotalHours'], errors='coerce').fillna(0)
    time_entries_orig['Date'] = pd.to_datetime(time_entries_orig['Date'], errors='coerce')
    time_entries_orig = time_entries_orig.dropna(subset=['Date'])
    if time_entries_orig.empty:
        logger.warning("No valid time entries with dates found for performance model.")
        return None, None, None
    time_entries_orig['Year'] = time_entries_orig['Date'].dt.year
    tech_avg_annual_hours = time_entries_orig.groupby(['EmployeeID'])['TotalHours'].mean().reset_index() # Average hours across all years in data
    tech_avg_annual_hours.rename(columns={'TotalHours': 'AvgAnnualHours'}, inplace=True)

    employees_orig['EmployeeID'] = employees_orig['EmployeeID'].astype(str)
    tech_avg_annual_hours['EmployeeID'] = tech_avg_annual_hours['EmployeeID'].astype(str)
    df_ml = tech_avg_annual_hours.merge(employees_orig, on='EmployeeID', how='inner')

    df_ml['NumCertifications'] = df_ml['Certifications'].apply(lambda x: len(x.split('|')) if isinstance(x, str) and x else 0)
    df_ml['HireDate'] = pd.to_datetime(df_ml['HireDate'], errors='coerce')
    df_ml['YearsOfService'] = (CURRENT_DATE - df_ml['HireDate']).dt.days / 365.25
    df_ml['YearsOfService'] = df_ml['YearsOfService'].fillna(0).clip(lower=0)

    df_ml['Role'] = df_ml['Role'].astype(str).fillna('Unknown')
    le_role = LabelEncoder()
    df_ml['RoleEncoded'] = le_role.fit_transform(df_ml['Role'])

    features = ['NumCertifications', 'YearsOfService', 'RoleEncoded']
    target = 'AvgAnnualHours'
    df_ml = df_ml.dropna(subset=[target])
    df_ml = df_ml.dropna(subset=features)

    if df_ml.empty or len(df_ml) < 10:
        logger.warning(f"Insufficient valid data ({len(df_ml)} rows) after preprocessing for performance model.")
        return None, None, None

    X = df_ml[features]
    y = df_ml[target]

    if X.isnull().values.any() or y.isnull().values.any() or np.isinf(X.values).any() or np.isinf(y.values).any():
        logger.error("NaN or Inf values detected in final training data (X or y) for performance model.")
        return None, None, None

    # --- Training ---
    try:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf=3)
        model.fit(X, y)
        logger.info(f"Employee performance prediction model trained successfully with features: {features}")
        return model, features, le_role # Return model, features USED, and role encoder
    except Exception as e:
        logger.error(f"Error training employee performance model: {e}", exc_info=True)
        return None, None, None

# Train the model (runs once and caches)
performance_model, performance_features, le_role_encoder_perf = train_performance_model()

# --- Prediction Form ---
if performance_model is None or not performance_features or le_role_encoder_perf is None:
    st.warning("Employee performance prediction model is unavailable due to training issues.")
else:
    st.markdown("**Enter Employee Profile Details:**")
    # Initialize session state
    if 'emp_perf_prediction_result' not in st.session_state:
        st.session_state.emp_perf_prediction_result = None
    if 'emp_perf_prediction_context' not in st.session_state:
        st.session_state.emp_perf_prediction_context = None
    if 'emp_perf_prediction_inputs' not in st.session_state:
        st.session_state.emp_perf_prediction_inputs = {
            'NumCertifications': 3,
            'YearsOfService': 5.0,
            'Role': le_role_encoder_perf.classes_[0] if len(le_role_encoder_perf.classes_) > 0 else 'Unknown'
        }

    with st.form("employee_performance_prediction_form"):
        role_options_perf = sorted(list(le_role_encoder_perf.classes_))
        default_role_perf = st.session_state.emp_perf_prediction_inputs.get('Role', role_options_perf[0] if role_options_perf else 'Unknown')
        default_role_index_perf = role_options_perf.index(default_role_perf) if default_role_perf in role_options_perf else 0

        pred_num_certs_perf = st.number_input("Number of Certifications", min_value=0, max_value=15, value=int(st.session_state.emp_perf_prediction_inputs.get('NumCertifications', 3)), step=1, key="pred_certs_perf")
        pred_years_service_perf = st.number_input("Years of Service", min_value=0.0, max_value=40.0, value=float(st.session_state.emp_perf_prediction_inputs.get('YearsOfService', 5.0)), step=0.5, key="pred_years_perf")
        pred_role_perf = st.selectbox("Role", options=role_options_perf, index=default_role_index_perf, key="pred_role_perf")

        submitted_perf = st.form_submit_button("Predict Annual Hours")

        if submitted_perf:
            logger.info("Employee performance prediction form submitted.")
            # Store inputs
            st.session_state.emp_perf_prediction_inputs = {
                 'NumCertifications': pred_num_certs_perf,
                 'YearsOfService': pred_years_service_perf,
                 'Role': pred_role_perf
            }

            try:
                encoded_role_perf = le_role_encoder_perf.transform([pred_role_perf])[0]
                input_data_perf = pd.DataFrame({
                    "NumCertifications": [pred_num_certs_perf],
                    "YearsOfService": [pred_years_service_perf],
                    "RoleEncoded": [encoded_role_perf]
                })
                # Reindex to match training features exactly
                input_data_perf = input_data_perf.reindex(columns=performance_features, fill_value=0)

                prediction_perf = performance_model.predict(input_data_perf)[0]
                prediction_perf = max(0, prediction_perf) # Ensure non-negative

                # --- Store results ---
                st.session_state.emp_perf_prediction_result = prediction_perf
                st.session_state.emp_perf_prediction_context = (
                    "INPUTS USED FOR PREDICTION:\n"
                    f"- Number of Certifications: {pred_num_certs_perf}\n"
                    f"- Years of Service: {pred_years_service_perf:.1f}\n"
                    f"- Role: {pred_role_perf}\n\n"
                    f"PREDICTED AVERAGE ANNUAL HOURS: {prediction_perf:,.0f}\n\n"
                    f"(Current Filtered Avg. Technician Utilization: {technician_utilization:.1f}%)\n" # Compare to current filtered util
                    "Note: Prediction based on historical averages. Actual hours influenced by workload, efficiency, and project assignments."
                )
                logger.info(f"Employee performance prediction successful: {prediction_perf:,.0f} hours")

            except Exception as e:
                st.error(f"Error during performance prediction calculation: {e}")
                logger.error("Employee performance prediction failed.", exc_info=True)
                st.session_state.emp_perf_prediction_result = None
                st.session_state.emp_perf_prediction_context = None
            # Rerun implicitly

    # --- Display results OUTSIDE the form ---
    if st.session_state.emp_perf_prediction_result is not None:
        st.success(f"**Predicted Average Annual Hours: {st.session_state.emp_perf_prediction_result:,.0f}**")
        if st.session_state.emp_perf_prediction_context and client:
             render_ai_explanation(
                 "AI Explanation of Performance Prediction",
                 client,
                 VISUAL_EXPLANATION_PROMPT,
                 st.session_state.emp_perf_prediction_context
             )
        elif not client:
             st.warning("AI Client unavailable, cannot generate explanation.")
        if st.button("Clear Prediction Result", key="clear_emp_perf_pred"):
             st.session_state.emp_perf_prediction_result = None
             st.session_state.emp_perf_prediction_context = None
             st.rerun()

st.divider()

# --- Section: Chat Feature ---
st.header("💬 Ask Questions About Employee Data") # Updated header text

# Prepare context for the chat (reuse emp_summary_context if available)
# Ensure emp_summary_context is defined earlier in the script
if 'emp_summary_context' not in locals():
     emp_summary_context = f"Employee data analysis for {len(employees_final)} filtered employees."
     logger.warning("Using fallback chat context as emp_summary_context was not defined.")

chat_base_context = emp_summary_context # Use the determined context

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
    page_key="employee", # Unique key for this page
    placeholder="Ask about skills, performance, or certifications..." # Updated placeholder
)
# --- End of Section: Chat Feature ---