# pages/3_Customer.py
import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
import logging
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# Import shared utilities
from utils.data_utils import load_all_data
from utils.filter_utils import render_global_filters, apply_filters, recalculate_costs_after_filtering
from utils.ai_utils import (
    get_genai_client, render_ai_explanation, render_chat_interface,
    SUMMARY_ADVICE_PROMPT, VISUAL_EXPLANATION_PROMPT, CHAT_PROMPT_TEMPLATE
)
from config import CSS_FILE, CURRENT_DATE

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

st.title("Customer Dashboard")

# --- AI Client Initialization ---
client = get_genai_client()
if not client:
    st.error("AI Client failed to initialize. AI features will be unavailable.")

# --- Customer-Specific Data (Keep as it's specific to this page) ---
WA_CITY_NEIGHBORHOOD_COORDS = {
    ('Seattle', 'Medina'): (47.6210, -122.2285), ('Seattle', 'Clyde Hill'): (47.6319, -122.2179),
    ('Seattle', 'Hunts Point'): (47.6435, -122.2300), ('Seattle', 'Madison Park'): (47.6360, -122.2790),
    ('Seattle', 'Broadmoor'): (47.6350, -122.2900), ('Seattle', 'Laurelhurst'): (47.6620, -122.2760),
    ('Seattle', 'Queen Anne'): (47.6320, -122.3560), ('Seattle', 'Magnolia'): (47.6500, -122.3990),
    ('Bellevue', 'West Bellevue'): (47.6150, -122.2100), ('Bellevue', 'Bridle Trails'): (47.6500, -122.1700),
    ('Bellevue', 'Beaux Arts'): (47.5850, -122.2050), ('Bellevue', 'Somerset'): (47.5800, -122.1600),
    ('Bellevue', 'Newport Shores'): (47.5700, -122.1800), ('Kirkland', 'Juanita'): (47.7100, -122.2050),
    ('Kirkland', 'Houghton'): (47.6900, -122.1950), ('Redmond', 'Sammamish Valley'): (47.6800, -122.1200),
    ('Redmond', 'Education Hill'): (47.6900, -122.1100), ('Issaquah', 'Issaquah Highlands'): (47.5400, -122.0400),
    ('Issaquah', 'Squak Mountain'): (47.5100, -122.0500), ('Sammamish', 'Sahalee'): (47.6300, -122.0600),
    ('Sammamish', 'Trossachs'): (47.6100, -122.0300), ('Mercer Island', 'North End'): (47.5900, -122.2300),
    ('Mercer Island', 'South End'): (47.5500, -122.2200), ('Woodinville', 'Hollywood Hill'): (47.7500, -122.1400),
    ('Woodinville', 'Bear Creek'): (47.7300, -122.0900)
    # Add more coordinates as needed
}

# --- Data Loading ---
data_orig = load_all_data()
if data_orig is None:
    st.error("Fatal error loading data. Dashboard cannot be displayed.")
    st.stop()

# Prepare employee rates (standard pattern, needed for recalc)
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
customers_final = filtered_data.get('customers', pd.DataFrame())
service_calls_final = filtered_data.get('service_calls', pd.DataFrame())
# projects_final = filtered_data.get('projects', pd.DataFrame()) # Uncomment if needed for visuals/calcs
# employees_final = filtered_data.get('employees', pd.DataFrame()) # Uncomment if needed

# --- Section: Customer Summary ---
st.header("📋 Customer Summary")
st.markdown("Key customer metrics based on the selected global filters.")

if customers_final.empty:
    st.warning("No customer data available for the selected filters.")
else:
    total_customers = len(customers_final)
    repeat_customers_count = len(customers_final[customers_final['RepeatCustomerStatus'] == 'Yes'])
    avg_annual_spend = customers_final['AnnualSpend'].mean() if 'AnnualSpend' in customers_final.columns and not customers_final['AnnualSpend'].isnull().all() else 0
    non_repeat_customers_count = total_customers - repeat_customers_count

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", total_customers)
    col2.metric("Repeat Customers", repeat_customers_count)
    col3.metric("Avg. Annual Spend", f"${avg_annual_spend:,.2f}")

        # AI Customer Advice
    st.subheader("💡 AI Customer Strategy Advisor")

    # Calculate percentage safely beforehand for clarity
    repeat_percentage = (repeat_customers_count / total_customers * 100) if total_customers > 0 else 0

    summary_context = (
        f"Customer Data Summary (Filtered Period: {start_date.strftime('%Y-%m-%d') if start_date else 'N/A'} to {end_date.strftime('%Y-%m-%d') if end_date else 'N/A'}, City: {city}, Type: {project_type}):\n"
        f"- Total Customers: {total_customers}\n"
        # Use the pre-calculated percentage with standard f-string formatting
        f"- Repeat Customers: {repeat_customers_count} ({repeat_percentage:.1f}%)\n"
        f"- Non-Repeat Customers: {non_repeat_customers_count}\n"
        f"- Average Annual Spend: ${avg_annual_spend:,.2f}\n"
    ) # <--- Added the missing closing parenthesis here

    render_ai_explanation(
        "Get AI-Powered Customer Strategy Advice",
        client,
        SUMMARY_ADVICE_PROMPT, # Using the general advice prompt template
        summary_context,
        additional_format_args={'current_date': CURRENT_DATE.strftime('%Y-%m-%d')}
    )

st.divider()

# --- Section: Customer Heatmap ---
st.header("🗺️ Customer Heatmap")
st.markdown("Geographical distribution of customers based on the selected global filters.")

# --- Define the corrected heatmap function ---
def create_customer_heatmap(customers_df):
    """Generates Folium map with heatmap layer based on customer locations."""
    wa_center = [47.7511, -120.7401] # Approximate center of WA
    m = folium.Map(location=wa_center, zoom_start=7, tiles='CartoDB positron') # Use a lighter base map
    heat_data = []
    unmatched = [] # Original list to collect keys during loop
    matched_locations = set() # To count unique matched locations

    # --- Initialize unique_unmatched outside the if block ---
    unique_unmatched = []

    if customers_df is None or customers_df.empty or 'City' not in customers_df.columns or 'Neighborhood' not in customers_df.columns:
        logger.warning("Heatmap generation skipped: Invalid or empty customer dataframe.")
        # Ensure it returns the initialized empty list here as well
        return m, heat_data, unique_unmatched

    for _, row in customers_df.iterrows():
        # Ensure city and neighborhood are strings and handle potential None values
        city = str(row.get("City", "Unknown")).strip().title()
        neighborhood = str(row.get("Neighborhood", "Unknown")).strip().title()
        key = (city, neighborhood)

        if key in WA_CITY_NEIGHBORHOOD_COORDS:
            lat, lon = WA_CITY_NEIGHBORHOOD_COORDS[key]
            heat_data.append([lat, lon, 1]) # Weight can be adjusted later if needed (e.g., by spend)
            matched_locations.add(key)
        elif key != ('Unknown', 'Unknown'): # Don't track 'Unknown' entries as unmatched
            unmatched.append(key) # Collect potentially duplicate keys

    logger.info(f"Heatmap: Found coordinates for {len(heat_data)} customer entries at {len(matched_locations)} unique locations.")
    if heat_data:
        try:
            # Add HeatMap layer
            folium.plugins.HeatMap(heat_data, radius=15, blur=20).add_to(m)
        except Exception as e:
            logger.error(f"Error adding HeatMap layer: {e}", exc_info=True)
            st.warning("Could not generate heatmap visualization due to an error.")
    else:
        st.info("No customer locations with known coordinates found for the current filters.")

    # Process and display unmatched locations
    if unmatched: # Only proceed if the original unmatched list has items
        unique_unmatched = sorted(list(set(unmatched))) # Assign the unique, sorted list
        display_limit = 5
        unmatched_str = ', '.join([f'{city}, {neighborhood}' for city, neighborhood in unique_unmatched[:display_limit]])
        if len(unique_unmatched) > display_limit:
            unmatched_str += f" and {len(unique_unmatched) - display_limit} more"
        st.caption(f"Note: Could not map coordinates for locations like: {unmatched_str}.")
    # Else: unique_unmatched remains the initialized empty list []

    # Now unique_unmatched is guaranteed to be defined
    return m, heat_data, unique_unmatched
# --- End of heatmap function definition ---


# --- Generate and Display Heatmap ---
if customers_final.empty:
    st.warning("No customer data available for the selected filters to generate a heatmap.")
else:
    # Call the corrected function
    map_obj, heat_data_list, unmatched_list = create_customer_heatmap(customers_final)
    # Display the map
    st_folium(map_obj, width=700, height=500, key="customer_heatmap_folium")

    # AI Explanation for Heatmap
    if heat_data_list: # Check if any data points were actually plotted
        # Summarize top locations based on heat_data coordinates
        location_coords = [(d[0], d[1]) for d in heat_data_list] # Extract lat, lon pairs
        # Use Counter for efficient counting
        from collections import Counter
        location_counts = Counter(location_coords)
        # Get top 3 most frequent coordinate pairs
        top_coords = [coord for coord, count in location_counts.most_common(3)]

        # Find neighborhood names for top coords (handle potential floating point inaccuracies)
        top_locations_names = []
        coord_to_name_map = {(clat, clon): f"{city}, {neigh}"
                              for (city, neigh), (clat, clon) in WA_CITY_NEIGHBORHOOD_COORDS.items()}

        for lat, lon in top_coords:
            # Find the closest match within a small tolerance
            found_match = False
            for (clat, clon), name in coord_to_name_map.items():
                if abs(lat - clat) < 1e-5 and abs(lon - clon) < 1e-5:
                    top_locations_names.append(name)
                    found_match = True
                    break
            if not found_match:
                 top_locations_names.append(f"Unknown ({lat:.4f}, {lon:.4f})") # Show coords if no name match


        heatmap_context = (
            f"Customer location heatmap based on {len(customers_final)} filtered customers.\n"
            f"- Heatmap visualizes concentrations from {len(heat_data_list)} entries with valid coordinates.\n"
            f"- Top 3 concentrated areas appear to be around: {', '.join(top_locations_names) if top_locations_names else 'N/A'}.\n"
            f"- {len(unmatched_list)} unique locations could not be mapped."
        )
        render_ai_explanation("AI Analysis of Customer Heatmap", client, VISUAL_EXPLANATION_PROMPT, heatmap_context)
    else:
         st.info("AI analysis skipped as no plottable heatmap data was generated.")

st.divider()

# --- Section: Customer Lifetime Value (CLV) ---
st.header("📊 Customer Lifetime Value Analysis")
st.markdown("Estimated CLV and retention risk based on the selected global filters.")

# Calculate CLV metrics using filtered data
clv_final = pd.DataFrame()
if not customers_final.empty and not service_calls_final.empty:
    try:
        # Calculate average satisfaction score per customer from filtered service calls
        service_calls_final['SatisfactionScore'] = pd.to_numeric(service_calls_final['SatisfactionScore'], errors='coerce')
        service_scores = service_calls_final.groupby('CustomerID')['SatisfactionScore'].mean().reset_index()

        # Merge with filtered customer data
        clv_final = customers_final.merge(service_scores, on='CustomerID', how='left')

        # Calculate Retention Risk (simple example based on satisfaction)
        # Ensure SatisfactionScore exists after merge
        if 'SatisfactionScore' not in clv_final.columns:
            clv_final['SatisfactionScore'] = np.nan # Add if missing
        clv_final['SatisfactionScore'] = clv_final['SatisfactionScore'].fillna(0) # Fill NA scores with 0 for risk calc
        clv_final['RetentionRisk'] = clv_final['SatisfactionScore'].apply(lambda x: "High" if x < 3.5 else "Low") # Adjusted threshold

        # Estimate CLV (simple example: spend * multiplier based on repeat status)
        # Ensure necessary columns exist
        if 'AnnualSpend' not in clv_final.columns: clv_final['AnnualSpend'] = 0
        if 'RepeatCustomerStatus' not in clv_final.columns: clv_final['RepeatCustomerStatus'] = 'No'

        clv_final['AnnualSpend'] = pd.to_numeric(clv_final['AnnualSpend'], errors='coerce').fillna(0)
        clv_final['CLV'] = clv_final['AnnualSpend'] * clv_final['RepeatCustomerStatus'].apply(lambda x: 5 if x == 'Yes' else 2) # Example multipliers

        logger.info(f"CLV calculations performed on {len(clv_final)} filtered customers.")

    except Exception as e:
        logger.error(f"Error calculating CLV: {e}", exc_info=True)
        st.error("Could not calculate CLV due to an error.")
        clv_final = pd.DataFrame() # Reset on error

if clv_final.empty:
    st.warning("No data available to display CLV analysis for the selected filters.")
else:
    fig_clv = px.scatter(
        clv_final,
        x="AnnualSpend",
        y="CLV",
        color="RetentionRisk",
        size="CLV",  # Optional: size points by CLV
        hover_data=['CustomerID', 'Name', 'SatisfactionScore', 'RepeatCustomerStatus'],
        color_discrete_map={"High": "red", "Low": "green"},
        title="CLV vs. Annual Spend (Colored by Retention Risk)"
    )
    fig_clv.update_layout(xaxis_title="Annual Spend ($)", yaxis_title="Estimated Customer Lifetime Value ($)")
    st.plotly_chart(fig_clv, use_container_width=True)

    # AI Explanation for CLV
    clv_context = (
        f"CLV analysis based on {len(clv_final)} filtered customers.\n"
        f"- Average Estimated CLV: ${clv_final['CLV'].mean():,.0f}\n"
        f"- Customers at High Retention Risk: {len(clv_final[clv_final['RetentionRisk'] == 'High'])}\n"
        f"- Average Satisfaction Score (where available): {clv_final[clv_final['SatisfactionScore'] > 0]['SatisfactionScore'].mean():.2f}\n"
         f"- Average Annual Spend: ${clv_final['AnnualSpend'].mean():,.2f}\n"
        f"Note: CLV is estimated based on Annual Spend and Repeat Status. Retention Risk based on Satisfaction Score < 3.5."
    )
    render_ai_explanation("AI Analysis of CLV", client, VISUAL_EXPLANATION_PROMPT, clv_context)

st.divider()

# --- Section: Satisfaction Trends ---
st.header("📈 Satisfaction Trends")
st.markdown("Tracking average customer satisfaction scores over time based on the selected global filters.")

trend_df = pd.DataFrame()
if not service_calls_final.empty and 'DateStr' in service_calls_final.columns and 'SatisfactionScore' in service_calls_final.columns:
    try:
        # Prepare data for trend analysis
        trends_calc_df = service_calls_final.copy()
        trends_calc_df['DateStr'] = pd.to_datetime(trends_calc_df['DateStr'], errors='coerce')
        trends_calc_df['SatisfactionScore'] = pd.to_numeric(trends_calc_df['SatisfactionScore'], errors='coerce')
        trends_calc_df = trends_calc_df.dropna(subset=['DateStr', 'SatisfactionScore'])

        if not trends_calc_df.empty:
            # Calculate monthly average and then a 3-month rolling average
            trend_df = trends_calc_df.set_index('DateStr') \
                .resample('ME')['SatisfactionScore'] \
                .mean() \
                .rolling(window=3, min_periods=1) \
                .mean() \
                .reset_index()
            trend_df.rename(columns={'SatisfactionScore': 'AvgSatisfaction3M'}, inplace=True)
            logger.info(f"Calculated satisfaction trend over {len(trend_df)} months.")
        else:
            logger.info("No valid service call data with dates and scores for trend calculation.")

    except Exception as e:
        logger.error(f"Error calculating satisfaction trends: {e}", exc_info=True)
        st.error("Could not calculate satisfaction trends due to an error.")
        trend_df = pd.DataFrame()


if trend_df.empty:
    st.warning("No data available to display Satisfaction Trends for the selected filters.")
else:
    fig_trend = px.line(
        trend_df,
        x="DateStr",
        y="AvgSatisfaction3M",
        title="Customer Satisfaction Trend (3-Month Moving Average)",
        markers=True,
        labels={'DateStr': 'Month', 'AvgSatisfaction3M': 'Avg. Satisfaction Score (1-5)'}
    )
    fig_trend.update_layout(yaxis_range=[1, 5.1]) # Set Y-axis range for clarity
    st.plotly_chart(fig_trend, use_container_width=True)

    # AI Explanation for Trends
    trends_context = (
        f"Satisfaction score trend (3-month moving average) based on {len(service_calls_final)} filtered service calls.\n"
        f"- Trend Period: {trend_df['DateStr'].min().strftime('%Y-%m')} to {trend_df['DateStr'].max().strftime('%Y-%m')}\n"
        f"- Score Range in Period: {trend_df['AvgSatisfaction3M'].min():.2f} to {trend_df['AvgSatisfaction3M'].max():.2f}\n"
        f"- Latest Average Score: {trend_df['AvgSatisfaction3M'].iloc[-1]:.2f}\n"
    )
    render_ai_explanation("AI Analysis of Satisfaction Trends", client, VISUAL_EXPLANATION_PROMPT, trends_context)

st.divider()

# --- Section: Predict Satisfaction Score ---
st.header("🔮 Predict Satisfaction Score")
st.markdown("Estimate satisfaction for a future service call based on historical patterns (trained on all service calls).")

# --- Model Training (using original data, cached resource) ---
@st.cache_resource
def train_satisfaction_model():
    """Loads service call data, preprocesses, and trains the satisfaction model."""
    logger.info("Attempting to train satisfaction prediction model...")
    data_for_train = load_all_data() # Load fresh copy for training isolation
    if data_for_train is None or 'service_calls' not in data_for_train or data_for_train['service_calls'].empty:
        logger.error("Cannot train satisfaction model: Service call data unavailable.")
        return None, None # Return None for model and encoder

    df_ml = data_for_train['service_calls'].copy()

    # --- Preprocessing ---
    required_cols = ['SatisfactionScore', 'ResponseTime', 'ServiceType']
    if not all(col in df_ml.columns for col in required_cols):
        logger.error(f"Cannot train satisfaction model: Missing required columns ({required_cols}).")
        return None, None

    df_ml['SatisfactionScore'] = pd.to_numeric(df_ml['SatisfactionScore'], errors='coerce')
    df_ml['ResponseTime'] = pd.to_numeric(df_ml['ResponseTime'], errors='coerce')

    # Handle potential infinite values before fillna
    df_ml.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill NaNs for key features - decide strategy (e.g., mean, median, 0)
    df_ml['ResponseTime'] = df_ml['ResponseTime'].fillna(df_ml['ResponseTime'].median()) # Use median for response time

    # Encode ServiceType
    df_ml['ServiceType'] = df_ml['ServiceType'].astype(str).fillna('Unknown')
    le_service = LabelEncoder()
    df_ml['ServiceTypeEncoded'] = le_service.fit_transform(df_ml['ServiceType'])

    # Drop rows where target is still NaN
    df_ml = df_ml.dropna(subset=['SatisfactionScore'])

    features = ['ResponseTime', 'ServiceTypeEncoded']
    target = 'SatisfactionScore'

    if df_ml.empty or len(df_ml) < 10:
        logger.warning(f"Insufficient valid data ({len(df_ml)} rows) after preprocessing for satisfaction model training.")
        return None, None

    X = df_ml[features]
    y = df_ml[target]

    # Final check for NaNs/Infs in training data
    if X.isnull().values.any() or y.isnull().values.any() or np.isinf(X.values).any() or np.isinf(y.values).any():
        logger.error("NaN or Inf values detected in final training data (X or y) for satisfaction model. Aborting.")
        return None, None

    # --- Training ---
    try:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=8, min_samples_leaf=5)
        model.fit(X, y)
        logger.info(f"Satisfaction prediction model trained successfully with features: {features}")
        return model, le_service # Return trained model and the encoder
    except Exception as e:
        logger.error(f"Error training satisfaction prediction model: {e}", exc_info=True)
        return None, None

# Call the training function (will run once and cache the result)
satisfaction_model, le_service_encoder = train_satisfaction_model()

# --- Prediction Form and Result Display ---
if satisfaction_model is None or le_service_encoder is None:
    st.warning("Satisfaction prediction model is unavailable due to training issues. Please check logs.")
else:
    st.markdown("**Enter Service Call Details:**")

    # Use session state to store prediction results and context (like in Financials)
    if 'satisfaction_prediction_result' not in st.session_state:
        st.session_state.satisfaction_prediction_result = None
    if 'satisfaction_prediction_context' not in st.session_state:
        st.session_state.satisfaction_prediction_context = None
    if 'satisfaction_prediction_inputs' not in st.session_state:
        # Initialize with defaults matching the form
        st.session_state.satisfaction_prediction_inputs = {
            'ResponseTime': 60,
            'ServiceType': le_service_encoder.classes_[0] if len(le_service_encoder.classes_) > 0 else 'Unknown'
        }

    with st.form("satisfaction_prediction_form"):
        # Get available service types from the encoder used during training
        service_type_options = sorted(list(le_service_encoder.classes_))
        default_service_type = st.session_state.satisfaction_prediction_inputs.get('ServiceType', service_type_options[0] if service_type_options else 'Unknown')
        default_st_index = service_type_options.index(default_service_type) if default_service_type in service_type_options else 0

        # Form Inputs using session state defaults
        pred_response_time = st.number_input(
            "Expected Response Time (minutes)",
            min_value=0, max_value=1440, # Max 1 day
            value=int(st.session_state.satisfaction_prediction_inputs.get('ResponseTime', 60)),
            step=15
        )
        pred_service_type = st.selectbox(
            "Service Type",
            options=service_type_options,
            index=default_st_index
        )

        submitted = st.form_submit_button("Predict Satisfaction")

        if submitted:
            logger.info("Satisfaction prediction form submitted.")
            # Store current inputs
            st.session_state.satisfaction_prediction_inputs = {
                 'ResponseTime': pred_response_time,
                 'ServiceType': pred_service_type
            }

            try:
                # Encode the selected service type using the loaded encoder
                encoded_service_type = le_service_encoder.transform([pred_service_type])[0]

                # Prepare input DataFrame for the model
                input_data = pd.DataFrame({
                    "ResponseTime": [pred_response_time],
                    "ServiceTypeEncoded": [encoded_service_type]
                })
                # Ensure columns match model features exactly
                input_data = input_data[['ResponseTime', 'ServiceTypeEncoded']]

                # Make prediction
                prediction = satisfaction_model.predict(input_data)[0]
                # Clip prediction to reasonable bounds (1-5)
                prediction = np.clip(prediction, 1, 5)

                # --- Store results in session state ---
                st.session_state.satisfaction_prediction_result = prediction
                st.session_state.satisfaction_prediction_context = (
                    "INPUTS USED FOR PREDICTION:\n"
                    f"- Expected Response Time: {pred_response_time} minutes\n"
                    f"- Service Type: {pred_service_type}\n\n"
                    f"PREDICTED SATISFACTION SCORE: {prediction:.2f} / 5.0\n\n"
                    f"(Current Filtered Avg Satisfaction: {service_calls_final['SatisfactionScore'].mean():.2f} / 5.0 if available else 'N/A')\n"
                    "Note: Prediction based on historical patterns in all service calls. Factors like technician skill, issue complexity, and specific customer expectations also influence actual satisfaction."
                )
                logger.info(f"Satisfaction prediction successful: {prediction:.2f}")

            except Exception as e:
                st.error(f"Error during satisfaction prediction calculation: {e}")
                logger.error("Satisfaction prediction failed.", exc_info=True)
                st.session_state.satisfaction_prediction_result = None
                st.session_state.satisfaction_prediction_context = None
            # Rerun implicitly handled by form submission end

    # --- Display results OUTSIDE the form ---
    if st.session_state.satisfaction_prediction_result is not None:
        st.success(f"**Predicted Satisfaction Score: {st.session_state.satisfaction_prediction_result:.2f} / 5.0**")

        # Display AI Explanation using stored context
        if st.session_state.satisfaction_prediction_context and client:
             render_ai_explanation(
                 "AI Explanation of Satisfaction Prediction",
                 client,
                 VISUAL_EXPLANATION_PROMPT, # Can use this generic one
                 st.session_state.satisfaction_prediction_context
             )
        elif not client:
             st.warning("AI Client unavailable, cannot generate explanation.")

        # Button to clear prediction
        if st.button("Clear Prediction Result", key="clear_satisfaction_pred"):
             st.session_state.satisfaction_prediction_result = None
             st.session_state.satisfaction_prediction_context = None
             # Optionally reset inputs: st.session_state.satisfaction_prediction_inputs = {...}
             st.rerun()

st.divider()

# --- Section: Chat Feature ---
st.header("💬 Dig Into The Details") # Standardized Header matching other pages

# Prepare context for the chat.
# Reuse 'summary_context' if defined earlier, otherwise create a fallback.
if 'summary_context' not in locals():
     # Fallback context if summary section was skipped (e.g., no customer data)
     summary_context = f"Customer data analysis for {len(customers_final)} filtered customers."
     logger.warning("Using fallback chat context as summary_context was not defined.")

chat_base_context = summary_context # Use the determined context

# Prepare the dictionary with current filter values, matching other pages
filter_details_dict = {
    'start_date': start_date.strftime('%Y-%m-%d') if start_date else 'N/A',
    'end_date': end_date.strftime('%Y-%m-%d') if end_date else 'N/A',
    'city': city,
    'project_type': project_type # Key name used in filter_utils and expected by template
}

# Render the standard chat interface - this function will handle displaying
# suggestions based on the page_key="customer" (assuming they are defined in ai_utils.py)
render_chat_interface(
    client=client,
    chat_prompt_template=CHAT_PROMPT_TEMPLATE,
    base_context=chat_base_context,
    filter_details=filter_details_dict,
    page_key="customer", # Crucial: Unique key for this page
    placeholder="Ask about filtered customer data, CLV, or satisfaction..."
)