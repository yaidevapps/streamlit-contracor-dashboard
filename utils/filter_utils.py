# utils/filter_utils.py
import streamlit as st
import pandas as pd
import logging
import numpy as np
from config import DEFAULT_FILTER_START_DATE, DEFAULT_FILTER_END_DATE

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def render_global_filters(data):
    """Renders global filter widgets and returns selected values."""
    st.subheader("Global Filters")
    st.markdown("**Adjust filters below to explore data across all sections.**")

    with st.expander("Apply Global Filters", expanded=False):
        st.markdown("**Select a date range, city, or project type to filter all data (optional).**")

        # Date Range
        global_date_range = st.date_input(
            "Select Analysis Period",
            value=(DEFAULT_FILTER_START_DATE, DEFAULT_FILTER_END_DATE),
            help="Filters data based on primary date fields (Project Start, Service Call, Expense, Invoice, Bid)."
        )

        # City (from Customers)
        cities = ["All"]
        if data and "customers" in data and not data["customers"].empty:
            try:
                # Ensure City is string, handle NA, get unique, sort
                cities = ["All"] + sorted([str(c) for c in data["customers"]["City"].dropna().unique()])
            except Exception as e:
                logger.error(f"Error processing cities for filter: {e}")
                st.warning("Could not load city filter options.")
        else:
            logger.warning("Customer data not available for city filter.")

        global_city = st.selectbox("City (Global - Customer Location)", cities, key="global_city_filter")

        # Project Type (from Projects)
        project_types = ["All"]
        if data and "projects" in data and not data["projects"].empty:
             try:
                # Ensure ProjectType is string, handle NA, get unique, sort
                project_types = ["All"] + sorted([str(pt) for pt in data["projects"]["ProjectType"].dropna().unique()])
             except Exception as e:
                logger.error(f"Error processing project types for filter: {e}")
                st.warning("Could not load project type filter options.")
        else:
            logger.warning("Project data not available for project type filter.")

        global_project_type = st.selectbox("Project Type (Global)", project_types, key="global_project_type_filter")

    start_date = global_date_range[0] if global_date_range and len(global_date_range) == 2 else None
    end_date = global_date_range[1] if global_date_range and len(global_date_range) == 2 else None

    # Ensure dates are valid before returning
    if start_date and end_date and start_date > end_date:
        st.error("Start date cannot be after end date.")
        return None, None, global_city, global_project_type # Indicate error state

    return start_date, end_date, global_city, global_project_type


def apply_filters(data, start_date, end_date, city, project_type):
    """
    Applies global filters to the data dictionary.
    Returns a *new* dictionary with filtered dataframes.
    """
    if not data:
        logger.error("Cannot apply filters: Input data dictionary is empty.")
        return {}

    # Create copies to avoid modifying original cached data
    filtered_data = {key: df.copy() for key, df in data.items() if isinstance(df, pd.DataFrame)}
    logger.info(f"Applying filters - Date: {start_date}-{end_date}, City: {city}, Project Type: {project_type}")

    # 1. Date Filter
    if start_date and end_date:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        date_filter_cols = {
            # Primary date for filtering each table
            'projects': 'StartDateStr', 'service_calls': 'DateStr', 'invoices': 'InvoiceDate',
            'expenses': 'Date', 'bids': 'BidDate', 'time_entries': 'Date', 'marketing_campaigns': 'StartDate',
            'warranty_claims': 'ClaimDate', 'project_timelines': 'EstimatedDate', # Or ActualDate? Choose primary.
            'training_records': 'StartDate', 'equipment': 'PurchaseDate', 'customers': 'DateAcquired'
        }
        for key, date_col in date_filter_cols.items():
            if key in filtered_data and date_col in filtered_data[key].columns:
                df = filtered_data[key]
                # Ensure date column is datetime type before filtering
                if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                    logger.debug(f"Converting '{date_col}' in '{key}' to datetime for filtering.")
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

                original_count = len(df)
                # Apply filter, handling NaT dates which should not match the range
                mask = df[date_col].between(start_dt, end_dt, inclusive='both') & df[date_col].notna()
                df = df[mask]
                filtered_data[key] = df
                if len(df) < original_count:
                    logger.info(f"Date filter on '{key}': {original_count} -> {len(df)} rows")
            # else: logger.debug(f"Date column '{date_col}' not found or key '{key}' missing for date filter.")


    # 2. City Filter (Based on Customer/Project Location)
    customer_ids_in_city = []
    project_ids_in_city = []
    if city != "All":
        logger.info(f"Applying city filter: {city}")
        # Find customers in the city (use already date-filtered customer data)
        if "customers" in filtered_data and not filtered_data["customers"].empty:
            cust_df = filtered_data["customers"]
            customer_ids_in_city = cust_df[cust_df['City'] == city]['CustomerID'].astype(str).unique()
            logger.info(f"Found {len(customer_ids_in_city)} customers in {city} within date range.")

        # Find projects located directly in the city (use already date-filtered project data)
        if "projects" in filtered_data and not filtered_data["projects"].empty:
            proj_df = filtered_data["projects"]
            # Filter projects directly by City
            original_proj_count = len(proj_df)
            proj_df = proj_df[proj_df['City'] == city]
            filtered_data['projects'] = proj_df # Update filtered projects
            project_ids_in_city = proj_df['ProjectID'].astype(str).unique() # Get IDs *after* filtering projects
            if len(proj_df) < original_proj_count:
                logger.info(f"City filter on 'projects' (direct): {original_proj_count} -> {len(proj_df)} rows")

        # Filter related tables based on the Customers or Projects found in the city
        related_tables_to_filter = ['service_calls', 'invoices', 'expenses', 'bids', 'time_entries', 'warranty_claims', 'change_orders', 'project_timelines']
        for key in related_tables_to_filter:
            if key in filtered_data and (len(customer_ids_in_city) > 0 or len(project_ids_in_city) > 0): # Only filter if we found related IDs
                df = filtered_data[key]
                original_count = len(df)
                city_mask = pd.Series(False, index=df.index) # Start with False

                # Check by CustomerID if applicable and IDs found
                if 'CustomerID' in df.columns and len(customer_ids_in_city) > 0:
                    # Ensure type consistency before comparison
                    city_mask |= df['CustomerID'].astype(str).isin(customer_ids_in_city)

                # Check by ProjectID (handling different column names) if applicable and IDs found
                proj_id_col = None
                if 'ProjectID' in df.columns: proj_id_col = 'ProjectID'
                elif 'RelatedProjectID' in df.columns: proj_id_col = 'RelatedProjectID'
                if proj_id_col and len(project_ids_in_city) > 0:
                    # Ensure type consistency
                    city_mask |= df[proj_id_col].astype(str).isin(project_ids_in_city)

                filtered_data[key] = df[city_mask]
                if len(filtered_data[key]) < original_count:
                    logger.info(f"City filter on '{key}': {original_count} -> {len(filtered_data[key])} rows")


    # 3. Project Type Filter
    if project_type != "All":
        logger.info(f"Applying project type filter: {project_type}")
        project_ids_of_type = []
        # Filter projects dataframe first (use already date/city filtered project data)
        if "projects" in filtered_data and not filtered_data["projects"].empty:
            proj_df = filtered_data["projects"]
            original_proj_count = len(proj_df)
            proj_df = proj_df[proj_df['ProjectType'] == project_type]
            filtered_data['projects'] = proj_df # Update filtered projects again
            project_ids_of_type = proj_df['ProjectID'].astype(str).unique() # Get IDs *after* this filter
            if len(proj_df) < original_proj_count:
                logger.info(f"Project Type filter on 'projects': {original_proj_count} -> {len(proj_df)} rows")
            logger.info(f"Found {len(project_ids_of_type)} projects of type '{project_type}' matching previous filters.")

        # Filter related tables based on the ProjectIDs remaining after all filters
        related_tables_to_filter = ['service_calls', 'invoices', 'expenses', 'time_entries', 'warranty_claims', 'change_orders', 'project_timelines']
        for key in related_tables_to_filter:
            if key in filtered_data and len(project_ids_of_type) > 0: # Only filter if projects of type were found
                df = filtered_data[key]
                original_count = len(df)
                proj_id_col = None
                if 'ProjectID' in df.columns: proj_id_col = 'ProjectID'
                elif 'RelatedProjectID' in df.columns: proj_id_col = 'RelatedProjectID'

                if proj_id_col:
                    # Ensure type consistency
                    type_mask = df[proj_id_col].astype(str).isin(project_ids_of_type)
                    filtered_data[key] = df[type_mask]
                    if len(filtered_data[key]) < original_count:
                        logger.info(f"Project Type filter on '{key}': {original_count} -> {len(filtered_data[key])} rows")

        # Filter bids separately (can match ProjectType directly OR linked ProjectID)
        if 'bids' in filtered_data:
             df = filtered_data['bids']
             original_count = len(df)
             # Match directly on ProjectType OR if its RelatedProjectID matches one of the filtered projects
             type_mask = (df['ProjectType'] == project_type)
             if len(project_ids_of_type) > 0 and 'RelatedProjectID' in df.columns:
                  # Ensure type consistency before isin
                  type_mask |= df['RelatedProjectID'].astype(str).isin(project_ids_of_type)

             filtered_data['bids'] = df[type_mask]
             if len(filtered_data['bids']) < original_count:
                  logger.info(f"Project Type filter on 'bids': {original_count} -> {len(filtered_data['bids'])} rows")

    return filtered_data

def recalculate_costs_after_filtering(filtered_data, employee_rates_orig):
    """
    Recalculates labor costs based on *filtered* time entries and updates
    costs and margins in the *filtered* projects and service calls dataframes.

    Args:
        filtered_data (dict): Dictionary of filtered pandas DataFrames.
        employee_rates_orig (pd.DataFrame): Unfiltered employee rates (index=EmployeeID, cols=['HourlyRate', 'OvertimeRate']).

    Returns:
        dict: The filtered_data dictionary with updated cost/margin calculations.
    """
    # --- Input Validation ---
    if "time_entries" not in filtered_data or filtered_data["time_entries"].empty:
        logger.warning("Filtered time entries empty or missing. Cannot recalculate costs.")
        # Ensure cost columns exist even if calculation fails (initialize if needed)
        if "projects" in filtered_data:
            for col in ['ProjectLaborCost', 'TotalExpenses', 'ProfitMargin']:
                if col not in filtered_data["projects"].columns: filtered_data["projects"][col] = 0.0
        if "service_calls" in filtered_data:
            for col in ['ServiceLaborCost', 'TotalServiceCost']:
                if col not in filtered_data["service_calls"].columns: filtered_data["service_calls"][col] = 0.0
        return filtered_data

    if employee_rates_orig is None or employee_rates_orig.empty:
         logger.warning("Original employee rates are missing or empty. Cannot merge rates for cost recalculation. Costs may be inaccurate.")
         # Attempt to continue, but costs will likely be 0 or based on potentially existing old columns
         # Initialize cost columns if they don't exist
         if "projects" in filtered_data:
            for col in ['ProjectLaborCost', 'TotalExpenses', 'ProfitMargin']:
                 if col not in filtered_data["projects"].columns: filtered_data["projects"][col] = 0.0
         if "service_calls" in filtered_data:
            for col in ['ServiceLaborCost', 'TotalServiceCost']:
                 if col not in filtered_data["service_calls"].columns: filtered_data["service_calls"][col] = 0.0
         # Initialize LaborCost in time_entries if not present
         if 'LaborCost' not in filtered_data['time_entries'].columns:
              filtered_data['time_entries']['LaborCost'] = 0.0
         # Aggregation might still run but yield 0s if LaborCost is 0
         # Fall through to attempt aggregation with potentially zero costs

    logger.info("Recalculating costs based on filtered time entries...")
    time_entries_filt = filtered_data["time_entries"].copy()

    # --- Merge Employee Rates (with suffix handling) ---
    if employee_rates_orig is not None and not employee_rates_orig.empty:
        # Ensure EmployeeID is string for merging
        time_entries_filt['EmployeeID'] = time_entries_filt['EmployeeID'].astype(str)
        # Ensure index of rates df is also string if not already
        if employee_rates_orig.index.dtype != object and employee_rates_orig.index.dtype != str :
             employee_rates_orig.index = employee_rates_orig.index.astype(str)

        time_entries_filt = time_entries_filt.merge(
            employee_rates_orig,
            on='EmployeeID',
            how='left',
            suffixes=('_orig', '_new') # Define explicit suffixes
        )

        # Consolidate rate columns, prioritizing '_new' from the merge
        if 'HourlyRate_new' in time_entries_filt.columns:
            time_entries_filt['HourlyRate'] = time_entries_filt['HourlyRate_new'].fillna(0) # Fill NaNs from failed merge
            time_entries_filt.drop(columns=['HourlyRate_new'], inplace=True)
            if 'HourlyRate_orig' in time_entries_filt.columns: # Drop original if it exists
                time_entries_filt.drop(columns=['HourlyRate_orig'], inplace=True)
        elif 'HourlyRate' in time_entries_filt.columns: # If no suffix (no conflict), fillna existing
            time_entries_filt['HourlyRate'] = time_entries_filt['HourlyRate'].fillna(0)
        else: # Rate column was missing completely
            logger.error("'HourlyRate' column missing after merge. Setting to 0.")
            time_entries_filt['HourlyRate'] = 0

        if 'OvertimeRate_new' in time_entries_filt.columns:
            time_entries_filt['OvertimeRate'] = time_entries_filt['OvertimeRate_new'].fillna(0)
            time_entries_filt.drop(columns=['OvertimeRate_new'], inplace=True)
            if 'OvertimeRate_orig' in time_entries_filt.columns:
                time_entries_filt.drop(columns=['OvertimeRate_orig'], inplace=True)
        elif 'OvertimeRate' in time_entries_filt.columns:
            time_entries_filt['OvertimeRate'] = time_entries_filt['OvertimeRate'].fillna(0)
        else:
            logger.error("'OvertimeRate' column missing after merge. Setting to 0.")
            time_entries_filt['OvertimeRate'] = 0
    else:
         # If employee rates were missing, ensure rate columns exist and are 0
         logger.warning("Employee rates missing, setting HourlyRate and OvertimeRate to 0 for cost calculation.")
         time_entries_filt['HourlyRate'] = 0.0
         time_entries_filt['OvertimeRate'] = 0.0


    # --- Recalculate LaborCost on time_entries_filt ---
    # Ensure hours columns exist and are numeric
    if 'RegularHours' not in time_entries_filt.columns or 'OvertimeHours' not in time_entries_filt.columns:
        logger.error("Hours columns ('RegularHours', 'OvertimeHours') missing in filtered time_entries. Cannot calculate LaborCost.")
        time_entries_filt['LaborCost'] = 0.0 # Set default cost
    else:
        # Ensure numeric types for calculation
        time_entries_filt['RegularHours'] = pd.to_numeric(time_entries_filt['RegularHours'], errors='coerce').fillna(0)
        time_entries_filt['OvertimeHours'] = pd.to_numeric(time_entries_filt['OvertimeHours'], errors='coerce').fillna(0)
        # HourlyRate and OvertimeRate should be numeric (or 0) from previous step
        time_entries_filt['LaborCost'] = (time_entries_filt['RegularHours'] * time_entries_filt['HourlyRate']) + \
                                         (time_entries_filt['OvertimeHours'] * time_entries_filt['OvertimeRate'])
    logger.debug("LaborCost recalculated on filtered time entries.")


    # --- Update Projects ---
    if "projects" in filtered_data:
        projects_filt = filtered_data["projects"].copy()
        if not projects_filt.empty:
            # --- Pre-checks for project update ---
            if 'ProjectID' not in projects_filt.columns:
                logger.error("Cannot update projects: 'ProjectID' column missing.")
            elif 'ProjectID' not in time_entries_filt.columns:
                 logger.error("Cannot update projects: 'ProjectID' column missing in time_entries_filt.")
            else:
                # Ensure ProjectID is string type for aggregation and merge
                projects_filt['ProjectID'] = projects_filt['ProjectID'].astype(str)
                time_entries_filt['ProjectID'] = time_entries_filt['ProjectID'].astype(str)

                # Aggregate labor from FILTERED time entries (with recalculated LaborCost)
                proj_labor_filt = time_entries_filt[
                    time_entries_filt['ProjectID'].notna() & (time_entries_filt['ProjectID'] != 'None')
                ].groupby('ProjectID')['LaborCost'].sum().reset_index()
                proj_labor_filt.rename(columns={'LaborCost': 'ProjectLaborCost_recalc'}, inplace=True)
                proj_labor_filt['ProjectID'] = proj_labor_filt['ProjectID'].astype(str) # Ensure consistent type

                # Merge recalculated labor cost, potentially overwriting initial calculation
                projects_filt = projects_filt.merge(proj_labor_filt, on='ProjectID', how='left')

                # Consolidate: Use the recalculated value if present, else keep original (if exists), default 0
                if 'ProjectLaborCost_recalc' in projects_filt.columns:
                    projects_filt['ProjectLaborCost'] = projects_filt['ProjectLaborCost_recalc'].fillna(projects_filt.get('ProjectLaborCost', 0))
                    projects_filt.drop(columns=['ProjectLaborCost_recalc'], inplace=True)
                projects_filt['ProjectLaborCost'] = projects_filt['ProjectLaborCost'].fillna(0) # Ensure column exists and NAs are 0

                # Recalculate Total Expenses using recalculated labor and *existing* OtherExpenses
                if 'OtherExpenses' not in projects_filt.columns:
                    logger.warning("'OtherExpenses' missing during project recalc. Assuming 0.")
                    projects_filt['OtherExpenses'] = 0.0
                else:
                    # Ensure OtherExpenses is numeric
                    projects_filt['OtherExpenses'] = pd.to_numeric(projects_filt['OtherExpenses'], errors='coerce').fillna(0)

                projects_filt['TotalExpenses'] = projects_filt['ProjectLaborCost'] + projects_filt['OtherExpenses']

                # Recalculate Profit Margin
                projects_filt['ProfitMargin'] = 0.0
                if 'FinalAmount' in projects_filt.columns and 'Status' in projects_filt.columns:
                    projects_filt['FinalAmount'] = pd.to_numeric(projects_filt['FinalAmount'], errors='coerce') # Ensure numeric
                    completed_mask = projects_filt['Status'] == 'Completed'
                    revenue_mask = projects_filt['FinalAmount'].notna() & (projects_filt['FinalAmount'] > 1e-6)
                    valid_denom_mask = projects_filt['FinalAmount'].abs() > 1e-9
                    valid_mask = completed_mask & revenue_mask & valid_denom_mask

                    # Check if 'TotalExpenses' exists before calculation
                    if 'TotalExpenses' in projects_filt.columns:
                        projects_filt.loc[valid_mask, 'ProfitMargin'] = \
                           (projects_filt.loc[valid_mask, 'FinalAmount'] - projects_filt.loc[valid_mask, 'TotalExpenses']) / \
                            projects_filt.loc[valid_mask, 'FinalAmount']
                        projects_filt['ProfitMargin'] = projects_filt['ProfitMargin'].replace([np.inf, -np.inf], 0).fillna(0)
                    else:
                         logger.error("Cannot calculate ProfitMargin: 'TotalExpenses' column missing.")
                else:
                    logger.error("Missing FinalAmount or Status in projects_filt during recalc, cannot calculate ProfitMargin.")


                filtered_data["projects"] = projects_filt
                logger.info("Updated filtered projects with recalculated costs/margins.")
        else:
            logger.info("Filtered projects dataframe is empty, skipping cost/margin recalculation.")


    # --- Update Service Calls ---
    if "service_calls" in filtered_data:
        sc_filt = filtered_data["service_calls"].copy()
        if not sc_filt.empty:
            # --- Pre-checks for service call update ---
            if 'ServiceID' not in sc_filt.columns:
                logger.error("Cannot update service calls: 'ServiceID' column missing.")
            elif 'ServiceCallID' not in time_entries_filt.columns:
                 logger.error("Cannot update service calls: 'ServiceCallID' column missing in time_entries_filt.")
            else:
                # Ensure IDs are string type for aggregation and merge
                sc_filt['ServiceID'] = sc_filt['ServiceID'].astype(str)
                time_entries_filt['ServiceCallID'] = time_entries_filt['ServiceCallID'].astype(str)

                # Aggregate labor from FILTERED time entries (with recalculated LaborCost)
                sc_labor_filt = time_entries_filt[
                    time_entries_filt['ServiceCallID'].notna() & (time_entries_filt['ServiceCallID'] != 'None')
                ].groupby('ServiceCallID')['LaborCost'].sum().reset_index()
                sc_labor_filt.rename(columns={'LaborCost': 'ServiceLaborCost_recalc'}, inplace=True)
                sc_labor_filt['ServiceCallID'] = sc_labor_filt['ServiceCallID'].astype(str) # Ensure type

                # Merge recalculated labor cost
                sc_filt = sc_filt.merge(sc_labor_filt, left_on='ServiceID', right_on='ServiceCallID', how='left')

                # Consolidate ServiceLaborCost
                if 'ServiceLaborCost_recalc' in sc_filt.columns:
                    sc_filt['ServiceLaborCost'] = sc_filt['ServiceLaborCost_recalc'].fillna(sc_filt.get('ServiceLaborCost', 0))
                    sc_filt.drop(columns=['ServiceLaborCost_recalc'], inplace=True)
                if 'ServiceCallID' in sc_filt.columns: # Drop the merge key if present
                    sc_filt = sc_filt.drop(columns=['ServiceCallID'])
                sc_filt['ServiceLaborCost'] = sc_filt['ServiceLaborCost'].fillna(0) # Ensure column exists and NAs are 0

                # Recalculate Total Service Cost
                if 'MaterialsCost' not in sc_filt.columns:
                    logger.warning("'MaterialsCost' missing in service_calls during recalc. Assuming 0.")
                    sc_filt['MaterialsCost'] = 0.0
                else:
                    sc_filt['MaterialsCost'] = pd.to_numeric(sc_filt['MaterialsCost'], errors='coerce').fillna(0)

                # Ensure ServiceLaborCost is numeric before adding
                sc_filt['ServiceLaborCost'] = pd.to_numeric(sc_filt['ServiceLaborCost'], errors='coerce').fillna(0)

                sc_filt['TotalServiceCost'] = sc_filt['ServiceLaborCost'] + sc_filt['MaterialsCost']


                filtered_data["service_calls"] = sc_filt
                logger.info("Updated filtered service calls with recalculated costs.")
        else:
            logger.info("Filtered service_calls dataframe is empty, skipping cost recalculation.")

    return filtered_data