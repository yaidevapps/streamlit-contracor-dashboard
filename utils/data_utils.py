# utils/data_utils.py
import streamlit as st
import pandas as pd
import os
import logging
import numpy as np
from config import DATA_DIR, REQUIRED_FILES, REQUIRED_COLS, DTYPE_MAP, DATE_COLS, NULL_STR_ALLOWED_COLS

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _clean_null_strings(df, allowed_none_cols):
    """
    Replaces common null-like strings ('nan', 'NA', 'N/A', 'TBD', '', None) with pd.NA.
    Handles the string 'None' based on the allowed_none_cols list.
    Operates primarily on object/string columns to avoid errors.
    """
    null_markers = ['nan', 'NA', 'N/A', 'TBD', '', None] # 'TBD' is crucial here
    logger.debug(f"Cleaning null strings. Allowed 'None' columns: {allowed_none_cols}")

    for col in df.columns:
        # Check if column is object or string type before attempting string replacements
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            try:
                # Temporarily store original dtype if it's string, helps later if needed
                is_originally_string = pd.api.types.is_string_dtype(df[col])

                # Replace standard null markers -> pd.NA
                # Use .loc to modify DataFrame directly
                df.loc[:, col] = df[col].replace(null_markers, pd.NA)

                # Handle 'None' string specifically
                if col not in allowed_none_cols:
                    # If 'None' is NOT allowed, replace it with pd.NA
                    df.loc[df[col] == 'None', col] = pd.NA
                # else: 'None' string is allowed, leave it as is (it's already a string)

                # Optional: If column was originally string and now mixed NA/string, ensure consistent type
                if is_originally_string and df[col].isnull().any():
                     # Convert to nullable string type if it contains NAs after cleaning
                     try:
                          # Try converting to pandas' nullable string type
                          df.loc[:, col] = df[col].astype(pd.StringDtype())
                     except Exception as e_astype:
                          logger.warning(f"Could not convert column '{col}' back to StringDtype after cleaning NAs: {e_astype}")


            except TypeError as e_type:
                # This might happen if a column has truly mixed types replace can't handle
                logger.warning(f"TypeError during null string cleaning for column '{col}' (dtype: {df[col].dtype}): {e_type}. Skipping cleaning for this column.")
            except Exception as e_general:
                logger.error(f"Unexpected error during null string cleaning for column '{col}' (dtype: {df[col].dtype}): {e_general}", exc_info=True)

    return df

def _convert_types(df):
    """
    Applies type conversions based on DTYPE_MAP and DATE_COLS AFTER null strings have been cleaned.
    Handles potential errors during conversion gracefully.
    """
    logger.debug("Starting type conversions...")
    for col, target_dtype in DTYPE_MAP.items():
        if col in df.columns:
            current_dtype = df[col].dtype
            logger.debug(f"Processing column '{col}'. Current dtype: {current_dtype}, Target dtype: {target_dtype}")

            # Skip conversion if already the target type (or compatible nullable version)
            if pd.api.types.is_dtype_equal(current_dtype, target_dtype):
                logger.debug(f"Column '{col}' already has target dtype {target_dtype}. Skipping.")
                continue
            # Handle specific nullable types check
            if isinstance(target_dtype, pd.Int64Dtype) and pd.api.types.is_integer_dtype(current_dtype): # Already nullable int
                 if current_dtype.name == 'Int64': # Check specific name
                     logger.debug(f"Column '{col}' already Int64. Skipping.")
                     continue

            try:
                if target_dtype == float:
                    # Use pd.to_numeric, coerce errors (NaN for non-convertibles)
                    df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')
                    # Ensure final type is float64 if conversion happened
                    if not pd.api.types.is_float_dtype(df[col].dtype):
                         df.loc[:, col] = df[col].astype(float)

                elif isinstance(target_dtype, pd.Int64Dtype):
                    # Coerce to numeric first -> NaN for non-convertibles
                    numeric_col = pd.to_numeric(df[col], errors='coerce')
                    # Convert to nullable Int64Dtype (handles NaN -> pd.NA)
                    df.loc[:, col] = numeric_col.astype(pd.Int64Dtype())

                elif target_dtype == str or pd.api.types.is_string_dtype(target_dtype):
                    # Convert to pandas nullable StringDtype for consistency with NAs
                    # Check if already string-like to avoid unnecessary conversion
                    if not (pd.api.types.is_string_dtype(current_dtype) or current_dtype == object):
                        df.loc[:, col] = df[col].astype(pd.StringDtype())
                    # If it's object type, it might already contain strings and NAs, convert to StringDtype
                    elif current_dtype == object:
                        df.loc[:, col] = df[col].astype(pd.StringDtype())

                # Add handling for other specific types (e.g., boolean) if needed
                # elif target_dtype == bool:
                #    # Handle boolean conversion carefully, maybe map specific strings
                #    bool_map = {'Yes': True, 'No': False, 'True': True, 'False': False, 1: True, 0: False}
                #    df.loc[:, col] = df[col].replace(bool_map).astype(bool) # or pd.BooleanDtype()

            except Exception as e:
                logger.error(f"Error converting column '{col}' to target type {target_dtype}. Current dtype: {current_dtype}. Error: {e}. Leaving as is.", exc_info=True)

    # Convert date columns AFTER cleaning and other conversions
    logger.debug("Converting date columns...")
    for col in DATE_COLS:
        if col in df.columns:
            try:
                # Check if not already datetime
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    # 'TBD' etc. should be pd.NA now
                    df.loc[:, col] = pd.to_datetime(df[col], errors='coerce')
                    logger.debug(f"Converted date column '{col}' to datetime.")
            except Exception as e:
                logger.error(f"Error converting date column '{col}' to datetime: {e}. Skipping.", exc_info=True)
    logger.debug("Type conversions finished.")
    return df


def calculate_initial_costs(data):
    """Calculates initial labor costs and merges into time_entries."""
    if "time_entries" in data and "employees" in data:
        logger.info("Calculating initial labor costs...")
        # Ensure dataframes are not empty
        if data["time_entries"].empty or data["employees"].empty:
            logger.warning("Time entries or employees dataframe is empty, cannot calculate initial labor costs.")
            # Ensure LaborCost column exists even if calculation fails
            if "time_entries" in data and 'LaborCost' not in data["time_entries"].columns:
                 data["time_entries"]['LaborCost'] = 0.0
            return data

        time_entries_df = data["time_entries"].copy()
        employees_df = data["employees"].copy()

        # Prepare for merge: Ensure join key 'EmployeeID' is consistent (string) and exists
        if 'EmployeeID' not in time_entries_df.columns or 'EmployeeID' not in employees_df.columns:
            logger.error("'EmployeeID' column missing in time_entries or employees. Cannot merge for cost calculation.")
            if 'LaborCost' not in time_entries_df.columns: time_entries_df['LaborCost'] = 0.0
            data["time_entries"] = time_entries_df
            return data
        time_entries_df['EmployeeID'] = time_entries_df['EmployeeID'].astype(str)
        employees_df['EmployeeID'] = employees_df['EmployeeID'].astype(str)

        # Prepare employee rates (handle potential duplicates, ensure numeric rates)
        if 'HourlyRate' not in employees_df.columns or 'OvertimeRate' not in employees_df.columns:
             logger.error("Rate columns missing in employees dataframe.")
             if 'LaborCost' not in time_entries_df.columns: time_entries_df['LaborCost'] = 0.0
             data["time_entries"] = time_entries_df
             return data
        employees_df['HourlyRate'] = pd.to_numeric(employees_df['HourlyRate'], errors='coerce').fillna(0)
        employees_df['OvertimeRate'] = pd.to_numeric(employees_df['OvertimeRate'], errors='coerce').fillna(0)
        employee_rates = employees_df.drop_duplicates(subset=['EmployeeID']).set_index('EmployeeID')[['HourlyRate', 'OvertimeRate']]

        # Merge rates
        time_entries_df = time_entries_df.merge(employee_rates, on='EmployeeID', how='left', suffixes=('', '_empl'))
        # If merge creates duplicate columns (unlikely with set_index), handle them
        if 'HourlyRate_empl' in time_entries_df.columns:
             time_entries_df['HourlyRate'] = time_entries_df['HourlyRate_empl'].fillna(time_entries_df['HourlyRate'])
             time_entries_df.drop(columns=['HourlyRate_empl'], inplace=True)
        if 'OvertimeRate_empl' in time_entries_df.columns:
             time_entries_df['OvertimeRate'] = time_entries_df['OvertimeRate_empl'].fillna(time_entries_df['OvertimeRate'])
             time_entries_df.drop(columns=['OvertimeRate_empl'], inplace=True)

        # Fill rates potentially missing after merge (e.g., time entry for unknown employee)
        time_entries_df['HourlyRate'] = time_entries_df['HourlyRate'].fillna(0)
        time_entries_df['OvertimeRate'] = time_entries_df['OvertimeRate'].fillna(0)

        # Ensure hours columns exist and are numeric
        if 'RegularHours' not in time_entries_df.columns or 'OvertimeHours' not in time_entries_df.columns:
             logger.error("Hours columns missing in time_entries dataframe.")
             if 'LaborCost' not in time_entries_df.columns: time_entries_df['LaborCost'] = 0.0
             data["time_entries"] = time_entries_df
             return data
        time_entries_df['RegularHours'] = pd.to_numeric(time_entries_df['RegularHours'], errors='coerce').fillna(0)
        time_entries_df['OvertimeHours'] = pd.to_numeric(time_entries_df['OvertimeHours'], errors='coerce').fillna(0)

        # Calculate cost per time entry
        time_entries_df['LaborCost'] = (time_entries_df['RegularHours'] * time_entries_df['HourlyRate']) + \
                                      (time_entries_df['OvertimeHours'] * time_entries_df['OvertimeRate'])
        data["time_entries"] = time_entries_df # Update data dict
        logger.info("Initial LaborCost calculated for time_entries.")
    else:
        logger.warning("Could not calculate initial labor costs: time_entries or employees data missing or empty.")
        # Ensure LaborCost column exists even if calculation fails
        if "time_entries" in data and 'LaborCost' not in data["time_entries"].columns:
             data["time_entries"]['LaborCost'] = 0.0
    return data

def aggregate_initial_costs(data):
    """Aggregates initial costs (labor, other expenses) into projects and service calls."""
    # --- Aggregate Project Costs ---
    if "time_entries" in data and "expenses" in data and "projects" in data:
        logger.info("Aggregating initial project costs...")
        # Ensure required source dataframes are not empty
        if data["time_entries"].empty or data["expenses"].empty or data["projects"].empty:
            logger.warning("One or more source dataframes (time_entries, expenses, projects) are empty. Cannot aggregate project costs.")
            # Ensure target columns exist in projects df
            if "projects" in data:
                 for col in ['ProjectLaborCost', 'OtherExpenses', 'TotalExpenses', 'ProfitMargin']:
                      if col not in data['projects'].columns: data['projects'][col] = 0.0
            return data

        time_df = data["time_entries"]
        exp_df = data["expenses"]
        proj_df = data["projects"].copy()

        # --- Pre-aggregation checks ---
        if 'ProjectID' not in time_df.columns or 'LaborCost' not in time_df.columns:
            logger.error("Missing 'ProjectID' or 'LaborCost' in time_entries for project cost aggregation.")
            return data # Cannot proceed without these
        if 'RelatedProjectID' not in exp_df.columns or 'Amount' not in exp_df.columns:
            logger.error("Missing 'RelatedProjectID' or 'Amount' in expenses for project cost aggregation.")
            return data # Cannot proceed without these
        if 'ProjectID' not in proj_df.columns:
            logger.error("Missing 'ProjectID' in projects for merging aggregated costs.")
            return data # Cannot proceed without this

        # Aggregate Project Labor Cost (ensure ProjectID is str and handle NAs before grouping)
        time_df['ProjectID'] = time_df['ProjectID'].astype(str)
        proj_labor = time_df[time_df['ProjectID'].notna() & (time_df['ProjectID'] != 'None')].groupby('ProjectID')['LaborCost'].sum().reset_index()
        proj_labor.rename(columns={'LaborCost': 'ProjectLaborCost'}, inplace=True)
        proj_labor['ProjectID'] = proj_labor['ProjectID'].astype(str) # Ensure type after grouping


        # Aggregate Other Expenses (ensure RelatedProjectID is str and handle NAs)
        exp_df['RelatedProjectID'] = exp_df['RelatedProjectID'].astype(str)
        other_exp = exp_df[exp_df['RelatedProjectID'].notna() & (exp_df['RelatedProjectID'] != 'None')].groupby('RelatedProjectID')['Amount'].sum().reset_index()
        other_exp.rename(columns={'Amount': 'OtherExpenses', 'RelatedProjectID': 'ProjectID'}, inplace=True)
        other_exp['ProjectID'] = other_exp['ProjectID'].astype(str) # Ensure type after grouping


        # Merge into Projects (ensure ProjectID is str in proj_df)
        proj_df['ProjectID'] = proj_df['ProjectID'].astype(str)
        proj_df = proj_df.merge(proj_labor, on='ProjectID', how='left', suffixes=('', '_labor'))
        proj_df = proj_df.merge(other_exp, on='ProjectID', how='left', suffixes=('', '_other'))

        # Handle potential duplicate columns after merge, prioritize the newly calculated ones
        if 'ProjectLaborCost_labor' in proj_df.columns:
             proj_df['ProjectLaborCost'] = proj_df['ProjectLaborCost_labor'].fillna(proj_df.get('ProjectLaborCost', 0))
             proj_df.drop(columns=['ProjectLaborCost_labor'], inplace=True)
        if 'OtherExpenses_other' in proj_df.columns:
             proj_df['OtherExpenses'] = proj_df['OtherExpenses_other'].fillna(proj_df.get('OtherExpenses', 0))
             proj_df.drop(columns=['OtherExpenses_other'], inplace=True)

        # Fill NaNs introduced by merge with 0
        proj_df['ProjectLaborCost'] = proj_df['ProjectLaborCost'].fillna(0)
        proj_df['OtherExpenses'] = proj_df['OtherExpenses'].fillna(0)

        # Calculate Total Expenses
        proj_df['TotalExpenses'] = proj_df['ProjectLaborCost'] + proj_df['OtherExpenses']

        # Calculate Initial Profit Margin (for Completed only, check FinalAmount exists)
        proj_df['ProfitMargin'] = 0.0
        if 'FinalAmount' not in proj_df.columns or 'Status' not in proj_df.columns:
             logger.error("Missing 'FinalAmount' or 'Status' in projects, cannot calculate profit margin.")
        else:
            completed_mask = proj_df['Status'] == 'Completed'
            # Use already coerced FinalAmount
            revenue_mask = proj_df['FinalAmount'].notna() & (proj_df['FinalAmount'] > 1e-6)
            valid_denom_mask = proj_df['FinalAmount'].abs() > 1e-9
            valid_mask = completed_mask & revenue_mask & valid_denom_mask

            proj_df.loc[valid_mask, 'ProfitMargin'] = \
                (proj_df.loc[valid_mask, 'FinalAmount'] - proj_df.loc[valid_mask, 'TotalExpenses']) / \
                 proj_df.loc[valid_mask, 'FinalAmount']
            proj_df['ProfitMargin'] = proj_df['ProfitMargin'].replace([np.inf, -np.inf], 0).fillna(0)

        data["projects"] = proj_df
        logger.info("Initial costs/margin aggregated into projects.")
    else:
        logger.warning("Could not aggregate initial project costs: One or more required dataframes missing.")
        # Ensure relevant columns exist in projects df even if calc fails
        if "projects" in data:
            for col in ['ProjectLaborCost', 'OtherExpenses', 'TotalExpenses', 'ProfitMargin']:
                if col not in data['projects'].columns: data['projects'][col] = 0.0

    # --- Aggregate Service Call Costs ---
    if "time_entries" in data and "service_calls" in data:
        logger.info("Aggregating initial service call costs...")
        if data["time_entries"].empty or data["service_calls"].empty:
            logger.warning("Time entries or service calls dataframe is empty, cannot aggregate service call costs.")
            if "service_calls" in data:
                 for col in ['ServiceLaborCost', 'TotalServiceCost']:
                      if col not in data['service_calls'].columns: data['service_calls'][col] = 0.0
            return data

        time_df = data["time_entries"]
        sc_df = data["service_calls"].copy()

        # --- Pre-aggregation checks ---
        if 'ServiceCallID' not in time_df.columns or 'LaborCost' not in time_df.columns:
            logger.error("Missing 'ServiceCallID' or 'LaborCost' in time_entries for service call cost aggregation.")
            return data
        if 'ServiceID' not in sc_df.columns:
             logger.error("Missing 'ServiceID' in service_calls for merging aggregated costs.")
             return data


        # Aggregate Service Labor Cost (ensure ServiceCallID is str and handle NAs)
        time_df['ServiceCallID'] = time_df['ServiceCallID'].astype(str)
        sc_labor = time_df[time_df['ServiceCallID'].notna() & (time_df['ServiceCallID'] != 'None')].groupby('ServiceCallID')['LaborCost'].sum().reset_index()
        sc_labor.rename(columns={'LaborCost': 'ServiceLaborCost'}, inplace=True)
        sc_labor['ServiceCallID'] = sc_labor['ServiceCallID'].astype(str) # Ensure type after grouping

        # Merge into Service Calls (ensure ServiceID is str in sc_df)
        sc_df['ServiceID'] = sc_df['ServiceID'].astype(str)
        sc_df = sc_df.merge(sc_labor, left_on='ServiceID', right_on='ServiceCallID', how='left', suffixes=('', '_labor'))

        # Handle potential duplicate columns and fill NAs
        if 'ServiceLaborCost_labor' in sc_df.columns:
             sc_df['ServiceLaborCost'] = sc_df['ServiceLaborCost_labor'].fillna(sc_df.get('ServiceLaborCost', 0))
             sc_df.drop(columns=['ServiceLaborCost_labor'], inplace=True)
        if 'ServiceCallID' in sc_df.columns: # Drop the merge key
             sc_df = sc_df.drop(columns=['ServiceCallID'])
        sc_df['ServiceLaborCost'] = sc_df['ServiceLaborCost'].fillna(0)

        # Calculate Total Service Cost (Labor + Materials from service_calls.txt)
        # Ensure MaterialsCost exists and is numeric
        if 'MaterialsCost' not in sc_df.columns:
             logger.warning("'MaterialsCost' column missing in service_calls. Assuming 0 for TotalServiceCost calculation.")
             sc_df['MaterialsCost'] = 0.0
        else:
             sc_df['MaterialsCost'] = pd.to_numeric(sc_df['MaterialsCost'], errors='coerce').fillna(0)

        sc_df['TotalServiceCost'] = sc_df['ServiceLaborCost'] + sc_df['MaterialsCost']

        data["service_calls"] = sc_df
        logger.info("Initial costs aggregated into service calls.")
    else:
        logger.warning("Could not aggregate initial service call costs: time_entries or service_calls data missing.")
        if "service_calls" in data:
            for col in ['ServiceLaborCost', 'TotalServiceCost']:
                if col not in data['service_calls'].columns: data['service_calls'][col] = 0.0

    return data


@st.cache_data(ttl=3600) # Cache data for 1 hour
def load_all_data():
    """Loads, cleans, and performs initial calculations on all required data files."""
    data = {}
    all_files_found = True
    logger.info(f"Checking for required files in: {DATA_DIR}")
    for filename in REQUIRED_FILES:
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            st.error(f"Required data file not found: {path}")
            logger.error(f"Required data file not found: {path}")
            all_files_found = False
    if not all_files_found:
        st.error("One or more required data files are missing. Please ensure all data files are present in the 'contractor_data' directory.")
        return None # Stop if essential files are missing

    logger.info("Starting data load process...")
    loaded_keys = []
    for key in REQUIRED_COLS.keys(): # Use keys from required cols dict as the master list
        filename = f"{key}.txt"
        path = os.path.join(DATA_DIR, filename)
        # Double check existence, although checked above
        if not os.path.exists(path):
            logger.warning(f"File {filename} confirmed missing during load loop. Skipping.")
            data[key] = pd.DataFrame(columns=REQUIRED_COLS.get(key, [])) # Create empty placeholder
            continue

        try:
            logger.info(f"Loading {filename}...")

            # --- Read CSV: Only enforce STR dtypes initially ---
            # Identify potential ID columns present in this specific file
            header_df = pd.read_csv(path, delimiter="|", nrows=0)
            actual_columns = header_df.columns.tolist()
            # Define dtypes only for ID columns actually present
            id_cols_in_file = [k for k, v in DTYPE_MAP.items() if k in actual_columns and (v == str or pd.api.types.is_string_dtype(v))]
            read_dtypes = {col: str for col in id_cols_in_file} # Read all potential ID cols as string
            logger.debug(f"Reading {filename} with specific string dtypes for: {list(read_dtypes.keys())}")
            # --- End dtype modification ---

            df = pd.read_csv(
                path,
                delimiter="|",
                on_bad_lines='warn',
                dtype=read_dtypes, # Use the restricted dtype map for reading
                low_memory=False,
                skipinitialspace=True
                )
            logger.info(f"Loaded {filename} with {len(df)} rows. Columns: {df.columns.tolist()}")
            loaded_keys.append(key)

            # --- Basic Column Check ---
            missing_req_cols = [col for col in REQUIRED_COLS.get(key, []) if col not in df.columns]
            if missing_req_cols:
                 logger.error(f"Missing required columns in {key}.txt: {', '.join(missing_req_cols)}. Functionality may be limited.")
                 st.warning(f"Missing required columns in {key}.txt: {', '.join(missing_req_cols)}. Results may be affected.")
                 # Add missing columns as NA to prevent downstream errors, though data is incomplete
                 for col in missing_req_cols:
                      df[col] = pd.NA


            # --- Clean and Convert Types ---
            logger.debug(f"Starting cleaning and type conversion for {key}...")
            # 1. Clean problematic strings ('TBD', 'N/A', etc.) -> pd.NA
            df = _clean_null_strings(df, NULL_STR_ALLOWED_COLS)
            # 2. Convert types based on DTYPE_MAP (now handles pd.NA correctly)
            df = _convert_types(df)
            logger.debug(f"Finished cleaning and type conversion for {key}.")


            data[key] = df

        except pd.errors.EmptyDataError:
            logger.warning(f"File {path} is empty. Creating empty DataFrame for key '{key}'.")
            st.warning(f"Data file '{filename}' is empty.")
            data[key] = pd.DataFrame(columns=REQUIRED_COLS.get(key, [])) # Create empty df with expected columns
        except Exception as e:
            logger.error(f"FATAL: Failed to load or process {path}: {e}", exc_info=True)
            st.error(f"An critical error occurred loading {key}.txt: {e}. Cannot proceed.")
            return None # Fatal error during loading

    # --- Post-Load Calculations ---
    logger.info(f"Successfully loaded base data for keys: {loaded_keys}")
    # Check if essential data for calculations was loaded
    if "time_entries" in data and "employees" in data:
        data = calculate_initial_costs(data)
    else:
        logger.error("Cannot calculate initial costs because 'time_entries' or 'employees' data failed to load.")
        # Ensure cost columns exist even if calculation fails
        if "time_entries" in data and 'LaborCost' not in data["time_entries"].columns:
             data["time_entries"]['LaborCost'] = 0.0

    if "projects" in data and "expenses" in data and "time_entries" in data and \
       "service_calls" in data:
         data = aggregate_initial_costs(data)
    else:
         logger.error("Cannot aggregate initial costs because one or more required dataframes (projects, expenses, time_entries, service_calls) failed to load.")
         # Ensure relevant columns exist even if aggregation fails
         if "projects" in data:
            for col in ['ProjectLaborCost', 'OtherExpenses', 'TotalExpenses', 'ProfitMargin']:
                 if col not in data['projects'].columns: data['projects'][col] = 0.0
         if "service_calls" in data:
            for col in ['ServiceLaborCost', 'TotalServiceCost']:
                 if col not in data['service_calls'].columns: data['service_calls'][col] = 0.0


    logger.info("Data loading and initial processing complete.")
    # Final check: ensure all expected keys exist, even if loading failed for some
    for key in REQUIRED_COLS.keys():
        if key not in data:
             logger.warning(f"Key '{key}' missing from final data dict, likely due to loading error. Creating empty DataFrame.")
             data[key] = pd.DataFrame(columns=REQUIRED_COLS.get(key, []))

    return data