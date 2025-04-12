# config.py
import os
from datetime import datetime
import pandas as pd # Ensure pandas is imported

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "contractor_data")
CSS_FILE = os.path.join(BASE_DIR, "styles.css")

# --- Dates ---
# Use a consistent 'current' date for analysis across the app
CURRENT_DATE = datetime(2025, 4, 9) # Use the date from data_generator

# --- Other Settings ---
# Example: Could add default filter dates, AI model names, etc.
DEFAULT_FILTER_START_DATE = datetime(2023, 1, 1)
DEFAULT_FILTER_END_DATE = CURRENT_DATE # Use config date

# Define required files for data loading validation
REQUIRED_FILES = [
    "projects.txt", "service_calls.txt", "invoices.txt", "expenses.txt",
    "bids.txt", "employees.txt", "time_entries.txt", "customers.txt",
    "project_timelines.txt", "change_orders.txt", "warranty_claims.txt",
    "marketing_campaigns.txt", "inventory.txt", "equipment.txt", "suppliers.txt"
]

# --- MODIFIED REQUIRED_COLS ---
# List only columns expected *directly* from the source files for initial validation.
# Calculated columns (like TotalExpenses, ProjectLaborCost etc.) are removed from this check.
REQUIRED_COLS = {
    "projects": ['ProjectID', 'CustomerID', 'ProjectType', 'Status', 'StartDateStr', 'City', 'FinalAmount', 'BidAmount', 'CompletionDateStr', 'ActualHours', 'EstimatedCompletionDate', 'DelayImpactHours', 'ConstructionType', 'SquareFootage', 'ProfitMargin', 'CostOverrunAmount', 'Warranty', 'ServicePlan', 'Neighborhood', 'ArchitecturalFirm', 'GeneralContractor', 'BidWinDate', 'EstimatedHours', 'DelayReason', 'CustomerSatisfaction'], # Removed calculated cost columns. Kept raw ProfitMargin as it's in the file (even if TBD).
    "service_calls": ['ServiceID', 'CustomerID', 'ServiceType', 'RelatedProjectID', 'TechnicianID', 'DateStr', 'StartTime', 'EndTime', 'MaterialsCost', 'LaborHours', 'LaborCost', 'TotalCost', 'BilledAmount', 'SatisfactionScore', 'Priority', 'Resolution', 'ScheduledDate', 'ScheduledTime', 'ActualArrivalTime', 'ResponseTime', 'IssueCategory', 'FirstTimeFixStatus'], # Removed calculated ServiceLaborCost, TotalServiceCost. Kept raw LaborCost/TotalCost as they are in the file.
    "invoices": ['InvoiceID', 'ProjectID', 'CustomerID', 'RelatedServiceID', 'InvoiceDate', 'DueDate', 'Amount', 'Status', 'PaymentDate', 'PaymentMethod'],
    "expenses": ['ExpenseID', 'Category', 'Amount', 'Date', 'RelatedProjectID', 'Description', 'ApprovedBy', 'PaymentMethod', 'ExpenseStatus'],
    "bids": ['BidID', 'CustomerID', 'BidDate', 'ProjectType', 'BidAmount', 'Status', 'RelatedProjectID', 'DeclineReason', 'CompetitorPrice', 'SalesPersonID'],
    "employees": ['EmployeeID', 'Name', 'Address', 'City', 'ZIP', 'Phone', 'Email', 'Role', 'HireDate', 'HourlyRate', 'OvertimeRate', 'Certifications', 'LicenseExpirations', 'Languages'],
    "time_entries": ['EntryID', 'EmployeeID', 'Date', 'ServiceCallID', 'ProjectID', 'Category', 'RegularHours', 'OvertimeHours', 'TotalHours', 'Notes'], # Removed calculated LaborCost
    "customers": ['CustomerID', 'Name', 'Address', 'City', 'Neighborhood', 'ZIP', 'Phone', 'Email', 'CustomerType', 'ReferralSource', 'DateAcquired', 'HomeValue', 'PreferredContactMethod', 'AnnualSpend', 'RepeatCustomerStatus'],
    "project_timelines": ['TimelineEntryID', 'ProjectID', 'Milestone', 'EstimatedDate', 'ActualDate', 'Status', 'Notes'],
    "change_orders": ['ChangeOrderID', 'ProjectID', 'RequestDate', 'Type', 'Description', 'OriginalSpec', 'NewSpec', 'Amount', 'Status', 'TimelineImpactDays', 'ApprovedBy'],
    "warranty_claims": ['ClaimID', 'ProjectID', 'CustomerID', 'ClaimDate', 'IssueType', 'Description', 'ResolutionDate', 'ResolutionTimeDays', 'CostToFix', 'CoveredByWarranty', 'CustomerCost', 'TechnicianID', 'PreventativeMeasures'],
    "marketing_campaigns": ['CampaignID', 'Name', 'Type', 'StartDate', 'EndDate', 'Budget', 'LeadsGenerated', 'CustomersAcquired', 'RevenueGenerated', 'ROI', 'CostPerLead', 'TargetNeighborhoods'],
    "inventory": ['ItemID', 'Category', 'Manufacturer', 'Model', 'Description', 'CurrentStock', 'ReorderLevel', 'UnitCost', 'MarkupPercentage', 'SupplierID', 'LastOrderDate', 'LeadTime', 'Discontinued'],
    "equipment": ['AssetID', 'Type', 'Make', 'Model', 'PurchaseDate', 'Cost', 'MaintenanceSchedule', 'CurrentStatus', 'AssignedEmployeeID', 'CustomFeatures'],
    "suppliers": ['SupplierID', 'Name', 'ContactName', 'ContactPhone', 'ContactEmail', 'PaymentTerms', 'PreferredStatus', 'ProductCategories', 'AuthorizedDealerFor', 'MinimumOrder', 'LastOrderDate', 'LeadTime', 'Discontinued']
}
# --- END MODIFIED REQUIRED_COLS ---


# --- MODIFIED DTYPE_MAP ---
# Consistent data types for IDs and key numeric/date fields expected directly from files.
# Types for calculated columns are removed/commented out here.
DTYPE_MAP = {
    # IDs (Always String)
    'ProjectID': str, 'CustomerID': str, 'ServiceID': str, 'RelatedProjectID': str,
    'RelatedServiceID': str, 'EmployeeID': str, 'ExpenseID': str, 'BidID': str,
    'EntryID': str, 'InvoiceID': str, 'TimelineEntryID': str, 'ClaimID': str,
    'TrainingID': str, 'AssetID': str, 'SupplierID': str, 'ItemID': str,
    'CampaignID': str, 'ChangeOrderID': str, 'SalesPersonID': str,
    'ApprovedBy': str, 'TechnicianID': str, 'AssignedEmployeeID': str,

    # Key Numeric (Float or Int as appropriate, handle errors)
    'Amount': float, 'FinalAmount': float, 'BidAmount': float, 'CostToFix': float,
    'CustomerCost': float, 'Budget': float, 'RevenueGenerated': float, 'ROI': float,
    'CostPerLead': float, 'HourlyRate': float, 'OvertimeRate': float,
    'RegularHours': float, 'OvertimeHours': float, 'TotalHours': float,
    'LaborHours': float, 'MaterialsCost': float, 'UnitCost': float, 'Cost': float,
    'SquareFootage': float, 'DelayImpactHours': float, 'ActualHours': float,
    'EstimatedHours': float, 'AnnualSpend': float,
    'ProfitMargin': float, # Keep raw PM from projects.txt (will be object initially if 'TBD')
    'LaborCost': float, # Keep raw LaborCost from service_calls.txt (will be object initially if 'TBD')
    'TotalCost': float, # Keep raw TotalCost from service_calls.txt (will be object initially if 'TBD')
    'CostOverrunAmount': float,
    'MarkupPercentage': float,
    # Calculated fields removed from DTYPE map for initial load:
    # 'ProjectLaborCost': float,
    # 'OtherExpenses': float,
    # 'TotalExpenses': float,
    # 'ServiceLaborCost': float,
    # 'TotalServiceCost': float,
    # 'RestockCost': float, # This wasn't defined previously but is often calculated

    # Integers (handle errors)
    'CustomersAcquired': pd.Int64Dtype(), 'LeadsGenerated': pd.Int64Dtype(),
    'CurrentStock': pd.Int64Dtype(), 'ReorderLevel': pd.Int64Dtype(),
    'TimelineImpactDays': pd.Int64Dtype(),
    'ResolutionTimeDays': float, # Read as float first to handle NA, then maybe convert if needed

    # Explicitly define others if needed, otherwise pandas infers
    'Status': str, 'Category': str, 'ProjectType': str, 'City': str,
    'Neighborhood': str, 'ExpenseStatus': str, # Ensure these common text fields are strings
    'Warranty': str, 'ServicePlan': str, 'ConstructionType': str,
    'ArchitecturalFirm': str, 'GeneralContractor': str, 'DelayReason': str,
    'CustomerSatisfaction': str, # Keep as string if it can be 'N/A'
    'ServiceType': str, 'Resolution': str, 'IssueCategory': str, 'FirstTimeFixStatus': str,
    'PaymentMethod': str, 'DeclineReason': str, 'Name': str, 'Address': str, 'ZIP': str,
    'Phone': str, 'Email': str, 'Role': str, 'Certifications': str, 'Languages': str,
    'CustomerType': str, 'ReferralSource': str, 'HomeValue': str, 'PreferredContactMethod': str,
    'RepeatCustomerStatus': str, 'Milestone': str, 'Notes': str, 'Type': str, 'Description': str,
    'OriginalSpec': str, 'NewSpec': str, 'CoveredByWarranty': str, 'PreventativeMeasures': str,
    'Model': str, 'MaintenanceSchedule': str, 'CurrentStatus': str, 'CustomFeatures': str,
    'ContactName': str, 'ContactPhone': str, 'ContactEmail': str, 'PaymentTerms': str,
    'PreferredStatus': str, 'ProductCategories': str, 'AuthorizedDealerFor': str,
    'MinimumOrder': str, 'LeadTime': str, 'Discontinued': str,
}
# --- END MODIFIED DTYPE_MAP ---

# Columns expected to contain dates
DATE_COLS = [
    'DateAcquired', 'HireDate', 'LicenseExpirations', 'LastOrderDate', 'PurchaseDate',
    'StartDateStr', 'CompletionDateStr', 'BidWinDate', 'EstimatedCompletionDate',
    'Date', 'DateStr', 'InvoiceDate', 'DueDate', 'PaymentDate', 'StartDate', 'EndDate',
    'EstimatedDate', 'ActualDate', 'RequestDate', 'ClaimDate', 'ResolutionDate',
]

# Columns allowed to have 'None' or similar null-like string values (validated in data_utils)
NULL_STR_ALLOWED_COLS = [
    'RelatedProjectID', 'RelatedServiceID', 'AssignedEmployeeID', 'PaymentDate',
    'PaymentMethod', 'CompletionDateStr', 'ActualHours', 'DelayReason',
    'CustomerSatisfaction', 'FinalAmount', 'ApprovedBy', 'CompetitorPrice',
    'ActualDate', 'Notes', 'CompletionDate', 'Score', 'CertificationRenewed',
    'ProjectID', # Allow 'None' in invoices/time_entries specifically
    'ServiceCallID' # Allow 'None' in time_entries
]

# Roles considered 'billable' for utilization calculations
BILLABLE_ROLES = [
    'Integration Specialist', 'System Designer', 'Project Engineer', 'Lead Technician',
    'Senior Programmer', 'Lighting Control Specialist', 'Audio/Video Engineer',
    'Network Engineer', 'Security Integration Specialist'
]
# Realistic annual hours per employee for utilization capacity calculation
HOURS_PER_EMPLOYEE_PER_YEAR = 1900

# --- AI Configuration ---
AI_MODEL_NAME = "gemini-1.5-flash"
AI_FALLBACK_MESSAGE = "Unable to generate insight due to an issue. Please try again or refine your query."