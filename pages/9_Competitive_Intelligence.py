# pages/9_Competitive_Intelligence.py
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import logging

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

# --- Custom Prompt for Pricing Suggestions (Specific to this page) ---
# Keep this custom prompt as it fulfills a unique requirement
PRICING_SUGGESTIONS_PROMPT = """
You are a Competitive Intelligence Specialist with 25+ years of experience analyzing market positioning for luxury residential contractors in Washington State.

CONTEXT:
{context}

TASK:
Provide specific pricing strategy recommendations to improve bid win rates while maintaining appropriate profit margins for a high-end low voltage contractor based on the provided competitive context.

REQUIREMENTS:
- Suggest 3 specific, actionable pricing adjustments based *only* on the data in the CONTEXT section.
- For each suggestion, include potential implementation steps and expected outcomes (e.g., impact on win rate or perceived value).
- Focus on the luxury residential market in Seattle/Eastside.
- Consider competitor pricing patterns (if available in context) and project types (if available in context).

RESPONSE FORMAT:
Based on your competitive positioning (Win Rate: {win_rate:.1f}%, Avg Bid: ${avg_bid_amount:,.0f}, Avg Competitor Price: {avg_comp_price_str}), consider these pricing strategy adjustments:

1.  **Strategy 1:** [Description of strategy based on context data, e.g., Adjust pricing for 'Specific Project Type' where win rate is low/high]
    *   *Implementation:* [e.g., Conduct focused analysis on lost bids for this type, identify price sensitivity threshold]
    *   *Expected Outcome:* [e.g., Increase win rate for this type by X% without significantly impacting margin, Improve competitiveness in bids exceeding competitor prices by Y%]

2.  **Strategy 2:** [Description of strategy, e.g., Introduce value-based pricing for bids significantly below competitor prices]
    *   *Implementation:* [e.g., Develop standard value-add packages (extended warranty, premium support) to justify higher price points on certain bids]
    *   *Expected Outcome:* [e.g., Capture higher margin on bids where price isn't the primary deciding factor, increase average revenue per won bid]

3.  **Strategy 3:** [Description of strategy, e.g., Analyze competitor pricing data availability]
    *   *Implementation:* [e.g., Implement a system to consistently track competitor pricing when available on lost bids, refine data collection]
    *   *Expected Outcome:* [e.g., Enable more data-driven pricing decisions, identify specific competitors' pricing strategies]
"""


# --- Page Setup and Styling ---
st.set_page_config(layout="wide")
try:
    with open(CSS_FILE) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    logger.info("CSS loaded successfully.")
except FileNotFoundError:
    st.warning("CSS file not found. Using default styles.")

st.title("Competitive Intelligence Dashboard")

# --- AI Client Initialization ---
client = get_genai_client()
if not client:
    st.error("AI Client failed to initialize. AI features will be unavailable.")

# --- Data Loading ---
data_orig = load_all_data()
if data_orig is None:
    st.error("Fatal error loading data. Dashboard cannot be displayed.")
    st.stop()

# Prepare employee rates (standard pattern, not directly used but needed for recalc step)
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
# Note: Filters primarily affect which bids/projects are shown
filtered_data = apply_filters(data_orig, start_date, end_date, city, project_type)
if not filtered_data:
    st.error("Error applying filters or no data matches filters. Cannot display filtered data.")
    st.stop()

# Recalculate costs (standard step, minimal effect here)
filtered_data = recalculate_costs_after_filtering(filtered_data, employee_rates)

# --- Extract Filtered DataFrames for this Page ---
bids_final = filtered_data.get('bids', pd.DataFrame())
projects_final = filtered_data.get('projects', pd.DataFrame()) # Used for context

# --- Calculate Competitive KPIs (Using Filtered Data) ---
win_rate = 0.0
avg_bid_amount = 0.0
avg_comp_price = 0.0
comp_price_str = "N/A"
num_bids_with_comp_price = 0

if not bids_final.empty:
    # Bid Win Rate
    if 'Status' in bids_final.columns:
        won_bids = (bids_final["Status"] == "Won").sum()
        lost_bids = (bids_final["Status"] == "Lost").sum()
        total_submitted = won_bids + lost_bids
        if total_submitted > 0:
            win_rate = (won_bids / total_submitted) * 100

    # Average Bid Amount
    if 'BidAmount' in bids_final.columns:
        bids_final['BidAmount'] = pd.to_numeric(bids_final['BidAmount'], errors='coerce')
        avg_bid_amount = bids_final['BidAmount'].mean()
        if pd.isna(avg_bid_amount): avg_bid_amount = 0.0

    # Average Competitor Price (only where available)
    if 'CompetitorPrice' in bids_final.columns:
        bids_final['CompetitorPrice'] = pd.to_numeric(bids_final['CompetitorPrice'], errors='coerce')
        comp_price_available = bids_final.dropna(subset=['CompetitorPrice'])
        num_bids_with_comp_price = len(comp_price_available)
        if num_bids_with_comp_price > 0:
            avg_comp_price = comp_price_available['CompetitorPrice'].mean()
            comp_price_str = f"${avg_comp_price:,.0f}" # Format string only if calculated


# --- Render Competitive Intelligence Sections ---

# Competitive Summary
st.header("📋 Competitive Summary")
st.markdown("Key competitive metrics based on the selected global filters.")
col1, col2, col3 = st.columns(3)
col1.metric("Filtered Bids Analyzed", len(bids_final))
col2.metric("Bid Win Rate", f"{win_rate:.1f}%")
col3.metric("Average Bid Amount", f"${avg_bid_amount:,.0f}")
col3.metric(f"Avg. Competitor Price (where available, {num_bids_with_comp_price} bids)", comp_price_str)


# AI Competitive Advisor
st.subheader("💡 AI Competitive Strategy Advisor")
comp_summary_context = (
    f"Competitive Snapshot (Filtered Period: {start_date.strftime('%Y-%m-%d') if start_date else 'N/A'} to {end_date.strftime('%Y-%m-%d') if end_date else 'N/A'}, City: {city}, Type: {project_type}):\n"
    f"- Bids Analyzed: {len(bids_final)}\n"
    f"- Bid Win Rate: {win_rate:.1f}%\n"
    f"- Average Bid Amount: ${avg_bid_amount:,.0f}\n"
    f"- Average Competitor Price: {comp_price_str} (based on {num_bids_with_comp_price} bids with data)\n"
)
render_ai_explanation(
    "Get AI-Powered Competitive Advice",
    client,
    SUMMARY_ADVICE_PROMPT, # Using the general advice prompt
    comp_summary_context,
    additional_format_args={'current_date': CURRENT_DATE.strftime('%Y-%m-%d')}
)

st.divider()

# --- Section: Bid Pricing Analysis ---
st.header("📊 Bid Pricing vs. Competitor Analysis")
st.markdown("Scatter plot comparing your bid amounts against recorded competitor prices, colored by bid status.")

if bids_final.empty or num_bids_with_comp_price == 0:
    st.warning("No bid data with competitor pricing available for the selected filters to display this analysis.")
else:
    try:
        # Use only bids where competitor price is known for the scatter plot
        pricing_plot_df = bids_final.dropna(subset=['BidAmount', 'CompetitorPrice', 'Status'])

        if not pricing_plot_df.empty:
            fig_pricing = px.scatter(
                pricing_plot_df,
                x="BidAmount",
                y="CompetitorPrice",
                color="Status",
                title="Your Bid Amount vs. Competitor Price (Filtered)",
                hover_data=["ProjectType", "BidID", "DeclineReason"], # Added BidID
                labels={'BidAmount': 'Your Bid Amount ($)', 'CompetitorPrice': 'Competitor Price ($)'},
                 color_discrete_map={"Won": "green", "Lost": "red", "Pending":"grey", "Withdrawn":"lightblue"} # Custom colors
            )
            # Add a reference line (y=x)
            min_val = min(pricing_plot_df['BidAmount'].min(), pricing_plot_df['CompetitorPrice'].min())
            max_val = max(pricing_plot_df['BidAmount'].max(), pricing_plot_df['CompetitorPrice'].max())
            fig_pricing.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(color="Gray", width=1, dash="dash"), name="Equal Price Line")

            st.plotly_chart(fig_pricing, use_container_width=True)

            # AI Explanation for Pricing Analysis
            won_higher = len(pricing_plot_df[(pricing_plot_df['Status'] == 'Won') & (pricing_plot_df['BidAmount'] > pricing_plot_df['CompetitorPrice'])])
            lost_lower = len(pricing_plot_df[(pricing_plot_df['Status'] == 'Lost') & (pricing_plot_df['BidAmount'] < pricing_plot_df['CompetitorPrice'])])
            pricing_context = (
                f"Bid vs. competitor price analysis based on {len(pricing_plot_df)} filtered bids with competitor data.\n"
                f"- Average Bid Amount (this subset): ${pricing_plot_df['BidAmount'].mean():,.0f}\n"
                f"- Average Competitor Price (this subset): ${pricing_plot_df['CompetitorPrice'].mean():,.0f}\n"
                f"- Bids Won Higher than Competitor: {won_higher}\n"
                f"- Bids Lost Lower than Competitor: {lost_lower}\n"
                f"Plot shows your bid vs competitor price, colored by status. Dashed line indicates equal pricing."
            )
            render_ai_explanation("AI Analysis of Bid vs. Competitor Pricing", client, VISUAL_EXPLANATION_PROMPT, pricing_context)
        else:
             st.info("No bids remaining after filtering for valid bid amounts, competitor prices, and status.")

    except Exception as e:
        logger.error(f"Error generating bid pricing analysis: {e}", exc_info=True)
        st.error("Could not generate the bid pricing analysis visual.")


st.divider()

# --- Section: Loss Reason Analysis ---
st.header("📈 Loss Reason Analysis")
st.markdown("Distribution of reasons cited for losing bids based on filtered data.")

lost_bids_df = bids_final[(bids_final["Status"] == "Lost") & (bids_final["DeclineReason"].notna())] if not bids_final.empty else pd.DataFrame()

if lost_bids_df.empty:
    st.warning("No lost bids with recorded decline reasons available for the selected filters.")
else:
    try:
        loss_reason_counts = lost_bids_df['DeclineReason'].value_counts().reset_index()
        loss_reason_counts.columns = ['DeclineReason', 'Count']

        if not loss_reason_counts.empty:
            fig_loss = px.pie(
                loss_reason_counts,
                names="DeclineReason",
                values="Count",
                title="Reasons for Lost Bids (Filtered)"
            )
            st.plotly_chart(fig_loss, use_container_width=True)

            # AI Explanation for Loss Reasons
            top_reason = loss_reason_counts.iloc[0]['DeclineReason']
            loss_context = (
                f"Analysis of {len(lost_bids_df)} lost bids with decline reasons (filtered).\n"
                f"- Total unique reasons cited: {loss_reason_counts['DeclineReason'].nunique()}\n"
                f"- Most common reason: '{top_reason}' ({loss_reason_counts.iloc[0]['Count']} bids).\n"
                f"Overall Win Rate (Filtered): {win_rate:.1f}%."
            )
            render_ai_explanation("AI Analysis of Loss Reasons", client, VISUAL_EXPLANATION_PROMPT, loss_context)
        else:
            st.info("No decline reasons found in the filtered lost bids.")

    except Exception as e:
        logger.error(f"Error generating loss reason analysis: {e}", exc_info=True)
        st.error("Could not generate the loss reason analysis visual.")

st.divider()

# --- Section: Pricing Suggestions (AI Driven) ---
st.header("💡 AI Pricing Strategy Suggestions")
st.markdown("AI-generated suggestions for pricing adjustments based on overall competitive context.")

# Use the competitive summary context created earlier
if 'comp_summary_context' in locals():
    # Add formatting arguments needed by the custom prompt
    pricing_suggestion_args = {
        'win_rate': win_rate,
        'avg_bid_amount': avg_bid_amount,
        'avg_comp_price_str': comp_price_str # Pass the formatted string
    }
    render_ai_explanation(
        "Get AI Pricing Strategy Suggestions",
        client,
        PRICING_SUGGESTIONS_PROMPT, # Use the custom prompt
        comp_summary_context,       # Provide the base context
        additional_format_args=pricing_suggestion_args # Provide specific args for prompt formatting
    )
else:
    st.warning("Competitive summary context not available for generating pricing suggestions.")


st.divider()

# --- Section: Chat Feature ---
st.header("💬 Ask Questions About Competitive Intelligence") # Updated header

# Prepare context for the chat
if 'comp_summary_context' not in locals():
     comp_summary_context = f"Competitive intelligence analysis based on {len(bids_final)} filtered bids."
     logger.warning("Using fallback chat context as comp_summary_context was not defined.")

chat_base_context = comp_summary_context # Use the determined context

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
    base_context=chat_base_context, # Use the page-specific context variable
    filter_details=filter_details_dict,
    page_key="competitive", # Unique key for this page
    placeholder="Ask about win rates, competitor pricing, or loss reasons..." # Updated placeholder
)