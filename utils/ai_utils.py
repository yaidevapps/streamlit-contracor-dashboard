# utils/ai_utils.py
import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
import logging
from config import AI_MODEL_NAME, AI_FALLBACK_MESSAGE, CURRENT_DATE # Make sure CURRENT_DATE is imported if needed elsewhere, though not directly used in functions here

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv() # Load environment variables from .env file

# --- Prompt Templates ---

# Generic template for explaining visuals
VISUAL_EXPLANATION_PROMPT = """
You are a helpful analyst assistant interpreting a data visualization for a luxury low voltage contractor.

CONTEXT FOR THE VISUALIZATION:
{context}

TASK:
1. Provide a concise 2-3 sentence description of the visualization based on the context.
2. List exactly 3 key takeaways from the data shown in the visualization. Each takeaway should be a bullet point, highlighting a specific observation and its potential business implication (e.g., 'Observation X suggests potential for Y').

REQUIREMENTS:
- Base the description and takeaways strictly on the provided context data.
- Focus on insights relevant to the contracting business.
- Do not add recommendations unless they are direct implications of the takeaways.
- Ensure exactly 3 bullet points for the takeaways.

RESPONSE FORMAT:
[Description in 2-3 sentences]
- [Takeaway 1: Observation and implication]
- [Takeaway 2: Observation and implication]
- [Takeaway 3: Observation and implication]
"""

# Template for high-level business advice (e.g., Dashboard Summary)
SUMMARY_ADVICE_PROMPT = """
You are a Senior Business Consultant with 25+ years of experience specializing in the construction industry, with particular expertise in low voltage contracting for luxury residential clients in the Greater Seattle area.

CONTEXT (Based on current filters):
{context}

Current Date: {current_date}
Business Type: Luxury high-end residential low voltage contractor
Service Area: Greater Eastside and Seattle

TASK:
Analyze the business data overview provided in the context. Provide 3 high-impact, actionable recommendations focused on improving profitability or efficiency. Focus on practical steps that can be implemented within the next 30-60 days.

REQUIREMENTS:
- Base recommendations strictly on the provided data and metrics in the CONTEXT.
- Acknowledge any noted data limitations (like approximations for Utilization).
- Prioritize 3 high-impact recommendations.
- Consider the luxury residential market context.
- Focus on actionable items with potentially measurable outcomes.
"""

# Generic template for chat interactions
CHAT_PROMPT_TEMPLATE = """
You are an expert analyst and advisor for a luxury residential low voltage contractor in the Seattle/Eastside area. You have access to the following data summary based on the user's current filters.

AVAILABLE DATA CONTEXT (Filtered - {start_date} to {end_date}, City: {city}, Type: {ptype}):
{context}

USER QUERY:
'{user_prompt}'

INSTRUCTIONS:
1. Answer the user's query based *strictly* on the provided data context.
2. If the context contains specific data relevant to the query (like tables or summaries), prioritize using that directly in your answer.
3. If the exact data isn't in the context summary, state that you cannot provide the specific detail but may be able to answer based on the overall context provided (e.g., total revenue, average margins).
4. Be concise, clear, and action-oriented.
5. Do not make up data or perform calculations beyond simple interpretations of the provided context.
6. If the query is too complex or requires data not present, politely explain the limitation.

RESPONSE FORMAT:
- Begin with a direct answer to the query in 1-2 sentences based *only* on the context.
- Follow with 2-3 specific, actionable recommendations or insights relevant to the query and context, if applicable.
- Use bullet points for clarity.
"""


# --- AI Client and Generation ---

@st.cache_resource
def get_genai_client():
    """Initializes and returns the Generative AI client."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        # Avoid showing Streamlit error directly in utility function if possible
        # Log the error, caller function should handle the UI feedback
        logger.error("GOOGLE_API_KEY not found in .env file. AI Client cannot be initialized.")
        # Optionally: raise ValueError("GOOGLE_API_KEY not found.")
        return None # Return None, let the caller page handle st.error
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(AI_MODEL_NAME)
        logger.info(f"Generative AI client initialized with model: {AI_MODEL_NAME}")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize AI client: {e}", exc_info=True)
        # Again, let the caller handle st.error based on None return
        return None

def generate_content_with_fallback(client, prompt, fallback=AI_FALLBACK_MESSAGE):
    """Generates content using the AI client with error handling."""
    if not client:
        logger.error("AI client not available for content generation.")
        return fallback
    try:
        logger.debug(f"Generating AI content with prompt starting: {prompt[:150]}...")
        # Set safety settings (adjust as needed) - blocking harmful content
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        response = client.generate_content(prompt, safety_settings=safety_settings)

        # Check for blocking reasons first
        if response.prompt_feedback.block_reason:
            logger.warning(f"AI response blocked due to safety settings: {response.prompt_feedback.block_reason}")
            # Provide a user-friendly message about blocked content
            return f"AI response blocked due to content safety reasons ({response.prompt_feedback.block_reason}). Please refine your query or check safety settings."

        # Check if response parts exist (more reliable than just checking .text)
        if not response.parts:
             logger.warning("AI response received but has no parts (content).")
             # Check candidates if available, they might contain error messages
             if hasattr(response, 'candidates') and response.candidates:
                 candidate_info = response.candidates[0] # Check the first candidate
                 if candidate_info.finish_reason != "STOP":
                     logger.warning(f"AI generation finished unexpectedly: {candidate_info.finish_reason}. Safety ratings: {getattr(candidate_info, 'safety_ratings', 'N/A')}")
                     return f"AI generation finished unexpectedly ({candidate_info.finish_reason}). It might be due to safety filters or other issues."
             return fallback # Default fallback if no other info

        # Attempt to extract text safely using response.text getter
        try:
            generated_text = response.text # This getter handles joining parts and should raise ValueError if blocked
            if generated_text:
                logger.info("AI content generated successfully.")
                return generated_text
            else:
                # This case should be rare if .parts existed, but handle it
                logger.warning("AI response has parts but .text is empty.")
                return fallback

        except ValueError as ve:
             # This typically means content was blocked despite not being caught by prompt_feedback.block_reason
             logger.error(f"ValueError accessing response.text (likely blocked content): {ve}")
             # Check safety feedback within the response if possible
             safety_feedback_str = ""
             if hasattr(response, 'candidates') and response.candidates and hasattr(response.candidates[0], 'safety_ratings'):
                  safety_feedback_str = f" Safety Ratings: {response.candidates[0].safety_ratings}"
             return f"AI response generation failed. The content might have been blocked due to safety settings.{safety_feedback_str} Please refine your query."
        except Exception as e_text:
            logger.error(f"Unexpected error accessing AI response text: {e_text}", exc_info=True)
            return fallback

    except Exception as e:
        logger.error(f"Error during AI content generation API call: {e}", exc_info=True)
        # Avoid showing raw error in UI from utility function if possible
        st.warning(f"AI generation failed due to an API error. Please try again later.") # More generic message
        return fallback

# --- Chat Memory and Interface ---

def initialize_chat_memory(page_key):
    """Initializes chat history for a specific page in session state."""
    session_key = f"messages_{page_key}" # Use the passed page_key
    if session_key not in st.session_state:
        st.session_state[session_key] = []
        logger.info(f"Chat memory initialized for key: {session_key}")

# Modified render_chat_interface to accept filter details and page_key
def render_chat_interface(client, chat_prompt_template, base_context, filter_details, page_key, placeholder="Ask a question..."):
    """
    Renders the chat input and manages message history specific to a page_key.

    Args:
        client: The AI client.
        chat_prompt_template (str): The base prompt template.
        base_context (str): The primary data context for the AI.
        filter_details (dict): Dictionary containing current filter values for prompt formatting.
        page_key (str): A unique key for this page's chat (e.g., 'dashboard', 'financials').
        placeholder (str): Placeholder text for the chat input.
    """
    session_key = f"messages_{page_key}"
    initialize_chat_memory(page_key) # Ensure memory is initialized for the page

    # --- Chat Suggestions ---
    # Display suggestions based on the current page (page_key)
    st.markdown("Try asking:")
    if page_key == "dashboard":
        st.markdown("""
        - "How can I improve my on-time completion rate?"
        - "What's driving my technician utilization rate?"
        - "How does my cash position compare to industry standards?"
        """)
    elif page_key == "financials":
        st.markdown("""
        - "Which project type has the highest average profit margin?"
        - "What were the total labor costs last month?"
        - "Show me details of the 5 largest overdue invoices."
        """)
    elif page_key == "customer":
        st.markdown("""
        - "Which neighborhoods have our highest-value customers?"
        - "What factors most impact customer retention rates?"
        - "How can we improve satisfaction scores for our top services?"
        """)
    elif page_key == "operations":
        st.markdown("""
        - "How can we improve our first-time fix rate for 'Network Troubleshooting' calls?"
        - "Which technicians have the highest utilization but lowest satisfaction scores?"
        - "What's the optimal reorder strategy for high-cost inventory items?"
        """)
    elif page_key == "project":
        st.markdown("""
        - "How can we improve our bid win rate for luxury projects?"
        - "What steps can we take to reduce timeline variances?"
        - "Which project types are most profitable, and how can we prioritize them?"
        """)
    elif page_key == "employee":
        st.markdown("""
        - "Which employees need certification renewals soon?"
        - "Identify top performers based on total hours logged or overtime ratios."
        - "Show me the skills matrix breakdown for Lead Technicians."
        """)
    elif page_key == "marketing":
        st.markdown("""
        - "Which campaign type yielded the best ROI?"
        - "What is the customer acquisition cost for social media campaigns?"
        - "Suggest ways to improve lead generation in the Medina neighborhood."
        """)
    elif page_key == "warranty_quality":
        st.markdown("""
        - "How can we reduce our warranty claim rate?"
        - "What are the main causes of high warranty costs?"
        - "Which project types are most prone to warranty issues?"
        """)
    elif page_key == "competitive": # <--- THIS BLOCK IS NOW CORRECTLY INCLUDED
        st.markdown("""
        - "How can we improve our bid win rate?"
        - "What pricing strategy works best against competitors?"
        - "Why are we losing bids despite pricing lower than competitors?"
        """)
    else:
         # Default suggestion if page_key doesn't match known pages
         st.markdown("- Ask anything about the data shown on this page.")


    # Display past messages for THIS PAGE
    for message in st.session_state[session_key]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input
    # Ensure a unique key for the chat_input widget per page to avoid state conflicts
    if user_prompt := st.chat_input(placeholder, key=f"chat_input_{page_key}"):
        # Add user message to THIS PAGE'S state and display
        st.session_state[session_key].append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Prepare prompt for AI using the provided base context and filter details
        full_context = base_context # Use the potentially page-specific context

        # Format the prompt using all necessary keys from the template and filter_details
        try:
            # Ensure filter_details keys match placeholders exactly in CHAT_PROMPT_TEMPLATE
            # Common keys: start_date, end_date, city, ptype (for project_type)
            format_args = {
                'context': full_context,
                'user_prompt': user_prompt,
                'start_date': filter_details.get('start_date', 'N/A'),
                'end_date': filter_details.get('end_date', 'N/A'),
                'city': filter_details.get('city', 'N/A'),
                'ptype': filter_details.get('project_type', 'N/A') # Map project_type to ptype placeholder
            }
            formatted_prompt = chat_prompt_template.format(**format_args)
        except KeyError as e:
             logger.error(f"Missing key during chat prompt formatting: {e}. Template needs placeholders defined in CHAT_PROMPT_TEMPLATE. Provided keys: {list(format_args.keys())}")
             st.error(f"Internal error formatting chat prompt (missing key: {e}).")
             return # Stop processing this request
        except Exception as e:
             logger.error(f"Error formatting chat prompt: {e}", exc_info=True)
             st.error("Internal error formatting chat prompt.")
             return


        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Ensure the AI client is valid before generating
                if not client:
                     response = AI_FALLBACK_MESSAGE # Or a specific "AI client not available" message
                     st.error("AI Client is not available.") # Show error in UI
                else:
                     response = generate_content_with_fallback(client, formatted_prompt)
                st.markdown(response)
        # Add assistant response to THIS PAGE'S state
        st.session_state[session_key].append({"role": "assistant", "content": response})


# --- Helper for AI Explanations ---
def render_ai_explanation(expander_title, client, prompt_template, context_data, additional_format_args=None):
    """
    Renders an expander with a button to generate AI explanation and stores result in session_state.

    Args:
        expander_title (str): Title for the expander.
        client: The initialized AI client.
        prompt_template (str): The prompt template string (e.g., VISUAL_EXPLANATION_PROMPT).
        context_data (str): The primary context string to format into the prompt's {context}.
        additional_format_args (dict, optional): Extra keyword arguments for formatting the prompt.
    """
    # Create a unique key for storing this explanation in session state
    # Replace problematic characters for session state key robustness
    base_key = expander_title.lower().replace(' ', '_').replace('&','_and_').replace(':','_').replace('/','_').replace('?','_').replace('%','_pct_')
    explanation_key = f"explanation_{base_key}"


    with st.expander(expander_title, expanded=(explanation_key in st.session_state)): # Expand if already generated
        button_key = f"button_{explanation_key}" # Unique key for button

        # --- Display stored explanation ---
        if explanation_key in st.session_state:
            st.markdown(st.session_state[explanation_key])
            # Add a button to explicitly regenerate
            if st.button("Regenerate Analysis", key=f"regen_{button_key}"):
                 # Clear the specific explanation from session state and rerun
                 if explanation_key in st.session_state:
                      del st.session_state[explanation_key]
                 logger.info(f"Regenerating explanation for {explanation_key}")
                 st.rerun() # Rerun to trigger regeneration logic below

        # --- Generate Button (Show only if not generated yet) ---
        else:
            # Button to trigger generation the first time
            if st.button("Generate AI Analysis", key=button_key):
                if not context_data or not isinstance(context_data, str) or not context_data.strip():
                    st.warning("Cannot generate analysis: Context data is missing or empty.")
                    st.session_state[explanation_key] = "_Error: Context data missing or invalid._" # Store error state
                    st.rerun()
                    return
                if not client:
                    st.error("Cannot generate analysis: AI Client not available.")
                    st.session_state[explanation_key] = "_Error: AI client unavailable._" # Store error state
                    st.rerun()
                    return

                # Prepare format arguments, ensuring context is primary
                format_args = {'context': context_data}
                if additional_format_args:
                    # Ensure additional args don't overwrite 'context' accidentally
                    additional_format_args.pop('context', None)
                    format_args.update(additional_format_args)

                # Format the prompt carefully
                try:
                    # Attempt to find expected keys from the template string
                    import string
                    formatter = string.Formatter()
                    expected_keys = {name for _, name, _, _ in formatter.parse(prompt_template) if name is not None}
                    prompt = prompt_template.format(**format_args)
                except KeyError as e:
                    # Log detailed error including expected keys vs provided keys
                    logger.error(f"Missing key during explanation prompt formatting: {e}. Template expects: {expected_keys}. Provided args: {list(format_args.keys())}")
                    st.error(f"Internal error formatting AI prompt (missing key: {e}). Cannot generate analysis.")
                    st.session_state[explanation_key] = f"_Error: Prompt formatting failed (missing key: {e})._"
                    st.rerun()
                    return
                except Exception as e:
                    logger.error(f"Error formatting explanation prompt: {e}", exc_info=True)
                    st.error("Internal error formatting AI prompt. Cannot generate analysis.")
                    st.session_state[explanation_key] = "_Error: Prompt formatting failed._"
                    st.rerun()
                    return

                # Generate explanation and store it in session state
                with st.spinner("Generating analysis..."):
                    explanation = generate_content_with_fallback(client, prompt)
                    st.session_state[explanation_key] = explanation # Store result (success or fallback message)
                    logger.info(f"Generated explanation for {explanation_key}. Length: {len(explanation)}")
                    st.rerun() # Rerun immediately to display the stored result

            else:
                # Prompt user to click the button if explanation not yet generated
                st.info("Click button to generate AI analysis for this section.")