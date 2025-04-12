# Streamlit Contractor Dashboard

A Streamlit application providing analytics and insights for a luxury low-voltage contractor.

## Features

*   Executive Dashboard Overview
*   Financial Analysis (Revenue, Expenses, Profitability)
*   Customer Insights (Heatmap, CLV, Satisfaction)
*   Operational Metrics (Technician Productivity, Inventory)
*   Project Tracking (Pipeline, Bids, Timelines)
*   Employee Performance and Certifications
*   Marketing Campaign Analysis
*   Warranty & Quality Tracking
*   Competitive Intelligence
*   AI-powered insights and predictions

## Setup

1.  **Clone the repository (Optional - if someone else is using it):**
    ```bash
    git clone https://github.com/YOUR_USERNAME/streamlit-contractor-dashboard.git
    cd streamlit-contractor-dashboard
    ```

2.  **Create and Activate a Virtual Environment:**
    *   It's highly recommended to use a virtual environment.
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables:**
    *   Create a file named `.env` in the root project directory.
    *   Add your Google API key to this file:
        ```
        GOOGLE_API_KEY=PASTE_YOUR_ACTUAL_API_KEY_HERE
        ```
    *   **IMPORTANT:** This `.env` file is listed in `.gitignore` and will *not* be committed to GitHub. Users need to create their own.

5.  **Generate Sample Data:**
    *   Run the data generator script to create the necessary `.txt` files in the `contractor_data` folder:
    ```bash
    python data_generator.py
    ```
    *   *(Note: The `contractor_data` folder is ignored by Git and will not be uploaded.)*

## Running the Application

1.  Make sure your virtual environment is activated.
2.  Navigate to the project's root directory in your terminal.
3.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
4.  The application should open automatically in your web browser.