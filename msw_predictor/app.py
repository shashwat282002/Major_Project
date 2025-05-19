import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
import pandas as pd

# Custom CSS for sidebar and headers
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {background-color: #e8f5e9;}
    .stApp {background-color: #f5e6cc; color: #111 !important;}
    .big-header {color: #111 !important; font-size: 2.2em; font-weight: bold;}
    .section-header {color: #111 !important; font-size: 1.3em; font-weight: bold; margin-top: 1em;}
    .stButton > button {color: #fff !important;}
    .stDownloadButton > button {color: #fff !important;}
    .stTabs [data-baseweb="tab"] {color: #111 !important;}
    .stSlider > label, .stSelectbox > label, label {color: #111 !important;}
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar with project info
st.sidebar.title("About This App")
st.sidebar.info(
    "This app predicts annual municipal solid waste (MSW) generation for a country based on population, GDP, and income class.\n\nCreated by Shashwat Bindal & Priyanshu, Delhi Technological University."
)
with open("Plag_research_project.pdf", "rb") as pdf_file:
    PDFbyte = pdf_file.read()

st.sidebar.download_button(
    label="Download Project Report",
    data=PDFbyte,
    file_name="Plag_research_project.pdf",
    mime='application/octet-stream'
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Contact:** shashwat.bindal2002@gmail.com")

# Tabs for organization
tab1, tab2, tab3 = st.tabs(["Prediction", "About", "Contact"])

with tab1:
    st.markdown('<div class="big-header">♻️ Municipal Solid Waste (MSW) Predictor</div>', unsafe_allow_html=True)
    st.image("Solid_waste.jpeg", use_container_width=True, caption="Municipal Solid Waste Management")
    st.markdown(
        """
        Enter the details below to predict **annual municipal solid waste (MSW) generation** for a country.
        """
    )

    # UI for input with sliders and help text
    # Set initial/default value for population
    default_population = 1_000_000.0

    # Create two columns: one for slider, one for manual input for Population
    pop_col1, pop_col2 = st.columns(2)

    with pop_col1:
        population_slider = st.slider(
            "Population (use slider)",
            min_value=0.0,
            max_value=100_000_000.0,
            value=default_population,
            step=1.0,
            key="population_slider"
        )

    with pop_col2:
        population_input = st.number_input(
            "Population (or type manually)",
            min_value=0.0,
            max_value=100_000_000.0,
            value=population_slider,
            step=1.0,
            key="population_input"
        )

    # Sync: If the user changes the slider, update the input, and vice versa
    population = population_input if population_input != population_slider else population_slider

    # Set initial/default value for GDP
    default_gdp = 10000.0

    # Create two columns: one for slider, one for manual input for GDP
    col3, col4 = st.columns(2)

    with col3:
        gdp_slider = st.slider(
            "GDP (USD) (use slider)",
            min_value=0.0,
            max_value=100_000_000.0,
            value=default_gdp,
            step=0.01,
            key="gdp_slider"
        )

    with col4:
        gdp_input = st.number_input(
            "GDP (USD) (or type manually)",
            min_value=0.0,
            max_value=100_000_000.0,
            value=gdp_slider,
            step=0.01,
            key="gdp_input"
        )

    # Sync: If the user changes the slider, update the input, and vice versa
    gdp = gdp_input if gdp_input != gdp_slider else gdp_slider

    income_id = st.selectbox("Income Class", ["HIC", "LIC", "LMC", "UMC"], help="Select the income class as per World Bank classification.")

    # One-hot encoding for income_id (HIC is reference: all zeros)
    income_id_LIC = 1 if income_id == "LIC" else 0
    income_id_LMC = 1 if income_id == "LMC" else 0
    income_id_UMC = 1 if income_id == "UMC" else 0

    # Pass a dummy value for msw to match scaler's expected input shape
    # Only use the first two scaled values for model input
    with open('msw_predictor/xgb_msw_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('msw_predictor/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    scaled_vals = scaler.transform([[population, gdp, 0]])
    scaled_population = scaled_vals[0][0]
    scaled_gdp = scaled_vals[0][1]
    X_input = np.array([[scaled_population, scaled_gdp, income_id_LIC, income_id_LMC, income_id_UMC]])

    # Prediction and UI
    if st.button("Predict Municipal Solid Waste"):
        with st.spinner('Calculating prediction...'):
            prediction = model.predict(X_input)[0]
            # Inverse transform to get actual MSW in tons/year
            dummy = np.array([[0, 0, prediction]])
            msw_actual = scaler.inverse_transform(dummy)[0][2]
            st.markdown(f'<div style="color: #111; background-color: #d4edda; padding: 1em; border-radius: 8px; font-weight: bold;">Predicted Municipal Solid Waste Production: {msw_actual:,.2f} tons/year</div>', unsafe_allow_html=True)
            st.snow()

            # --- Bar Chart: Compare to average MSW for each income class ---
            st.markdown('<div class="section-header">Municipal Solid Waste Comparison by Income Class</div>', unsafe_allow_html=True)
            avg_msw_by_income = {
                "HIC": 3_000_000,
                "UMC": 2_000_000,
                "LMC": 1_000_000,
                "LIC": 500_000
            }
            bar_labels = list(avg_msw_by_income.keys())
            bar_values = list(avg_msw_by_income.values())
            highlight = [msw_actual if income_id == k else v for k, v in avg_msw_by_income.items()]
            fig, ax = plt.subplots()
            bars = ax.bar(bar_labels, bar_values, color=["#4CAF50" if k != income_id else "#FF9800" for k in bar_labels], alpha=0.7)
            bars[bar_labels.index(income_id)].set_height(msw_actual)
            bars[bar_labels.index(income_id)].set_color("#FF9800")
            ax.set_ylabel("MSW (tons/year)")
            ax.set_title("Predicted vs. Average MSW by Income Class")
            st.pyplot(fig)

            # --- Pie Chart: Example Waste Composition ---
            st.markdown('<div class="section-header">Sample Waste Composition</div>', unsafe_allow_html=True)
            waste_labels = ["Organic", "Plastic", "Paper", "Glass", "Metal", "Other"]
            waste_sizes = [50, 20, 15, 5, 5, 5]  # Example percentages
            fig2, ax2 = plt.subplots()
            ax2.pie(waste_sizes, labels=waste_labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
            ax2.axis('equal')
            st.pyplot(fig2)

            # --- Download Button for Prediction ---
            result_df = pd.DataFrame({
                'Population': [population],
                'GDP (USD)': [gdp],
                'Income Class': [income_id],
                'Predicted MSW (tons/year)': [msw_actual]
            })
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Prediction as CSV",
                data=csv,
                file_name='msw_prediction.csv',
                mime='text/csv',
            )

            # --- Fun Fact ---
            facts = [
                "Did you know? Recycling one aluminum can saves enough energy to run a TV for 3 hours.",
                "Composting food waste can reduce landfill waste by up to 30%.",
                "The average person generates over 700 kg of MSW per year.",
                "Recycling 1 ton of paper saves 17 trees and 7,000 gallons of water.",
                "Plastic can take up to 1,000 years to decompose in landfills."
            ]
            random_fact = random.choice(facts)
            st.markdown(f'<div style="color: #111; background-color: #e3f2fd; padding: 0.7em 1em; border-radius: 8px; margin-bottom: 0.5em;">{random_fact}</div>', unsafe_allow_html=True)

            # --- Placeholder for map visualization (future) ---
            # st.map()  # Uncomment and provide data for real map

#             

with tab2:
    st.markdown('<div class="big-header">About the Project</div>', unsafe_allow_html=True)
    st.write("""
    This project demonstrates the use of machine learning for forecasting municipal solid waste (MSW) generation using country-level socioeconomic and demographic data.\n\n
    **Features:**
    - Predict MSW based on population, GDP, and income class
    - Visualize results with interactive charts
    - Download predictions
    - Learn fun facts about waste management
    """)
    st.write("**Developed by:** Shashwat Bindal & Priyanshu, Delhi Technological University")
    st.write("**Contact:** shashwat.bindal2002@gmail")

with tab3:
    st.markdown('<div class="big-header">Contact</div>', unsafe_allow_html=True)
    st.write("For questions, suggestions, or collaborations, please email: shashwat.bindal2002@gmail")
    st.write("[LinkedIn](https://www.linkedin.com/) | [GitHub](https://github.com/)") 
