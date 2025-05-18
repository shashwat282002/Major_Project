# MSW Predictor

A simple Streamlit app to predict municipal solid waste (MSW) generation using an XGBRegressor model.

## How to Run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Place your trained model as `xgb_msw_model.pkl` in this directory.
3. Start the app:
   ```
   streamlit run app.py
   ```
4. Open the provided local URL in your browser.

## Features

- Enter population, GDP, and income class
- Get instant MSW prediction
- Modern, interactive UI 