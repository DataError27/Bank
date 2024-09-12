import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime, timedelta
import altair as alt

# Define the calculate_rsi function locally
def calculate_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Set up Streamlit app layout
st.set_page_config(layout="wide")
st.markdown("<h1 style='font-size: 3rem;'>Singapore Bank Stock Analysis Tool</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='font-size: 1rem; color: grey;'>Leverage data analytics and machine learning to anticipate future market movements and make strategic choices.</h3>", unsafe_allow_html=True)

# Split the page into a sidebar and main content with a 1:2 ratio
sidebar_col, main_col = st.columns([1, 2])

# Sidebar content for bank selection and predictions
with sidebar_col:
    st.markdown("<h3 style='font-size: 1.5rem; color: white;'>Bank Stock Analysis</h3>", unsafe_allow_html=True)

    # Define the options for the selectbox
    options = {
        "Select": "Please select an option:",
        "DBS": "DBS Bank",
        "OCBC": "OCBC Bank",
        "UOB": "UOB Bank"
    }

    # Create a selectbox for user to select a bank
    selected_bank = st.selectbox("Choose a Bank:", list(options.keys()))

    # Get the current stock price based on the selected bank
    if selected_bank == "DBS":
        ticker = "D05.SI"
        data_file = 'dbs_data1.csv'
        monthly_closing_file = 'dbs_monthly_closing.csv'
        model_file = 'DBSbest_sarimax_model.pkl'
        logo_file = 'dbs.png'
        exog_vars = ['RSI', '50_day_MA', '200_day_MA',
                     'Government Securities - 5-Year Bond Yield (Per Cent Per Annum)',
                     'Government Securities - 2-Year Bond Yield (Per Cent Per Annum)',
                     'Government Securities - 10-Year Bond Yield (Per Cent Per Annum)',
                     'Government Securities - 1-Year Treasury Bills Yield (Per Cent Per Annum)',
                     'Government Securities - 15-Year Bond Yield (Per Cent Per Annum)',
                     'Government Securities - 20-Year Bond Yield (Per Cent Per Annum)',
                     'Singapore Overnight Rate Average (Per Cent Per Annum)', 'Dividends',
                     'ROA', 'ROE', 'Cost/Income Ratio']  # Full exog_vars list for DBS
    elif selected_bank == "OCBC":
        ticker = "O39.SI"
        data_file = 'ocbc_data1.csv'
        monthly_closing_file = 'ocbc_monthly_closing.csv'
        model_file = 'OCBCbest_sarimax_model.pkl'
        logo_file = 'ocbc.png'
        exog_vars = ['RSI', '50_day_MA', '200_day_MA',
                     'Government Securities - 5-Year Bond Yield (Per Cent Per Annum)',
                     'Government Securities - 2-Year Bond Yield (Per Cent Per Annum)',
                     'Government Securities - 10-Year Bond Yield (Per Cent Per Annum)',
                     'Government Securities - 1-Year Treasury Bills Yield (Per Cent Per Annum)',
                     'Government Securities - 15-Year Bond Yield (Per Cent Per Annum)',
                     'Government Securities - 20-Year Bond Yield (Per Cent Per Annum)',
                     'Singapore Overnight Rate Average (Per Cent Per Annum)', 'Dividends']  # Adjusted exog_vars list for OCBC
    elif selected_bank == "UOB":
        ticker = "U11.SI"
        data_file = 'uob_data1.csv'
        monthly_closing_file = 'uob_monthly_closing.csv'
        model_file = 'UOBbest_sarimax_model.pkl'
        logo_file = 'uob.png'
        exog_vars = ['RSI', '50_day_MA', '200_day_MA',
                     'Government Securities - 5-Year Bond Yield (Per Cent Per Annum)',
                     'Government Securities - 2-Year Bond Yield (Per Cent Per Annum)',
                     'Government Securities - 10-Year Bond Yield (Per Cent Per Annum)',
                     'Government Securities - 1-Year Treasury Bills Yield (Per Cent Per Annum)',
                     'Government Securities - 15-Year Bond Yield (Per Cent Per Annum)',
                     'Government Securities - 20-Year Bond Yield (Per Cent Per Annum)',
                     'Singapore Overnight Rate Average (Per Cent Per Annum)', 'Dividends']  # Adjusted exog_vars list for UOB
    else:
        ticker = None
        data_file = None
        model_file = None
        monthly_closing_file = None
        logo_file = None
        exog_vars = []

    if ticker and data_file and model_file:
        stock = yf.Ticker(ticker)
        stock_price = stock.history(period="1d")['Close'][0]
        st.markdown(f"### 📊 Current {selected_bank} Stock Price: **${stock_price:.2f}**", unsafe_allow_html=True)

        # Informational message about the machine learning model's runtime
        st.info("Note: Running the machine learning model may take a while. Please be patient as we process the data.")

        # Define the stock symbol and period for which to pull current data
        symbol = ticker
        end_date = datetime.today().date()
        start_date = end_date - timedelta(days=365)

        # Fetch historical data from Yahoo Finance
        data = yf.download(symbol, start=start_date, end=end_date, interval='1d')

        # Calculate technical indicators
        data['50_day_MA'] = data['Close'].rolling(window=50).mean()
        data['200_day_MA'] = data['Close'].rolling(window=200).mean()
        data['RSI'] = calculate_rsi(data['Close'])

        # Load the existing data from the CSV file
        previous_exog_data = pd.read_csv(data_file)
        previous_exog_data['Date'] = pd.to_datetime(previous_exog_data['Date'])
        previous_exog_data.set_index('Date', inplace=True)

        # Merge new technical indicators with the existing data, ensuring alignment
        combined_data = pd.concat([previous_exog_data, data[['RSI', '50_day_MA', '200_day_MA']]], axis=1)
        combined_data = combined_data.replace([np.inf, -np.inf], np.nan).ffill().bfill()  # Clean and align data

        # Verify that combined_data does not contain any columns of unequal length
        combined_data = combined_data.loc[combined_data.index.dropna()]  # Drop rows with missing indices

        # Ensure that there are no missing values in exogenous data
        if combined_data[exog_vars].isnull().values.any():
            st.error("Exogenous variables contain NaNs even after cleaning. Check your data source for gaps.")
        else:
            # Train a new SARIMAX model using the combined dataset
            model = SARIMAX(combined_data['Close'], exog=combined_data[exog_vars], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            fitted_model = model.fit(disp=False)

            # Predict the closing price for the next period (e.g., next month)
            # Use the integer index to align predictions with the model's data
            future_exog = combined_data[exog_vars].iloc[-1].to_frame().T
            future_exog.index = [combined_data.index[-1] + pd.DateOffset(months=1)]

            # Adjusting indices for prediction using integer-based indexing
            start_idx = len(combined_data)
            end_idx = start_idx

            predicted_close = fitted_model.predict(start=start_idx, end=end_idx, exog=future_exog)

            # Display the predicted closing price prominently on the sidebar
            st.markdown(f"### 🔮 Predicted End-of-Month Closing Price for {selected_bank}: **${predicted_close.iloc[0]:.2f}**", unsafe_allow_html=True)

            # Provide commentary based on the predicted vs. current price
            price_difference = abs(predicted_close.iloc[0] - stock_price)

            if predicted_close.iloc[0] > stock_price:
                if price_difference <= 0.08:
                    st.write(f"The predicted price for {selected_bank} is expected to rise slightly above the current price, but the difference is minimal (${price_difference:.2f}). This suggests the stock may remain stable, and investors might want to hold or wait for further signals.")
                else:
                    st.write(f"The predicted price for {selected_bank} is expected to rise above the current price, indicating potential growth. This could be a good opportunity for investors seeking capital appreciation.")
            elif predicted_close.iloc[0] < stock_price:
                if price_difference <= 0.08:
                    st.write(f"The predicted price for {selected_bank} is expected to fall slightly below the current price, with a minimal difference (${price_difference:.2f}). This indicates a potential period of consolidation, and investors might consider holding or monitoring the stock closely.")
                else:
                    st.write(f"The predicted price for {selected_bank} is expected to be lower than the current price, suggesting caution. Investors may want to consider the potential risks before making investment decisions.")
            else:
                st.write(f"The predicted price for {selected_bank} is very close to the current price, suggesting stability. Investors might expect minimal price movement in the near term and may hold or wait for further developments.")

            # Display the bank logo below the predicted price
            if logo_file:
                st.image(logo_file, width=200)  # Adjusted size

            # Add specific development for each bank
            if selected_bank == "DBS":
                st.markdown("### Latest Development")
                st.write("""
                **Upcoming Change in Leadership:** DBS is set to undergo a significant leadership transition with the appointment of a new CEO. 
                The current CEO, Piyush Gupta, who has been at the helm for over a decade, is expected to step down in early 2025. 
                The new CEO, Tan Su Shan, will be tasked with steering DBS through the next phase of its digital transformation and regional expansion.
                This leadership change is closely watched by investors, as it could influence the strategic direction of the bank in the coming years.
                """)
            elif selected_bank == "OCBC":
                st.markdown("### Latest Development")
                st.write("""
                **Strategic Expansion into Digital Banking:** OCBC has announced plans to further expand its digital banking services across Southeast Asia, 
                leveraging new technologies to enhance customer experience and streamline operations. This move is part of OCBC's broader strategy to 
                solidify its market position and capture a larger share of the growing digital banking sector in the region.
                """)
            elif selected_bank == "UOB":
                st.markdown("### Latest Development")
                st.write("""
                **Focus on Sustainability Initiatives:** UOB is ramping up its efforts in sustainability, with new initiatives aimed at reducing the bank's carbon footprint and promoting green financing. 
                UOB has committed to increasing its sustainable financing portfolio to support businesses in their transition to more sustainable practices. 
                This commitment reflects UOB's strategic direction towards responsible banking and investment.
                """)

# Display the methodology section once, outside of the bank selection logic
with main_col:
    st.markdown("<h3 style='font-size: 1.5rem;'>Methodology</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        <p style='font-size: 1rem;'>
        Our analysis is built on four foundational aspects:
        </p>
        <ul style='font-size: 1rem;'>
            <li><strong>Crowd Mentality:</strong> We consider the impact of market sentiment and crowd behavior, often reflected in technical indicators such as Relative Strength Index (RSI), Moving Averages, and other momentum indicators. These tools help gauge the collective mood and actions of market participants, which can drive price movements independent of fundamental or macroeconomic factors. By incorporating these technical indicators, we aim to capture the short-term dynamics driven by crowd mentality and investor sentiment.</li>
            <li><strong>Company's Health:</strong> We evaluate the financial stability of each bank by examining fundamental indicators like Return on Assets (ROA), Return on Equity (ROE), and the Cost/Income Ratio. These metrics provide insight into the bank's operational efficiency and are key components in our machine learning models.</li>
            <li><strong>Regulatory Environment:</strong> The influence of macroeconomic factors, particularly interest rates and government securities yields, reflects the regulatory landscape that affects the banks' operations and profitability. Understanding these variables helps in forecasting market conditions that impact bank performance.</li>
            <li><strong>Seasonality:</strong> By analyzing average monthly movements in stock prices, we capture seasonal trends that highlight recurring patterns within each year. This seasonal behavior suggests that certain months consistently exhibit similar stock price movements. Browse through the selections to explore each bank's seasonality trends.</li>
        </ul>
        """,
        unsafe_allow_html=True
    )

    # Plotting the average monthly movement (current close - previous close) for seasonality analysis
    if monthly_closing_file and selected_bank != "Select":
        # Load the monthly closing data
        monthly_closing_data = pd.read_csv(monthly_closing_file, parse_dates=['Date'])
        monthly_closing_data.set_index('Date', inplace=True)

        # Calculate the monthly movement (current close - previous close)
        monthly_closing_data['Monthly_Movement'] = monthly_closing_data['Close'].diff()

        # Group by month to calculate the average monthly movement
        monthly_closing_data['Month'] = monthly_closing_data.index.month
        avg_monthly_movement = monthly_closing_data.groupby('Month')['Monthly_Movement'].mean()

        # Create a DataFrame with month names for better display on the bar chart
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Ensure the index is sorted by month number (1 to 12)
        avg_monthly_movement = avg_monthly_movement.sort_index()
        
        # Create a new DataFrame to map month names and ensure correct ordering
        avg_monthly_movement = pd.DataFrame(avg_monthly_movement).reset_index()
        avg_monthly_movement['Month'] = avg_monthly_movement['Month'].apply(lambda x: month_names[x-1])

        # Reindex to ensure January to December order explicitly
        avg_monthly_movement.set_index('Month', inplace=True)
        avg_monthly_movement = avg_monthly_movement.reindex(month_names)

        # Displaying the average monthly movement using Altair's bar chart with month names correctly ordered
        st.subheader(f"Average Monthly Movement for {selected_bank}")
        chart = alt.Chart(avg_monthly_movement.reset_index()).mark_bar().encode(
            x=alt.X('Month', sort=month_names),
            y='Monthly_Movement',
            tooltip=[alt.Tooltip('Month', title='Month'),
             alt.Tooltip('Monthly_Movement', title='Average Movement', format=".2f")]
        ).properties(
            width=600,
            height=400
        )
        st.altair_chart(chart, use_container_width=True)
