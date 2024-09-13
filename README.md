![Alt text](https://raw.githubusercontent.com/DataError27/Bank/main/image.png)

# Singapore Bank Stocks Analysis Tool
 - [Background](#Background)
 - [Problem Statement](#Problem-Statement)
 - [Data Sources](#Data-Sources)
 - [Executive Summary](#Executive-Summary)
 - [Conclusion](#Conclusion)
 

## Background

Singapore’s financial sector is a cornerstone of its economy, with major banks such as DBS, OCBC, and UOB playing crucial roles. The stock prices of these banks are influenced by a variety of factors, including macroeconomic conditions, regulatory policies, and investor sentiment. Recent years have seen increased volatility due to a dynamic market environment and economic uncertainties. Predictive models that provide insights into future stock price movements are becoming essential for investors and financial analysts aiming to make informed decisions.

## Problem Statement
Navigating the fluctuating stock market, particularly within the banking sector, presents significant challenges due to both domestic and global economic influences. This project explores whether machine learning models can accurately predict the stock prices of Singapore’s major banks.

**Can machine learning models accurately predict stock prices?**


--- 
## Data Sources
The dataset includes daily and monthly stock prices of DBS, OCBC, and UOB from Yahoo Finance, alongside additional macroeconomic data such as interest rates and bond yields. Key technical indicators such as Moving Averages (50-day, 200-day), Relative Strength Index (RSI), and other financial ratios were also calculated and used as exogenous variables in the models.

Data Range: Historical stock data up to the current date

Frequency: Daily for technical analysis; Monthly for seasonality and predictive modeling

Data Size: Approximately 20 years of historical data for each bank

## Executive Summary
**INTRODUCTION**

This project aims to utilize machine learning models to forecast stock prices of major Singapore banks. The objective is to equip investors with tools to make more informed decisions, helping them manage risk effectively rather than just focusing on wealth accumulation.


**METHODOLOGY**

1. Preprocessing - Cleaning and aligning data, feature engineering technical indicators.
2. EDA - Data visualisation to identify trends, seasonality and relationships between variables.
3. Modeling - We tested several models to determine the best fit for predicting stock prices: SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors): This model captures both time series patterns and external influences, handling seasonality and trends effectively.
5. Evaluation - Metrics used include RMSE (Root Mean Square Error) and MAPE (Mean Absolute Percentage Error) to assess model performance and accuracy.
6. Backtesting - Evaluating models on test data to validate their predictive capability and practical utility.
7. Deployment - Building a Streamlit application to provide users with an interactive interface for inputting parameters and receiving stock price predictions.

The application was deployed on Streamlit and can be accessed through this [link](https://bank-analyst2.streamlit.app/). A screenshot showing the app is shown below. Please note that this app is only intended for educational purposes and is not a financial advice tool.


**FEATURES SELECTION/ADDITION**

Key features used in the models include:

Technical Indicators: RSI, Moving Averages (50-day, 200-day)

Fundamental indicators: Dividends, Return on Assets and Equity, Cost / Income ratio

Macroeconomic Variables: Interest rates, bond yields, treasury bills

Seasonality Analysis: Monthly movements to capture recurring patterns


**SIGNIFICANT FINDINGS**

We tested three machine learning models and analyzed their performance:

SARIMAX:
Effectively handles seasonality and external regressors.
Strong performance with low RMSE and MAPE, making it a robust choice for time series prediction.


FBProphet:
Handles seasonality and trends well but may require additional customisation for financial data.
Slightly higher RMSE compared to SARIMAX, but still a viable model with good interpretability.


Random Forest:
Provides good performance but less suited for capturing temporal dependencies compared to SARIMAX.
Effective in handling non-linear relationships but shows slightly higher RMSE and MAPE than SARIMAX.


**Overall Summary**

SARIMAX emerged as the best model due to its capability to incorporate seasonality, trends, and external variables, with the lowest RMSE and MAPE values.
FBProphet and Random Forest were also effective, but SARIMAX's comprehensive approach to time series data made it the preferred choice for accuracy and practical application.
In conclusion, SARIMAX is our final model of choice for predicting Singapore bank stock prices, offering a balance of predictive accuracy and practical application for investors.

---
## Conclusion

In conclusion, this project demonstrates that machine learning models, particularly SARIMAX, can be highly effective in accurately predicting stock prices for Singapore's major banks. By incorporating a range of technical indicators and macroeconomic variables, the models have shown considerable success in forecasting future stock movements. This effectiveness underscores the potential of machine learning to address the challenges of predicting stock prices amid fluctuating market conditions.

The deployment of our Streamlit application offers users an accessible and interactive tool for making informed investment decisions based on these predictions. While the primary focus is on enhancing decision-making rather than guaranteeing financial success, the ability to forecast stock prices with greater precision can significantly benefit investors navigating the complexities of the financial market.

The success of this project underscores the potential of machine learning in financial analysis and forecasting, paving the way for more sophisticated tools that can adapt to evolving market conditions and investor needs.
