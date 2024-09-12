
# HDB-resale-price
 - [Background](#Background)
 - [Problem Statement](#Problem-Statement)
 - [Data Sources](#Data-Sources)
 - [Executive Summary](#Executive-Summary)
 - [Reflection](#Reflections)
 - [Conclusion](#Conclusion)
 

## Background

Singapore’s financial sector is a critical pillar of its economy, with major banks such as DBS, OCBC, and UOB playing pivotal roles. These banks' stock prices are influenced by various factors including macroeconomic conditions, regulatory policies, and investor sentiment. In recent years, the dynamic market environment and economic uncertainties have introduced significant volatility in stock prices. Investors and financial analysts are increasingly interested in predictive models that can provide insights into future stock price movements, enabling them to make informed investment decisions.

## Problem Statement
Investors face several challenges in navigating the fluctuating stock market, particularly in the banking sector where prices are heavily influenced by both domestic and global economic factors. This project aims to address the following question:

Can machine learning models be used to predict the stock prices of Singapore’s major banks, and how can these predictions be leveraged to inform investment strategies?


**Can we use machine learning to make predictions for resale prices?**

We will follow the data science process to answer this problem.
1. Define the problem
2. Gather & clean the data
3. Explore the data
4. Model the data
5. Evaluate the model
6. Answer the problem

--- 
## Data Sources
The dataset includes daily and monthly stock prices of DBS, OCBC, and UOB from Yahoo Finance, alongside additional macroeconomic data such as interest rates and bond yields. Key technical indicators such as Moving Averages (50-day, 200-day), Relative Strength Index (RSI), and other financial ratios were also calculated and used as exogenous variables in the models.

Data Range: Historical stock data up to the current date
Frequency: Daily for technical analysis; Monthly for seasonality and predictive modeling
Data Size: Approximately 20 years of historical data for each bank

## Executive Summary
**INTRODUCTION**

This project seeks to make predictions on the outcome of HDB resale prices through regression models. With the prediction price, agents will be able to use different strategies to meet their client's needs. 

**METHODOLOGY**

1. Preprocessing - Cleaning and aligning data, feature engineering technical indicators
2. EDA - Data visualization to identify trends, seasonality, and relationships between variables
3. Modeling - Using SARIMAX models with exogenous variables to capture both time series patterns and external influences
5. Evaluation - Assessing model performance using metrics such as RMSE and MAPE
6. Backtesting - Using test data to answer the problem statement of whether we can predict resale prices
7. Deployment - Building a Streamlit app to allow users to input simplified parameters and receive stock price predictions

The application was deployed on Streamlit and can be accessed through this [link](https://.streamlit.app/). A screenshot showing the app is shown below. Please note that this app was only intended as an educational purposes and is not a financial advice tool.


**FEATURES SELECTION/ADDITION**

Key features were derived from the stock data and external economic indicators:

Technical Indicators - RSI, Moving Averages (50-day, 200-day)
Macroeconomic Variables - Interest rates, bond yields, and treasury bills
Seasonality Analysis - Average monthly movements to capture recurring patterns in stock prices


**SIGNIFICANT FINDINGS**

Based on a context of structured tabular dataset, we managed to filter out the three strongest options for modeling which are Light Gradient Boosting Machine (LightGBM), CatBoost and Extreme Gradient Boosting Machine (XGBoost).

We further analysed performance of three machine learning models using metrics such as Train RMSE, Test RMSE, Train R², Test R² and Run Time (s). 
![models_metrics_table](images/metrics_quantitative_table.jpg)
Here is the analysis
1. Light Gradient Boosting Machine
This model shows strong performance with a low RMSE (Root Mean Square Error) on both the training and test datasets, indicating good predictive accuracy.The R² (coefficient of determination) of 0.97 is high, signifying that the model explains 97% of the variance in the data.
The run time of 50 seconds suggests that the model is relatively fast.
2. CatBoost
This model also performs well, with slightly higher RMSE values compared to LightGBM. However, the difference is minimal, and the model still maintains high accuracy.The R² is consistent at 0.97, indicating that it explains 97% of the data’s variance, similar to LightGBM.
The run time is 72 seconds, making it slower than LightGBM but still within a reasonable range.
3. XGBoost
XGBoost has the lowest RMSE among the three models, indicating the best predictive performance on both the training and test datasets.
The R² is consistent at 0.97, similar to the other models, explaining 97% of the variance in the data.
The run time is the fastest at 38 seconds, making XGBoost not only the most accurate but also the most efficient in terms of computational speed.

Overall Summary
LightGBM is a strong contender with very close performance metrics to XGBoost and a slightly longer run time.
CatBoost is still a solid model, particularly when we have a lot of categorical data, but it lags slightly behind in terms of RMSE and run time.
XGBoost outperforms the other two models in terms of both accuracy (lowest RMSE) and efficiency (shortest run time). It is our final model of choice since we prioritize predictive performance and computation time.

---
## Reflections
We want to reflect on key stages in our team work process:
1. Data Handling
- Cleaning & Feature Engineering: The team effectively cleaned the data and engineered key features, like flat_type and hdb_age, which were crucial for accurate predictions.
- Variable Selection: Relevant variables were carefully chosen, including handling categorical data properly and exploring interactions, improving model accuracy.
2. Model Fine-Tuning
- Hyperparameter Tuning: The team used grid and random search techniques to optimize hyperparameters, ensuring models were finely tuned for performance.
- Cross-Validation: Cross-validation helped prevent overfitting and ensured the models generalized well.
- Technical Challenges: Challenges like balancing model complexity with run time and handling large datasets were tackled effectively, ensuring efficient model training.
Summary
The team’s strong approach to data handling, variable selection, and model fine-tuning, combined with overcoming technical challenges, led to the development of accurate and reliable models for predicting HDB resale prices.
---
## Conclusion

In this project, we aimed to predict HDB resale prices in Singapore by leveraging a comprehensive dataset that included various factors such as the property's location, type, size, age, and proximity to amenities. Through careful data handling, feature engineering, and model selection, we successfully identified key variables that significantly impact resale prices. By comparing advanced machine learning models—LightGBM, XGBoost, and CatBoost—we were able to fine-tune and validate our models to achieve high accuracy and reliability. The results demonstrate the effectiveness of these models in capturing the complexities of the HDB resale market, providing valuable insights for stakeholders and setting the stage for future enhancements and potential real-world applications.
