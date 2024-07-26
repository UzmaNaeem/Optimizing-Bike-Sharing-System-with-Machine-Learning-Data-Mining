# MACHINE LEARNING & DATA MINING: BIKE SHARING INSIGHTS IN SEOUL
## A Data Analysis & Visualization Journey

By: Uzma Naeem

## Executive Summary
This project delves into the bike sharing demand prediction for Seoul, aiming to explore and model urban transportation patterns to enhance the efficiency of bike sharing systems. Central to our analysis is a comprehensive dataset that integrates rental bike counts with weather conditions and holiday information, offering a holistic perspective on urban mobility. Our findings illuminate the potential for transforming urban mobility through predictive modeling, enabling cities to allocate resources better, enhance user satisfaction, and reduce environmental impacts.

## Description of the Data
The dataset utilized is the "Seoul Bike Sharing Demand" dataset, donated on February 29, 2020. It provides a detailed hourly account of public bicycle rentals in Seoul, combined with relevant weather and holiday information, and is designed for regression tasks. The dataset includes:

- **8,760 instances and 13 features**: date, number of bikes rented each hour, hour of the day, temperature, humidity, wind speed, visibility, dew point temperature, solar radiation, rainfall, snowfall, season, holiday, and functional hour.

Notably, the dataset contains no missing values, ensuring data completeness for accurate analysis. Each variable is meticulously recorded to reflect the dynamic factors influencing bike rental demand, from environmental conditions to temporal and special day indicators.

## Key Findings & Insights
### Seasonal Trends
The seasonal analysis of bike rental demand revealed clear trends:
- **Summer**: Peak bike rental demand.
- **Autumn and Spring**: Moderate demand.
- **Winter**: Lowest demand.

This pattern suggests that warmer weather and longer daylight hours in summer significantly boost bike rental usage, while colder winter conditions lead to a substantial drop in demand. Understanding these trends is crucial for optimizing bike-sharing operations.

![Seasonal Trends](images/seasonal_trends.png)

### Correlation Analysis
The correlation matrix provided valuable insights into the relationships between different variables and the target variable, 'Rented Bike Count'.

- **Positively correlated with**:
  - Temperature (0.54)
  - Hour (0.41)
  - Dew Point Temperature (0.38)
  - Seasons_1 (0.30) (Likely represents Summer)
- **Negatively correlated with**:
  - Seasons_3 (-0.42) (Likely represents Winter)
  - Humidity (-0.20)

![Correlation Matrix](images/correlation_matrix.png)

### Model Performance Comparison
We compared the performance of different regression models using RMSE and R-squared metrics. The results are summarized below:

| Model             | RMSE       | R²        |
|-------------------|------------|-----------|
| Linear Regression | 7.103216   | 0.671092  |
| Decision Tree     | 4.793213   | 0.850232  |
| Random Forest     | 3.804972   | 0.915332  |

## Conclusion
Our analysis of bike sharing demand in Seoul demonstrates the significant impact of weather conditions, seasonal variations, and temporal factors on rental patterns. The insights derived from this study underscore the value of machine learning and data mining in urban planning and the management of bike sharing systems. Predictive models, particularly the Random Forest model, proved highly effective in forecasting demand, offering a robust tool for city planners and service providers to optimize bike availability and improve user satisfaction. This study paves the way for future research and the potential integration of more advanced predictive techniques to further enhance urban mobility solutions.
