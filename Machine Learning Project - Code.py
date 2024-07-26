#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00560/SeoulBikeData.csv'
data = pd.read_csv(url, encoding='ISO-8859-1')

# Display the first few rows of the dataset
data.head()

# Check for missing values
print(data.isnull().sum())


# In[2]:


#The code begins by importing essential libraries for data analysis, preprocessing, and modeling, including NumPy, pandas, seaborn, sklearn, and statsmodels. It then loads the Seoul Bike Sharing Demand dataset from a specified URL and displays the first few rows to provide an overview of its structure. The dataset comprises various features such as temperature, humidity, wind speed, visibility, and more, alongside the target variable 'Rented Bike Count'. An initial check for missing values in the dataset reveals that there are no missing values in any of the columns. This preliminary analysis indicates that the dataset is clean and ready for further analysis and modeling.


# In[3]:


# Lets look at the average graph of seasons

import matplotlib.pyplot as plt
import seaborn as sns

# Group data by 'Seasons' and calculate average rental numbers
seasonal_data = data.groupby('Seasons')['Rented Bike Count'].mean()

# Plotting the seasonal trends
plt.figure(figsize=(10, 5))
seasonal_data.plot(kind='bar')
plt.title('Average Bike Rental Demand by Season')
plt.xlabel('Season')
plt.ylabel('Average Bike Rentals')
plt.show()


# In[4]:


#The provided code generates a bar plot to visualize the average bike rental demand across different seasons. By grouping the dataset by the 'Seasons' column and calculating the mean of the 'Rented Bike Count' for each season, the code highlights seasonal trends in bike rentals. The resulting plot reveals that bike rentals peak during the summer, followed by autumn and spring, with winter showing the lowest demand. This pattern suggests that warmer weather and possibly longer daylight hours in summer significantly boost bike rental usage, while colder winter conditions lead to a substantial drop in demand. This seasonal analysis is crucial for understanding user behavior and can aid in planning and optimizing bike-sharing operations.


# In[5]:


# Now let us run an OLS model to understand the model fit, important co-efficients to be considered for identifying the best features.

# First lets Identify categorical variables and encode them
categorical_vars = data.select_dtypes(include=['object']).columns.tolist()
print("Categorical Variables:", categorical_vars)
# Encoding 'Seasons'
data.loc[data['Seasons'] == 'Spring', 'Seasons'] = 0
data.loc[data['Seasons'] == 'Summer', 'Seasons'] = 1
data.loc[data['Seasons'] == 'Autumn', 'Seasons'] = 2
data.loc[data['Seasons'] == 'Winter', 'Seasons'] = 3
# Encoding 'Holiday'
data.loc[data['Holiday'] == 'No Holiday', 'Holiday'] = 0
data.loc[data['Holiday'] == 'Holiday', 'Holiday'] = 1
# Encoding 'Functioning Day'
data.loc[data['Functioning Day'] == 'No', 'Functioning Day'] = 0
data.loc[data['Functioning Day'] == 'Yes', 'Functioning Day'] = 1
# Drop 'Date' and 'Rented Bike Count' columns
X = data.drop(columns=['Date', 'Rented Bike Count'])
y = data['Rented Bike Count']

# Now lets run the OLS Regression model


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression
lr_params = {'fit_intercept': [True, False]}
lr_grid = GridSearchCV(LinearRegression(), lr_params, cv=5)
lr_grid.fit(X_train_scaled, y_train)
lr_best_params = lr_grid.best_params_
lr_best_model = lr_grid.best_estimator_
lr_pred = lr_best_model.predict(X_test_scaled)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2 = r2_score(y_test, lr_pred)

# Fit Ordinary Least Squares (OLS) model
X_train_ols = sm.add_constant(X_train_scaled)  # Add a constant term to the features
X_test_ols = sm.add_constant(X_test_scaled)

ols_model = sm.OLS(y_train, X_train_ols)
ols_results = ols_model.fit()

# Print summary of OLS results
print(ols_results.summary())


# In[6]:


#R-squared Value: The R-squared value is 0.529, indicating that approximately 52.9% of the variance in the 'Rented Bike Count' is explained by the predictors in the model. This suggests a moderate fit, as just over half of the variability in bike rentals is accounted for by the model.
#x1 (Temperature): Positive coefficient (196.0379) with a p-value of 0.000, indicating higher temperatures are associated with increased bike rentals.
#x2 (Humidity): Positive coefficient (281.2083) with a p-value of 0.000, suggesting higher humidity is also linked to more bike rentals.
#x3 (Wind Speed): Negative coefficient (-180.0759) with a p-value of 0.000, indicating that higher wind speeds reduce bike rentals.
#x8, x10, and other predictors: Several other variables, such as x8 (visibility) and x10 (holiday), show significant negative impacts with p-values of 0.000, suggesting these factors decrease bike rentals.


# In[7]:


# Generate predictions for the training set using the OLS model
ols_train_pred = ols_results.predict(X_train_ols)

# Calculate residuals for the training set
ols_residuals = y_train - ols_train_pred

# Plotting residuals vs. fitted values
plt.figure(figsize=(10, 5))
plt.scatter(ols_train_pred, ols_residuals, alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.show()


# In[8]:


#The residuals versus fitted values plot provides insight into the performance and assumptions of the OLS regression model. The plot shows the residuals (errors) on the vertical axis and the fitted values (predicted bike rentals) on the horizontal axis. Ideally, residuals should be randomly distributed around the horizontal axis (residual = 0) with no discernible pattern, indicating that the model's errors are unbiased and homoscedastic (having constant variance). However, in this plot, there is a clear funnel shape, with residuals spreading out more as the fitted values increase. This suggests heteroscedasticity, where the variance of the errors increases with the predicted values, indicating potential issues with the model's assumptions and suggesting that further model refinement or transformation of variables may be needed to improve predictive accuracy.


# In[9]:


# Plotting actual vs predicted rented bike count
plt.figure(figsize=(8, 6))
plt.scatter(y_test, lr_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.title('Actual vs Predicted Rented Bike Count')
plt.xlabel('Actual Rented Bike Count')
plt.ylabel('Predicted Rented Bike Count')
plt.grid(True)
plt.show()


# In[10]:


#In this plot, while there is some alignment, especially in the mid-range of values, there is considerable scatter and deviation from the line, particularly at higher rental counts. This indicates that while the model captures some general trends in bike rentals, its accuracy diminishes for higher values, suggesting potential issues with the model's capacity to predict extreme values accurately. This discrepancy highlights the need for further model refinement, potentially through feature engineering, addressing heteroscedasticity, or using more complex models to improve prediction accuracy.


# In[11]:


# Plotting Q-Q plot
plt.figure(figsize=(10, 5))
sm.qqplot(ols_residuals, line='45', fit=True)
plt.title('Q-Q Plot of OLS Residuals')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.show()


# In[12]:


#The data points are spread but show a pattern of deviation from the line as the actual counts increase, indicating potential heteroscedasticity.
#There are some predictions that go into the negative, which is not practical for counts hence we will run log-transform to the target variable.
#The Q-Q plot assesses whether the OLS residuals follow a normal distribution. Ideally, the points should lie along the 45-degree reference line. In this plot, the residuals align with the line in the middle range but deviate significantly at both tails, indicating heavier tails than a normal distribution. This suggests that while the residuals are approximately normal in the middle, there are more extreme values than expected, indicating potential issues with the normality assumption of the OLS model.


# In[13]:


# Identify and encode categorical variables
categorical_vars = data.select_dtypes(include=['object']).columns.tolist()
print("Categorical Variables:", categorical_vars)

# Encoding categorical variables
data.loc[data['Seasons'] == 'Spring', 'Seasons'] = 0
data.loc[data['Seasons'] == 'Summer', 'Seasons'] = 1
data.loc[data['Seasons'] == 'Autumn', 'Seasons'] = 2
data.loc[data['Seasons'] == 'Winter', 'Seasons'] = 3

data.loc[data['Holiday'] == 'No Holiday', 'Holiday'] = 0
data.loc[data['Holiday'] == 'Holiday', 'Holiday'] = 1

data.loc[data['Functioning Day'] == 'No', 'Functioning Day'] = 0
data.loc[data['Functioning Day'] == 'Yes', 'Functioning Day'] = 1

# Drop 'Date' and log-transform the target variable 'Rented Bike Count'
X = data.drop(columns=['Date', 'Rented Bike Count'])
y = np.log1p(data['Rented Bike Count'])  # log1p is log(1 + x) which handles log(0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression with GridSearchCV
lr_params = {'fit_intercept': [True, False]}
lr_grid = GridSearchCV(LinearRegression(), lr_params, cv=5)
lr_grid.fit(X_train_scaled, y_train)
lr_best_params = lr_grid.best_params_
lr_best_model = lr_grid.best_estimator_
lr_pred = lr_best_model.predict(X_test_scaled)

# Inverse log transformation for predictions
lr_pred_exp = np.expm1(lr_pred)
y_test_exp = np.expm1(y_test)

# Calculate RMSE and R2 score
lr_rmse = np.sqrt(mean_squared_error(y_test_exp, lr_pred_exp))
lr_r2 = r2_score(y_test_exp, lr_pred_exp)

print(f'Linear Regression RMSE: {lr_rmse}')
print(f'Linear Regression R2: {lr_r2}')

# Fit Ordinary Least Squares (OLS) model
X_train_ols = sm.add_constant(X_train_scaled)  # Add a constant term to the features
X_test_ols = sm.add_constant(X_test_scaled)

ols_model = sm.OLS(y_train, X_train_ols)
ols_results = ols_model.fit()

# Print summary of OLS results
print(ols_results.summary())


# In[14]:


# Plotting actual vs predicted rented bike count
plt.figure(figsize=(8, 6))
plt.scatter(y_test, lr_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.title('Actual vs Predicted Rented Bike Count')
plt.xlabel('Actual Rented Bike Count')
plt.ylabel('Predicted Rented Bike Count')
plt.grid(True)
plt.show()


# In[ ]:


#This plot compares the actual and predicted rented bike counts. The points are closer to the red line of perfect prediction compared to the previous model, indicating better accuracy in predictions. However, there are still some deviations, particularly for higher counts, suggesting that while the model has improved, it still has some limitations in predicting extreme values.


# In[15]:


# Predicting using OLS model
y_train_pred_ols = ols_results.predict(X_train_ols)
y_test_pred_ols = ols_results.predict(X_test_ols)

# Residuals
train_residuals = y_train - y_train_pred_ols
test_residuals = y_test - y_test_pred_ols

# Plot Residuals vs Fitted Values
plt.figure(figsize=(10, 6))
plt.scatter(y_test_pred_ols, test_residuals, color='blue', alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.show()


# In[16]:


#The residuals vs fitted values plot shows a more concentrated pattern around zero compared to the previous model, indicating less heteroscedasticity. However, there are still distinct clusters and some outliers, suggesting potential issues with variance that could affect the model's reliability. The presence of a linear pattern in residuals also suggests possible omitted variables or non-linear relationships not captured by the model.


# In[17]:


# Q-Q Plot of residuals
sm.qqplot(test_residuals, line='45')
plt.title('Q-Q Plot of OLS Residuals')
plt.show()


# In[18]:


#The Q-Q plot indicates that while the residuals are closer to the theoretical quantiles of a normal distribution compared to the previous model, there are still deviations at the tails. This suggests that the residuals are not perfectly normally distributed, which could affect the accuracy of confidence intervals and hypothesis tests derived from the model.


# In[20]:


# Identify and encode categorical variables
categorical_vars = data.select_dtypes(include=['object']).columns.tolist()
print("Categorical Variables:", categorical_vars)

# Encoding categorical variables
data.loc[data['Seasons'] == 'Spring', 'Seasons'] = 0
data.loc[data['Seasons'] == 'Summer', 'Seasons'] = 1
data.loc[data['Seasons'] == 'Autumn', 'Seasons'] = 2
data.loc[data['Seasons'] == 'Winter', 'Seasons'] = 3

data.loc[data['Holiday'] == 'No Holiday', 'Holiday'] = 0
data.loc[data['Holiday'] == 'Holiday', 'Holiday'] = 1

data.loc[data['Functioning Day'] == 'No', 'Functioning Day'] = 0
data.loc[data['Functioning Day'] == 'Yes', 'Functioning Day'] = 1

# Drop 'Date' and apply square root transformation to the target variable 'Rented Bike Count'
X = data.drop(columns=['Date', 'Rented Bike Count'])
y = np.sqrt(data['Rented Bike Count'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression with GridSearchCV
lr_params = {'fit_intercept': [True, False]}
lr_grid = GridSearchCV(LinearRegression(), lr_params, cv=5)
lr_grid.fit(X_train_scaled, y_train)
lr_best_params = lr_grid.best_params_
lr_best_model = lr_grid.best_estimator_
lr_pred = lr_best_model.predict(X_test_scaled)

# Inverse square root transformation for predictions
lr_pred_exp = np.square(lr_pred)
y_test_exp = np.square(y_test)

# Calculate RMSE and R2 score
lr_rmse = np.sqrt(mean_squared_error(y_test_exp, lr_pred_exp))
lr_r2 = r2_score(y_test_exp, lr_pred_exp)

print(f'Linear Regression RMSE: {lr_rmse}')
print(f'Linear Regression R2: {lr_r2}')

# Fit Ordinary Least Squares (OLS) model
X_train_ols = sm.add_constant(X_train_scaled)  # Add a constant term to the features
X_test_ols = sm.add_constant(X_test_scaled)

ols_model = sm.OLS(y_train, X_train_ols)
ols_results = ols_model.fit()

# Print summary of OLS results
print(ols_results.summary())



# In[21]:


# Plot actual vs. predicted values
plt.figure(figsize=(10, 5))
plt.scatter(y_test_exp, lr_pred_exp, alpha=0.3)
plt.plot([y_test_exp.min(), y_test_exp.max()], [y_test_exp.min(), y_test_exp.max()], 'k--', lw=2)
plt.title('Actual vs Predicted Bike Rentals')
plt.xlabel('Actual Bike Rentals')
plt.ylabel('Predicted Bike Rentals')
plt.show()


# In[22]:


#The "Actual vs Predicted Bike Rentals" plot shows the relationship between actual and predicted values. The spread of points around the line of perfect prediction (black dashed line) indicates that the model predicts the lower range of bike rentals reasonably well but struggles with higher counts. The deviations at higher values suggest that the model may not be accurately capturing the extremes, leading to under- or over-predictions.


# In[23]:


# Q-Q Plot of residuals
sm.qqplot(test_residuals, line='45')
plt.title('Q-Q Plot of OLS Residuals')
plt.show()


# In[24]:


#The Q-Q plot of OLS residuals shows the residuals' quantiles against the theoretical quantiles of a normal distribution. The plot reveals some deviation from the red diagonal line, particularly at the tails, indicating that the residuals are not perfectly normally distributed. This non-normality at the extremes could affect the model's performance in accurately predicting the bike rentals.


# In[25]:


# Residuals
train_residuals = y_train - y_train_pred_ols
test_residuals = y_test - y_test_pred_ols

# Plot Residuals vs Fitted Values
plt.figure(figsize=(10, 6))
plt.scatter(y_test_pred_ols, test_residuals, color='blue', alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.show()


# In[26]:


#The "Residuals vs Fitted Values" plot shows residuals against predicted values. There is a clear pattern and clustering of residuals, especially as fitted values increase. This suggests heteroscedasticity and potential model misspecification, indicating that the variance of the residuals is not constant. The presence of outliers further suggests that the model may not fully capture the underlying data structure, impacting its overall predictive power.


# In[27]:


#The Log Transformation Model is the best among the three. It has the highest R-squared value, indicating it explains the most variance in bike rentals. Additionally, it shows improvements in both residual plots and Q-Q plots compared to the initial model, though it still has some minor issues that could be further addressed. This model provides a better balance between fit and assumptions of the OLS regression.


# In[28]:


# Now inorder to address the issue of multicolinearity. Now since 'Solar Radiation','Temperature' and 'Snowfall'have higher p-values and would lead to multicolinearity in our model,we will apply backward elimination method and will drop highest p-value co-efficients one by one.


# In[29]:


#In the log transformation model solar Radiation(x7) has the highest p-value among all (0.802),hence we will drop it and run ols model.

# Drop  'Solar Radiation' column as it has the highest p-value
data = data.drop(columns=['Solar Radiation (MJ/m2)'])

# Drop 'Date' and 'Rented Bike Count' columns
X = data.drop(columns=['Date', 'Rented Bike Count'])
y = np.sqrt(data['Rented Bike Count'])

data.head()


# In[30]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression with GridSearchCV
lr_params = {'fit_intercept': [True, False]}
lr_grid = GridSearchCV(LinearRegression(), lr_params, cv=5)
lr_grid.fit(X_train_scaled, y_train)
lr_best_params = lr_grid.best_params_
lr_best_model = lr_grid.best_estimator_
lr_pred = lr_best_model.predict(X_test_scaled)

# Inverse log transformation for predictions
lr_pred_exp = np.expm1(lr_pred)
y_test_exp = np.expm1(y_test)

# Calculate RMSE and R2 score
lr_rmse = np.sqrt(mean_squared_error(y_test_exp, lr_pred_exp))
lr_r2 = r2_score(y_test_exp, lr_pred_exp)

print(f'Linear Regression RMSE: {lr_rmse}')
print(f'Linear Regression R2: {lr_r2}')

# Fit Ordinary Least Squares (OLS) model
X_train_ols = sm.add_constant(X_train_scaled)  # Add a constant term to the features
X_test_ols = sm.add_constant(X_test_scaled)

ols_model = sm.OLS(y_train, X_train_ols)
ols_results = ols_model.fit()

# Print summary of OLS results
print(ols_results.summary())


# In[31]:


# Drop  'Snowfall' column as it has the highest p-value
data = data.drop(columns=['Snowfall (cm)'])

# Drop 'Date' and 'Rented Bike Count' columns
X = data.drop(columns=['Date', 'Rented Bike Count'])
y = np.sqrt(data['Rented Bike Count'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression with GridSearchCV
lr_params = {'fit_intercept': [True, False]}
lr_grid = GridSearchCV(LinearRegression(), lr_params, cv=5)
lr_grid.fit(X_train_scaled, y_train)
lr_best_params = lr_grid.best_params_
lr_best_model = lr_grid.best_estimator_
lr_pred = lr_best_model.predict(X_test_scaled)

# Inverse log transformation for predictions
lr_pred_exp = np.expm1(lr_pred)
y_test_exp = np.expm1(y_test)

# Calculate RMSE and R2 score
lr_rmse = np.sqrt(mean_squared_error(y_test_exp, lr_pred_exp))
lr_r2 = r2_score(y_test_exp, lr_pred_exp)

print(f'Linear Regression RMSE: {lr_rmse}')
print(f'Linear Regression R2: {lr_r2}')

# Fit Ordinary Least Squares (OLS) model
X_train_ols = sm.add_constant(X_train_scaled)  # Add a constant term to the features
X_test_ols = sm.add_constant(X_test_scaled)

ols_model = sm.OLS(y_train, X_train_ols)
ols_results = ols_model.fit()

# Print summary of OLS results
print(ols_results.summary())


# In[32]:


# Generate predictions for the training set using the OLS model
ols_train_pred = ols_results.predict(X_train_ols)

# Calculate residuals for the training set
ols_residuals = y_train - ols_train_pred

# Plotting residuals vs. fitted values
plt.figure(figsize=(10, 5))
plt.scatter(ols_train_pred, ols_residuals, alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.show()


# In[33]:


#Now that all the co-eeficients with higher p-values are dropped let us interpret our OLS model.


# In[34]:


# Plotting Q-Q plot
plt.figure(figsize=(10, 5))
sm.qqplot(ols_residuals, line='45', fit=True)
plt.title('Q-Q Plot of OLS Residuals')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.show()


# In[ ]:


#We can see that even after dropping insignificant co-efficients it is not helping to achieve a better result, hence we will apply regularization models to detect which one gives a better result.


# In[35]:


# Encode categorical variable (Seasons)
data = pd.get_dummies(data, columns=['Seasons'], drop_first=False)

# Define the target variable y
y = data['Rented Bike Count']

# Exclude 'Hour', 'Holiday', and 'Functioning Day' from X features
X = data.drop(['Rented Bike Count', 'Date', 'Holiday','Hour'], axis=1)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)


# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#We'll analyze the correlation between weather variables and bike rental count:


# Calculate correlation matrix
corr_matrix = data.corr(numeric_only=True)

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Print correlation matrix
print("Correlation Matrix:")
corr_matrix


# In[ ]:


#The correlation heatmap illustrates the relationships between various variables, with the correlation coefficients ranging from -1 to 1. Here's a detailed interpretation:
#Rented Bike Count:
#Positively correlated with:

#Temperature (0.54): More bikes are rented as the temperature rises.
#Hour (0.41): Certain hours of the day see more rentals.
#Dew point temperature (0.38): Higher dew points, indicating more humid air, correlate with higher bike rentals.
#Seasons_1 (0.30): This likely represents a specific season that has more bike rentals.

#Negatively correlated with:
#Seasons_3 (-0.42): This likely represents a specific season with fewer bike rentals.
#Humidity (-0.20): Higher humidity is associated with fewer bike rentals.


# In[36]:


data.head()


# In[37]:


# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')

# Extract month from date
data['Month'] = data['Date'].dt.month

# Dictionary to hold data for each month
monthly_data = {month: data[data['Month'] == month].copy() for month in range(1, 13)}

# Dictionaries to hold train and test data for each month
train_data = {}
test_data = {}

# Perform train-test split for each month and store in dictionaries
for month, month_data in monthly_data.items():
    X = month_data.drop(columns=['Date', 'Rented Bike Count'])
    y = np.sqrt(month_data['Rented Bike Count'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_data[month] = {'X': X_train, 'y': y_train}
    test_data[month] = {'X': X_test, 'y': y_test}

# Concatenate all the train and test data across months
merged_train_X = pd.concat([train_data[month]['X'] for month in range(1, 13)], axis=0)
merged_train_y = pd.concat([train_data[month]['y'] for month in range(1, 13)], axis=0)
merged_test_X = pd.concat([test_data[month]['X'] for month in range(1, 13)], axis=0)
merged_test_y = pd.concat([test_data[month]['y'] for month in range(1, 13)], axis=0)

# Encode categorical variables if needed
merged_train_X_encoded = pd.get_dummies(merged_train_X)
merged_test_X_encoded = pd.get_dummies(merged_test_X)

# Ensure the columns in the training and testing datasets are aligned
merged_train_X_encoded, merged_test_X_encoded = merged_train_X_encoded.align(merged_test_X_encoded, join='outer', axis=1, fill_value=0)

# Standardize the features if necessary
scaler = StandardScaler()
merged_train_X_scaled = scaler.fit_transform(merged_train_X_encoded)
merged_test_X_scaled = scaler.transform(merged_test_X_encoded)


# In[ ]:


#Now we start by converting the 'Date' column in the dataset to a datetime format and extract the month into a new 'Month' column. Next, we divide the dataset by month, storing each month's data in a dictionary. For each month's data, we perform a train-test split, allocating 80% for training and 20% for testing. We then concatenate all the monthly train and test data into single datasets. To handle categorical variables, we use one-hot encoding to convert them into numerical format. We ensure that the training and testing datasets have the same columns by aligning them and filling any missing columns with zeros. Finally, we standardize the features to have a mean of zero and a standard deviation of one. This preparation ensures that the data is clean, consistent, and ready for effective machine learning model training and evaluation.


# In[48]:


# Perform log transformation on the target variable
merged_train_y_log = np.log1p(merged_train_y)  # log1p is used to avoid log(0)
merged_test_y_log = np.log1p(merged_test_y)

# Hyperparameter tuning for Linear Regression
lr_param_grid = {'fit_intercept': [True, False]}
lr_grid_search = GridSearchCV(LinearRegression(), lr_param_grid, cv=5)
lr_grid_search.fit(merged_train_X_scaled, merged_train_y_log)  # Fit using the log-transformed target
lr_best_model = lr_grid_search.best_estimator_

# Predict and inverse log transformation on predictions
lr_pred_log = lr_best_model.predict(merged_test_X_scaled)
lr_pred = np.expm1(lr_pred_log)  # Use expm1 to reverse the log1p transformation

# Calculate performance metrics
lr_rmse = np.sqrt(mean_squared_error(merged_test_y, lr_pred))
lr_r2 = r2_score(merged_test_y, lr_pred)

# Output the results
print(f"Best Linear Regression Model: {lr_best_model}")
print(f"RMSE: {lr_rmse}")
print(f"R^2: {lr_r2}")


# In[39]:


# Apply log transformation to the target variable
merged_train_y_log = np.log1p(merged_train_y)  # log1p is log(1 + x) to handle zero values
merged_test_y_log = np.log1p(merged_test_y)

# Define the hyperparameter grid for Decision Tree
dt_param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV with DecisionTreeRegressor
dt_grid_search = GridSearchCV(DecisionTreeRegressor(random_state=42), dt_param_grid, cv=5)

# Fit the model with the log-transformed target variable
dt_grid_search.fit(merged_train_X_scaled, merged_train_y_log)

# Get the best estimator
dt_best_model = dt_grid_search.best_estimator_

# Predict on the test set
dt_pred_log = dt_best_model.predict(merged_test_X_scaled)

# Inverse log transform the predictions
dt_pred = np.expm1(dt_pred_log)  # expm1 is exp(x) - 1 to reverse log1p

# Calculate RMSE and R^2 score
dt_rmse = np.sqrt(mean_squared_error(merged_test_y, dt_pred))
dt_r2 = r2_score(merged_test_y, dt_pred)

print(f"RMSE: {dt_rmse}")
print(f"R^2 Score: {dt_r2}")


# In[40]:


# Apply log transformation to the target variable
merged_train_y_log = np.log1p(merged_train_y)  # log1p is log(1 + x) to handle zero values
merged_test_y_log = np.log1p(merged_test_y)

# Define the hyperparameter grid for Random Forest
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2', 1.0]
}

# Initialize GridSearchCV with RandomForestRegressor
rf_grid_search = GridSearchCV(RandomForestRegressor(random_state=42), rf_param_grid, cv=5, n_jobs=-1)

# Fit the model with the log-transformed target variable
rf_grid_search.fit(merged_train_X_scaled, merged_train_y_log)

# Get the best estimator
rf_best_model = rf_grid_search.best_estimator_

# Predict on the test set
rf_pred_log = rf_best_model.predict(merged_test_X_scaled)

# Inverse log transform the predictions
rf_pred = np.expm1(rf_pred_log)  # expm1 is exp(x) - 1 to reverse log1p

# Calculate RMSE and R^2 score
rf_rmse = np.sqrt(mean_squared_error(merged_test_y, rf_pred))
rf_r2 = r2_score(merged_test_y, rf_pred)

# Print the best parameters and performance metrics
print("Best Parameters for Random Forest:", rf_grid_search.best_params_)
print("RMSE:", rf_rmse)
print("R2 Score:", rf_r2)


# In[41]:


# Calculate residuals
rf_residuals = merged_test_y - rf_pred

# Plot residuals vs fitted values
plt.figure(figsize=(10, 5))
plt.scatter(rf_pred, rf_residuals, alpha=0.3)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.show()


# In[42]:


# Actual vs Predicted Plot
plt.figure(figsize=(10, 5))
plt.scatter(merged_test_y, rf_pred, alpha=0.3)
plt.plot([merged_test_y.min(), merged_test_y.max()], [merged_test_y.min(), merged_test_y.max()], 'k--', lw=2)
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()


# In[43]:


# Apply log transformation to the target variable
merged_train_y_log = np.log1p(merged_train_y)  # log1p is log(1 + x) to handle zero values
merged_test_y_log = np.log1p(merged_test_y)

# Define the hyperparameter grid for Lasso
lasso_param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10]}

# Initialize GridSearchCV with Lasso
lasso_grid_search = GridSearchCV(Lasso(random_state=42), lasso_param_grid, cv=5)

# Fit the model with the log-transformed target variable
lasso_grid_search.fit(merged_train_X_scaled, merged_train_y_log)

# Get the best estimator
lasso_best_model = lasso_grid_search.best_estimator_

# Predict on the test set
lasso_pred_log = lasso_best_model.predict(merged_test_X_scaled)

# Inverse log transform the predictions
lasso_pred = np.expm1(lasso_pred_log)  # expm1 is exp(x) - 1 to reverse log1p

# Calculate RMSE and R^2 score
lasso_rmse = np.sqrt(mean_squared_error(merged_test_y, lasso_pred))
lasso_r2 = r2_score(merged_test_y, lasso_pred)

# Print the best parameters and performance metrics
print("Best Parameters for Lasso:", lasso_grid_search.best_params_)
print("RMSE:", lasso_rmse)
print("R2 Score:", lasso_r2)


# In[44]:


# Define the hyperparameter grid for Ridge
ridge_param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10]}

# Initialize GridSearchCV with Ridge
ridge_grid_search = GridSearchCV(Ridge(random_state=42), ridge_param_grid, cv=5)

# Fit the model with the log-transformed target variable
ridge_grid_search.fit(merged_train_X_scaled, merged_train_y_log)

# Get the best estimator
ridge_best_model = ridge_grid_search.best_estimator_

# Predict on the test set
ridge_pred_log = ridge_best_model.predict(merged_test_X_scaled)

# Inverse log transform the predictions
ridge_pred = np.expm1(ridge_pred_log)  # expm1 is exp(x) - 1 to reverse log1p

# Calculate RMSE and R^2 score
ridge_rmse = np.sqrt(mean_squared_error(merged_test_y, ridge_pred))
ridge_r2 = r2_score(merged_test_y, ridge_pred)

# Print the best parameters and performance metrics
print("Best Parameters for Ridge:", ridge_grid_search.best_params_)
print("RMSE:", ridge_rmse)
print("R2 Score:", ridge_r2)


# In[45]:


# Create a DataFrame to store the results
results_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Decision Tree', 'Random Forest', 'Lasso Regression', 'Ridge Regression'],
    'RMSE': [lr_rmse, dt_rmse, rf_rmse, lasso_rmse, ridge_rmse],
    'R2': [lr_r2, dt_r2, rf_r2, lasso_r2, ridge_r2]
})


# In[46]:


# Display the results DataFrame
print("\nModel Performance Comparison:")
print(results_df)


# In[47]:


# Get feature importances from Random Forest model
rf_feature_importances = rf_best_model.feature_importances_

# Create a Series with feature importances and corresponding feature names
feature_importance_series = pd.Series(rf_feature_importances, index=merged_train_X_encoded.columns)

# Sort feature importances in descending order
feature_importance_series_sorted = feature_importance_series.sort_values(ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
feature_importance_series_sorted.plot(kind='bar')
plt.title('Feature Importance - Random Forest')
plt.xlabel('Features')
plt.ylabel('Feature Importance')
plt.show()


# In[ ]:


#The graph displays the feature importance scores from a Random Forest model used to predict a target variable. 
#The most influential features are 'Functioning Day_0' and 'Functioning Day_1', with importance scores of approximately 0.30 and 0.20 respectively, indicating their significant impact on the model's predictions. 
#Other notable features include 'Temperature(°C)', 'Hour', 'Rainfall(mm)', and 'Humidity(%)', with decreasing importance. Features such as 'Seasons_2', 'Visibility(10m)', and 'Month' have relatively lower importance, while 'Seasons_0' and 'Seasons_1' have the least influence on the model's output.


# In[ ]:





# In[ ]:





# In[ ]:




