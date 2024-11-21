library(forecast)
library(ggplot2)
library(dplyr)
library(tidyr)
library(xgboost)
library(slider)

##### SETUP DATA #####

set.seed(123)

# Generate time series data
time <- 1:200
trend <- 0.05 * time  # Linear trend
seasonality <- 10 * sin(2 * pi * time / 12)  # Monthly seasonality
noise <- rnorm(200, mean = 0, sd = 5)  # Random noise
nonlinear_effect <- ifelse(time > 100, (time - 100)^0.5, 0)  # Nonlinear change after time=100

# Combine components to form the target series
series <- trend + seasonality + nonlinear_effect + noise

# Convert to a data frame
data <- data.frame(time = time, value = series)

# Visualize the data
ggplot(data, aes(x = time, y = value)) +
  geom_line() +
  labs(title = "Generated Time Series Data", x = "Time", y = "Value")



##### ARIMA MODEL #####

# Fit ARIMA model
arima_model <- auto.arima(data$value)

# Forecast using ARIMA
arima_forecast <- forecast(arima_model, h = 20)

# Visualize ARIMA fit
autoplot(arima_forecast) +
  labs(title = "ARIMA Model Fit", x = "Time", y = "Value")

# data not stationary - ARIMA likely would perform well, but data requires transformation to meet model assumptions
# also has seasonal component - ARIMA could fit this, but needs more transformation and configuration


##### MACHINE LEARNING WITH TS FEATURE ENGINEERING #####

# Prepare data for XGBoost
lagged_data <- data %>%
  mutate(lag1 = lag(value, 1), lag2 = lag(value, 2), lag3 = lag(value, 3)) %>%
  na.omit()

train <- lagged_data[1:180, ]
test <- lagged_data[181:200, ]

# Define X and y
X_train <- as.matrix(train[, c("lag1", "lag2", "lag3")])
y_train <- train$value
X_test <- as.matrix(test[, c("lag1", "lag2", "lag3")])
y_test <- test$value

# Fit XGBoost model
xgb_model <- xgboost(data = X_train, label = y_train, nrounds = 100, objective = "reg:squarederror", verbose = 0)

# Predict with XGBoost
xgb_pred <- predict(xgb_model, X_test)

# Visualize XGBoost fit
test <- test %>%
  mutate(predicted = xgb_pred)

ggplot(test, aes(x = time)) +
  geom_line(aes(y = value, color = "Actual")) +
  geom_line(aes(y = predicted, color = "XGBoost Prediction")) +
  labs(title = "XGBoost Model Fit", x = "Time", y = "Value", color = "Legend")





##########################################
########## Multiple time series ##########
##########################################


### set up data

set.seed(42)

dates <- seq(as.Date("2023-01-01"), as.Date("2023-12-31"), by = "days")
ids <- c("A", "B", "C")  # Three time series

# Create time series data with unique patterns for each id
generate_series <- function(id) {
  n <- length(dates)
  trend <- switch(id,
                  "A" = seq(0, 10, length.out = n),  # Linear upward trend
                  "B" = seq(10, 0, length.out = n),  # Linear downward trend
                  "C" = rep(c(0, 5, 10, 5), each = n / 4)  # Cyclical step changes
  )
  
  seasonality <- switch(id,
                        "A" = 5 * sin(2 * pi * (1:n) / 30),  # Smooth monthly seasonality
                        "B" = 3 * cos(2 * pi * (1:n) / 15),  # Faster seasonality
                        "C" = 2 * sin(2 * pi * (1:n) / 60)  # Slower seasonality
  )
  
  noise <- rnorm(n, mean = 0, sd = switch(id, "A" = 2, "B" = 4, "C" = 1))  # Different noise levels
  
  value <- trend + seasonality + noise
  data.frame(id = id, date = dates, value = value)
}

df <- bind_rows(lapply(ids, generate_series))

# Visualize the data
ggplot(df, aes(x = date, y = value, color = id)) +
  geom_line() +
  labs(title = "Generated Time Series Data with Distinct Patterns", x = "Date", y = "Value") +
  theme_minimal()


### Feature engineering

# Create lag features and rolling statistics
df <- df %>%
  group_by(id) %>%
  arrange(date) %>%
  mutate(
    lag_1 = lag(value, 1),
    lag_2 = lag(value, 2),
    lag_3 = lag(value, 3),
    rolling_mean_7 = slide_dbl(value, mean, .before = 6, .complete = TRUE),
    rolling_sd_7 = slide_dbl(value, sd, .before = 6, .complete = TRUE)
  ) %>%
  ungroup() %>%
  drop_na()  # Remove rows with NA values


### train/test split (80/20)

train <- df %>% filter(date < as.Date("2023-10-01"))
test <- df %>% filter(date >= as.Date("2023-10-01"))


### train ML model

# Prepare training and test matrices
train_matrix <- as.matrix(train %>% select(lag_1, lag_2, lag_3, rolling_mean_7, rolling_sd_7))
test_matrix <- as.matrix(test %>% select(lag_1, lag_2, lag_3, rolling_mean_7, rolling_sd_7))

train_labels <- train$value
test_labels <- test$value

# Train XGBoost model
xgb_model <- xgboost(
  data = train_matrix,
  label = train_labels,
  nrounds = 100,
  objective = "reg:squarederror",
  verbose = 0
)

# Predict on test data
test$predicted <- predict(xgb_model, test_matrix)


### Visualize predictions against test data

# Plot predictions vs. actual values
ggplot(test, aes(x = date)) +
  geom_line(aes(y = value, color = "Actual")) +
  geom_line(aes(y = predicted, color = "Predicted")) +
  facet_wrap(~id, scales = "free_y", ncol=1) +
  labs(
    title = "XGBoost Predictions vs Actual Values for Each Series",
    x = "Date",
    y = "Value",
    color = "Legend"
  ) +
  theme_minimal() +
  scale_color_manual(values=c("dodgerblue3","goldenrod"))


### tsfresh package from Python for automated ts feature engineering:
# https://tsfresh.readthedocs.io/en/latest/text/quick_start.html



