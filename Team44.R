#region[Libraries]
library(randomForest)
library(dplyr)
library(caret)
library(gbm)
library(tidyverse)
library(corrplot)
library(xgboost)
library(e1071)
#endregion

#region[Load Data]
setwd("D:\\Kolleya\\8th Term\\Distributed\\Practical\\House-Prices-Problem")

train <- read.csv('train.csv', stringsAsFactors = F)
test <- read.csv('test.csv', stringsAsFactors = F)
test$SalePrice <- rep(NA, nrow(test))

# Combine train and test data to apply the same data preprocessing steps to both datasets
house_prices <- bind_rows(train, test)
#endregion

#region[Drop & Impute]

# Drop variables with too many missing values
na_cols <- colSums(is.na(house_prices)) / nrow(house_prices)  # Calculate the percentage of NA values in each column
cols_to_remove <- names(na_cols[na_cols > 0.5])  # Get the column names with NA percentage exceeding 50%
house_prices <- house_prices[, !names(house_prices) %in% cols_to_remove]  # Remove the columns from the data frame

# Impute missing values for numeric columns
num_cols <- sapply(house_prices[, -which(names(house_prices) == "SalePrice")], is.numeric)
house_prices[, -which(names(house_prices) == "SalePrice")][, num_cols] <- 
  lapply(house_prices[, -which(names(house_prices) == "SalePrice")][, num_cols], function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))

# Impute missing values for non-numeric columns
nonnum_cols <- sapply(house_prices[, -which(names(house_prices) == "SalePrice")], function(x) !is.numeric(x))
house_prices[, -which(names(house_prices) == "SalePrice")][, nonnum_cols] <- 
  lapply(house_prices[, -which(names(house_prices) == "SalePrice")][, nonnum_cols], function(x) ifelse(is.na(x), as.character(mode(x)), x))
#endregion

#region[Visualization]
train <- subset(house_prices, !is.na(SalePrice))

#Histogram: SalePrice
ggplot(train, aes(x = SalePrice)) +
  geom_histogram(fill = "skyblue", color = "black") +
  labs(x = "Sale Price", y = "Frequency", title = "Distribution of Sale Prices")

# Violin Plot: SalePrice by Neighborhood
train$Neighborhood <- factor(train$Neighborhood, levels = c("MeadowV", "IDOTRR", "BrDale", "OldTown", "Edwards", "BrkSide", "Sawyer", "Blueste", "SWISU", "NAmes", "NPkVill", "Mitchel", "SawyerW", "NWAmes", "Gilbert", "Blmngtn", "CollgCr", "Crawfor", "ClearCr", "Somerst", "Veenker", "Timber", "StoneBr", "NridgHt", "NoRidge"))

ggplot(train, aes(x = Neighborhood, y = SalePrice)) +
  geom_violin(fill = "skyblue", color = "black") +
  labs(x = "Neighborhood", y = "Sale Price", title = "Sale Price by Neighborhood") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

#Boxplot: SalePrice by OverallQual
ggplot(train, aes(x = as.factor(OverallQual), y = SalePrice)) +
  geom_boxplot(fill = "skyblue", color = "black") +
  labs(x = "Overall Quality", y = "Sale Price", title = "Sale Price by Overall Quality")

#Boxplot: SalePrice by ExterQual
train$ExterQual <- factor(train$ExterQual, levels = c("Po", "Fa", "TA", "Gd", "Ex"))
ggplot(train, aes(x = as.factor(ExterQual), y = SalePrice)) +
  geom_boxplot(fill = "skyblue", color = "black") +
  labs(x = "External Quality", y = "Sale Price", title = "Sale Price by External Quality")

#Scatter Plot: SalePrice vs. GrLivArea
ggplot(train, aes(x = GrLivArea, y = SalePrice)) +
  geom_point(color = "skyblue") +
  labs(x = "GrLivArea", y = "Sale Price", title = "Sale Price vs. GrLivArea")

#Bar Plot: SaleType
ggplot(train, aes(x = SaleType)) +
  geom_bar(fill = "skyblue", color = "black") +
  labs(x = "Sale Type", y = "Count", title = "Distribution of Sale Types")
#endregion

#region[Feature Conversion]

# Convert character columns to factors
char_cols <- sapply(house_prices, is.character)
house_prices[char_cols] <- lapply(house_prices[char_cols], as.factor)

# Convert remaining non-numeric columns to numeric
non_numeric_cols <- sapply(house_prices, function(x) !is.numeric(x))
house_prices[non_numeric_cols] <- lapply(house_prices[non_numeric_cols], as.numeric)
#endregion

#region[Data Split]

# Split the data into train and test sets
train <- subset(house_prices, !is.na(SalePrice))
test <- subset(house_prices, is.na(SalePrice))

# Extract the Id column from the original house_prices dataset
test_ids <- house_prices$Id[is.na(house_prices$SalePrice)]
#endregion

#region[Correlation Matrix]
cor_matrix <- cor(train[, sapply(train, is.numeric)], use = "complete.obs")
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.7)
#endregion

#region[Feature Selection]
# highly_correlated <- findCorrelation(cor_matrix, cutoff = 0.2, verbose = FALSE)
# highly_correlated_cols <- colnames(cor_matrix)[highly_correlated]
# train <- train[,  highly_correlated_cols]
# test <- test[, highly_correlated_cols]
#endregion

#region[Models Training]
label <- train$SalePrice # Extract the SalePrice column
set.seed(45) # Init seed for all models

## Linear Regression Model #####################
modelLR <- lm(SalePrice ~ ., data = train)
# print(modelLR)
predictionsLR <- predict(modelLR, newdata = test)

## SVR Model ###################################
modelSVR <- svm(
  x = as.matrix(train[, -which(names(train) == "SalePrice")]),
  y = label,
  kernel = "radial",
  cost = 20,
  gamma = 0.001
)
# print(modelSVR)
predictionsSVR <- predict(modelSVR, newdata = as.matrix(test[, -which(names(test) == "SalePrice")]))

## XGBOOST Model ###############################
params <- list(
  objective = "reg:squarederror",
  eta = 0.01,
  max_depth = 5,
  colsample_bytree = 0.5,
  subsample = 0.5
)

modelXG <- xgboost(
  data = as.matrix(train[, -which(names(train) == "SalePrice")]),
  label = label,
  params = params,
  nrounds = 1000
)

predictionsXG <- predict(modelXG, newdata = as.matrix(test[, -which(names(test) == "SalePrice")]))
#endregion

#region[Save to CSV]
submissionLR <- data.frame(Id = test_ids, SalePrice = predictionsLR)
write.csv(submissionLR, "predictions_linear_regression.csv", row.names = FALSE)

submissionSVR <- data.frame(Id = test_ids, SalePrice = predictionsSVR)
write.csv(submissionSVR, "predictions_SVR.csv", row.names = FALSE)

submissionXG <- data.frame(Id = test_ids, SalePrice = predictionsXG)
write.csv(submissionXG, "predictions_XGBoost.csv", row.names = FALSE)
#endregion

#region[Visualize Predictions]
prediction_data <- data.frame(Predicted = predictionsSVR)

# Plot the histogram of predicted values
ggplot(prediction_data, aes(x = Predicted)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "white") +
  labs(x = "Predicted SalePrice", y = "Count") +
  ggtitle("Distribution of Predicted SalePrice")

ggplot(prediction_data, aes(y = Predicted)) +
  geom_boxplot() +
  ylab("Predicted SalePrice") +
  ggtitle("Box Plot of Predicted SalePrice")
#endregion