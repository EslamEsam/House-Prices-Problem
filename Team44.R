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

#region[Load & Combine Data]
setwd("D:\\Kolleya\\8th Term\\Distributed\\Practical\\House-Prices-Problem")

train_original <- read.csv('train.csv', stringsAsFactors = F)
test_original <- read.csv('test.csv', stringsAsFactors = F)

test_original$SalePrice <- rep(NA, nrow(test_original))

# Combine train and test data to apply the same data preprocessing steps to both datasets
combined_dataset <- bind_rows(train_original, test_original)
#endregion

#region[Drop & Impute]

# Drop variables with too many missing values
na_cols <- colSums(is.na(combined_dataset)) / nrow(combined_dataset)  # Calculate the percentage of NA values in each column
cols_to_remove <- names(na_cols[na_cols > 0.5])  # Get the column names with NA percentage exceeding 50%
house_prices <- combined_dataset[, !names(combined_dataset) %in% cols_to_remove]  # Remove the columns from the data frame

# Impute missing values for numeric columns with mean
num_cols <- sapply(house_prices[, -which(names(house_prices) == "SalePrice")], is.numeric)
house_prices[, -which(names(house_prices) == "SalePrice")][, num_cols] <- 
  lapply(house_prices[, -which(names(house_prices) == "SalePrice")][, num_cols], function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))

# Impute missing values for non-numeric columns with mode
nonnum_cols <- sapply(house_prices[, -which(names(house_prices) == "SalePrice")], function(x) !is.numeric(x))
house_prices[, -which(names(house_prices) == "SalePrice")][, nonnum_cols] <- 
  lapply(house_prices[, -which(names(house_prices) == "SalePrice")][, nonnum_cols], function(x) ifelse(is.na(x), as.character(mode(x)), x))
#endregion

#region[Visualization]
train_dataset <- subset(house_prices, !is.na(SalePrice))

#Histogram: SalePrice
ggplot(train_dataset, aes(x = SalePrice)) +
  geom_histogram(fill = "skyblue", color = "black") +
  labs(x = "Sale Price", y = "Frequency", title = "Distribution of Sale Prices")

# Violin Plot: SalePrice by Neighborhood
train_dataset$Neighborhood <- factor(train_dataset$Neighborhood, levels = c("MeadowV", "IDOTRR", "BrDale", "OldTown", "Edwards", "BrkSide", "Sawyer", "Blueste", "SWISU", "NAmes", "NPkVill", "Mitchel", "SawyerW", "NWAmes", "Gilbert", "Blmngtn", "CollgCr", "Crawfor", "ClearCr", "Somerst", "Veenker", "Timber", "StoneBr", "NridgHt", "NoRidge"))

ggplot(train_dataset, aes(x = Neighborhood, y = SalePrice)) +
  geom_violin(fill = "skyblue", color = "black") +
  labs(x = "Neighborhood", y = "Sale Price", title = "Sale Price by Neighborhood") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

#Boxplot: SalePrice by OverallQual
ggplot(train_dataset, aes(x = as.factor(OverallQual), y = SalePrice)) +
  geom_boxplot(fill = "skyblue", color = "black") +
  labs(x = "Overall Quality", y = "Sale Price", title = "Sale Price by Overall Quality")

#Boxplot: SalePrice by ExterQual
train_dataset$ExterQual <- factor(train_dataset$ExterQual, levels = c("Po", "Fa", "TA", "Gd", "Ex"))
ggplot(train_dataset, aes(x = as.factor(ExterQual), y = SalePrice)) +
  geom_boxplot(fill = "skyblue", color = "black") +
  labs(x = "External Quality", y = "Sale Price", title = "Sale Price by External Quality")

#Scatter Plot: SalePrice vs. GrLivArea
ggplot(train_dataset, aes(x = GrLivArea, y = SalePrice)) +
  geom_point(color = "skyblue") +
  labs(x = "GrLivArea", y = "Sale Price", title = "Sale Price vs. GrLivArea")

#Bar Plot: SaleType
ggplot(train_dataset, aes(x = SaleType)) +
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
train_dataset <- subset(house_prices, !is.na(SalePrice))
test_dataset <- subset(house_prices, is.na(SalePrice))

# Extract the Id column from the original house_prices dataset
test_ids <- house_prices$Id[is.na(house_prices$SalePrice)] # Get the Ids of the test dataset for the submission file
#endregion

#region[Correlation Matrix]
c_Mat <- cor(train_dataset[, sapply(train_dataset, is.numeric)], use = "complete.obs")
corrplot(c_Mat, method = "color", type = "upper", tl.cex = 0.7)
#endregion

#region[Models Training]
dependentVariable <- train_dataset$SalePrice # Extract the SalePrice column
set.seed(45) # Init seed for all models

## Linear Regression Model #####################
modelLR <- lm(SalePrice ~ ., data = train_dataset)
predictionsLR <- predict(modelLR, newdata = test_dataset)

## SVR Model ###################################
modelSVR <- svm(
  x = as.matrix(train_dataset[, -which(names(train_dataset) == "SalePrice")]),
  y = dependentVariable,
  kernel = "radial",
  cost = 20,
  gamma = 0.001
)
predictionsSVR <- predict(modelSVR, newdata = as.matrix(test_dataset[, -which(names(test_dataset) == "SalePrice")]))

## XGBOOST Model ###############################
modelXG <- xgboost(
  data = as.matrix(train_dataset[, -which(names(train_dataset) == "SalePrice")]),
  label = dependentVariable,
  nrounds = 1000,
  params = list(objective = "reg:squarederror", eta = 0.01, max_depth = 5, colsample_bytree = 0.5, subsample = 0.5)
)
predictionsXG <- predict(modelXG, newdata = as.matrix(test_dataset[, -which(names(test_dataset) == "SalePrice")]))
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