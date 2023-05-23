#region[libraries]
library(randomForest)
library(dplyr)
library(caret)
library(tidyverse)
library(corrplot)
library(xgboost)
library(e1071)

#endregion

#region[load_data]
setwd("D:\\Kolleya\\8th Term\\Distributed\\Practical\\House-Prices-Problem")

train <- read.csv('train.csv', stringsAsFactors = F)
test <- read.csv('test.csv', stringsAsFactors = F)
test$SalePrice <- rep(NA, nrow(test))

# Combine train and test data to apply the same data preprocessing steps to both datasets
house_prices <- bind_rows(train, test)
#endregion

#region[drop_&_impute]

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


#region[Feature_conversion]

# Convert character columns to factors
char_cols <- sapply(house_prices, is.character)
house_prices[char_cols] <- lapply(house_prices[char_cols], as.factor)

# Convert remaining non-numeric columns to numeric
non_numeric_cols <- sapply(house_prices, function(x) !is.numeric(x))
house_prices[non_numeric_cols] <- lapply(house_prices[non_numeric_cols], as.numeric)

#endregion

# Split the data into train and test sets
train <- subset(house_prices, !is.na(SalePrice))
test <- subset(house_prices, is.na(SalePrice))

# Extract the Id column from the original house_prices dataset
test_ids <- house_prices$Id[is.na(house_prices$SalePrice)]

##################################################################

# Data visualization

# Calculate correlation matrix
# cor_matrix <- cor(train[, sapply(train, is.numeric)], use = "complete.obs")

# Visualize correlation matrix
# corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.7)


##################################################################         
# # Find highly correlated columns
# highly_correlated <- findCorrelation(cor_matrix, cutoff = 0.2, verbose = FALSE)
# highly_correlated_cols <- colnames(cor_matrix)[highly_correlated]


# print(highly_correlated_cols)


# # Extract the Id column from the original house_prices dataset
# test_ids <- house_prices$Id[is.na(house_prices$SalePrice)]



# # Subset the train and test data to include only the highly correlated columns
# train <- train[,  highly_correlated_cols]
# test <- test[, highly_correlated_cols]
##################################################################

# # Train a random forest model
# set.seed(69)
# model <- randomForest(SalePrice ~ ., data = train, ntree = 20000, mtry = 20)

# # Print summary of the model
# print(model)

# # Make predictions on the test set
# predictions <- predict(model, newdata = test)

##################################################################

# # Train an SVR model
# set.seed(69)

# # Convert character columns to factors
# char_cols <- sapply(train, is.character)
# train[char_cols] <- lapply(train[char_cols], as.factor)

# # Convert remaining non-numeric columns to numeric
# non_numeric_cols <- sapply(train, function(x) !is.numeric(x))
# train[non_numeric_cols] <- lapply(train[non_numeric_cols], as.numeric)

# # Extract the SalePrice column
# label <- train$SalePrice

# # Train the SVR model
# model <- svm(
#   x = as.matrix(train[, -which(names(train) == "SalePrice")]),
#   y = label,
#   kernel = "radial",
#   cost = 20,
#   gamma = 0.001
# )


# # Make predictions on the test set

# # Extract the SalePrice column
# label <- test$SalePrice

# # Predict using the trained model
# predictions <- predict(model, newdata = as.matrix(test[, -which(names(test) == "SalePrice")]))

##################################################################

# Train a model using XGBoost
# set.seed(69)

# # Extract the SalePrice column
# label <- train$SalePrice

# params <- list(
#   objective = "reg:squarederror",
#   eta = 0.01,
#   max_depth = 5,
#   colsample_bytree = 0.5,
#   subsample = 0.5
# )

# model <- xgboost(
#   data = as.matrix(train[, -which(names(train) == "SalePrice")]),
#   label = label,
#   params = params,
#   nrounds = 1000
# )
# # Make predictions on the test set

# # Extract the SalePrice column
# label <- test$SalePrice

# # Predict using the trained model
# predictions <- predict(model, newdata = as.matrix(test[, -which(names(test) == "SalePrice")]))

# ##################################################################

# # Save predictions to CSV file
# submission <- data.frame(Id = test_ids, SalePrice = predictions)
# write.csv(submission, "XGBoost.csv", row.names = FALSE)


# ##################################################################
# # Create a data frame with predicted values
# prediction_data <- data.frame(Predicted = predictions)


# # Plot the histogram of predicted values
# ggplot(prediction_data, aes(x = Predicted)) +
#   geom_histogram(binwidth = 1000, fill = "blue", color = "black") +
#   labs(x = "Predicted SalePrice", y = "Frequency") +
#   ggtitle("Distribution of Predicted SalePrice")

# ggplot(prediction_data, aes(x = Predicted)) +
#   geom_density(fill = "blue", alpha = 0.5) +
#   labs(x = "Predicted SalePrice", y = "Density") +
#   ggtitle("Density Plot of Predicted SalePrice")

