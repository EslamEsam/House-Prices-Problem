library(randomForest)
library(dplyr)
library(caret)
library(tidyverse)

setwd("C:/Users/eslam/Downloads/Compressed/8th term/Distributed Computing/project")

# Load necessary libraries and data
train <- read.csv('train.csv', stringsAsFactors = F)
test <- read.csv('test.csv', stringsAsFactors = F)
test$SalePrice <- rep(NA, nrow(test))

# Combine train and test data
house_prices <- bind_rows(train, test)

# Drop variables with too many missing values
house_prices <- house_prices %>% select(-c(Alley, PoolQC, Fence, MiscFeature)) 

# Impute missing values for numeric columns
num_cols <- sapply(house_prices[, -which(names(house_prices) == "SalePrice")], is.numeric)
house_prices[, -which(names(house_prices) == "SalePrice")][, num_cols] <- 
  lapply(house_prices[, -which(names(house_prices) == "SalePrice")][, num_cols], function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))

# Impute missing values for non-numeric columns
nonnum_cols <- sapply(house_prices[, -which(names(house_prices) == "SalePrice")], function(x) !is.numeric(x))
house_prices[, -which(names(house_prices) == "SalePrice")][, nonnum_cols] <- 
  lapply(house_prices[, -which(names(house_prices) == "SalePrice")][, nonnum_cols], function(x) ifelse(is.na(x), as.character(mode(x)), x))

# Encode categorical variables
categorical_cols <- c("MSZoning", "Street", "LotShape", "LandContour", "Utilities", "LotConfig", 
                      "LandSlope", "Neighborhood", "Condition1","BldgType", "HouseStyle", 
                      "RoofStyle",  "Exterior1st", "MasVnrType", "ExterQual", 
                      "ExterCond", "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", 
                      "BsmtFinType2", "Heating", "HeatingQC", "CentralAir", "KitchenQual", 
                      "Functional", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", 
                      "PavedDrive", "SaleType", "SaleCondition")
for (col in categorical_cols) {
  house_prices[[col]] <- as.factor(house_prices[[col]])
}

# Split the data into train and test sets
train <- subset(house_prices, !is.na(SalePrice))
test <- subset(house_prices, is.na(SalePrice))

# Train a random forest model
set.seed(69)
model <- randomForest(SalePrice ~ ., data = train, ntree = 10000, mtry = 20)

# Print summary of the model
print(model)

# Make predictions on the test set
predictions <- predict(model, newdata = test)

# Save predictions to CSV file
submission <- data.frame(Id = test$Id, SalePrice = predictions)
write.csv(submission, "prediction.csv", row.names = FALSE)
