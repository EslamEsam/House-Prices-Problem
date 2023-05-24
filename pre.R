# Read the 'train.csv' file into the 'rain_original' dataframe, treating strings as non-factor variables
train_original <- read.csv('train.csv', stringsAsFactors = FALSE)

# Read the 'test.csv' file into the 'test_original' dataframe, treating strings as non-factor variables
test_original <- read.csv('test.csv', stringsAsFactors = FALSE)

# Add a new 'SalePrice' column to the 'test_original' dataframe and fill it with NA values
test_original$SalePrice <- rep(NA, nrow(test_original))

# Combine the 'train_original' and 'test_original' dataframes into the 'combined_dataset' dataframe
combined_dataset <- bind_rows(train_original, test_original)

# Drop variables from 'combined_dataset' with a high percentage of missing values
na_cols <- colSums(is.na(combined_dataset)) / nrow(combined_dataset)
cols_to_remove <- names(na_cols[na_cols > 0.5])
house_prices <- combined_dataset[, !names(combined_dataset) %in% cols_to_remove]

# Impute missing values for numeric columns in 'house_prices' with the mean
num_cols <- sapply(house_prices[, -which(names(house_prices) == "SalePrice")], is.numeric)
house_prices[, -which(names(house_prices) == "SalePrice")][, num_cols] <- 
  lapply(house_prices[, -which(names(house_prices) == "SalePrice")][, num_cols], function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))

# Impute missing values for non-numeric columns in 'house_prices' with the mode
nonnum_cols <- sapply(house_prices[, -which(names(house_prices) == "SalePrice")], function(x) !is.numeric(x))
house_prices[, -which(names(house_prices) == "SalePrice")][, nonnum_cols] <- 
  lapply(house_prices[, -which(names(house_prices) == "SalePrice")][, nonnum_cols], function(x) ifelse(is.na(x), as.character(mode(x)), x))
