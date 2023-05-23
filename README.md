
# Housing Price Problem


### Techniques used in Preprocessing and Feature Selection

1. The train and test datasets are read from CSV files and stored in variables train and test respectively then the datasets are then combined into a single dataset called _house\_prices_ to apply preprocessing on both datasets and separating them before the training process
2. We dropped columns (_Alley, PoolQC, Fence, MiscFeature_) with too many NA values after looking at the summary.
3. We filled the missing values in the numeric columns using the mean.
4. We filled the missing values in the non-numeric columns using the most frequently occurring value in that column (mode).
5. We applied factorization on the non-numeric columns to make all the data numeric


### Data Visualization

1. We visualized the **correlation matrix** between each attribute and the _SalePrice_ column.

![Rplot001](https://github.com/EslamEsam/House-Prices-Problem/assets/62443536/0e645455-68ba-4540-a5d7-0636f3c2b470)

2. We used a **Histogram** to show the distribution of the _SalePrice_ column in the data.

![Rplot002](https://github.com/EslamEsam/House-Prices-Problem/assets/62443536/d4532b74-984b-42b5-b255-5fbc3b047916)

3. We used a **Scatter Plot** to show the relation between _SalePrice_ and _GrLivArea_.

![Rplot003](https://github.com/EslamEsam/House-Prices-Problem/assets/62443536/fbc7a7b3-9a6b-430f-bd18-eaec0a758fe7)

4. We used a **BoxPlot** to show the distribution of _SalePrice_ for each level of _OverallQual_.

![Rplot004](https://github.com/EslamEsam/House-Prices-Problem/assets/62443536/a105e6df-3162-4135-803f-27ca02a3049d)


5. We used a **Histogram** to show the distribution and density of the predicted sale prices_._

![Rplot006](https://github.com/EslamEsam/House-Prices-Problem/assets/62443536/5007cd77-a9a8-43a0-ac9b-3205c7f03611)
![Rplot005](https://github.com/EslamEsam/House-Prices-Problem/assets/62443536/452148cd-51b4-4dc9-a47c-b7b104496e28)


##
### Experiments and Results

1. We tried visualizing the correlation between each attribute and the _SalePrice_ column and dropped the columns with low correlation _but_ it increased the MSE.
2. We tried different models so we can see which one will give us the lowest error and this is the results

| **Model Name** | **Error** |
| --- | --- |
| XGBoost | 0.123 |
| SVR | 0.125 |
| Random Forest | 0.14 |
