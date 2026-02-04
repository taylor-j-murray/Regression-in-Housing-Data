`final.ipbynb`  features an end to end ML project. Consists of a Jupyter Notebook presentation on EDA and Machine learning modeling/performace 
on predicticing housing prices using the Kaggle data set: https://www.kaggle.com/datasets/austinreese/usa-housing-listings

We quickly note some key takeaways from `final.ipbynb`:

**Objective:** predict prices from structured housing data through engineered features and the XGBoost Regression model.

**Final Performance:** 
|   Data    | Relative RMSE Score | $R^2$ Score|
|:--------:|:---------------:|:--------:|
|Training| 11.03% | 0.8769|
|Cross-Validated | 12.94%|0.8273|
|Testing| 12.96% |0.8241|

Our final models $R^2$ score indicates that the model accounts for 82.41% of the total variation in housing price.

`preprocessing.py` consists of a number of custom made preprocessing functions.

`taylors_pipes.py` consists of custom made pipeline estimators compatible with scikit-learn.

`utilities.py` consists of a custom made function for splitting data into testing and training data.
