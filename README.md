`final.ipbynb`  features an end to end ML project. Consists of a Jupyter Notebook presentation on EDA and Machine learning modeling/performace 
on predicticing housing prices using the Kaggle data set: https://www.kaggle.com/datasets/austinreese/usa-housing-listings

We quickly note some key takeaways from `final.ipbynb`:

**Objective:**  predict prices from structured housing data.

**Final Performance:** 
|   Data     | $R^2$ Score|
|:--------:|:--------:|
|Training|  0.87580|
|Testing|  |0.82307|

Our final models $R^2$ score indicates that the model accounts for approximately 82.31% of the total variation in housing price for this data set.

`preprocessing.py` consists of a number of custom made preprocessing functions.

`taylors_pipes.py` consists of custom made pipeline estimators compatible with scikit-learn.

`utilities.py` consists of a custom made function for splitting data into testing and training data.
