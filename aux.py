import pandas as pd
import numpy as np
import statistics

import warnings
warnings.filterwarnings("ignore")

#---------------------------------------------DESCRIPTIVE STATISTICS---------------------------------------------#

"""Auxiliaries"""

def cat_vs_num(data):
    numerical_data = []
    categorical_data = []
    features = data.columns

    for i in range(len(features)):
        if type(data.loc[0, str(features[i])]) == str:
            categorical_data.append(features[i])
        else:
            numerical_data.append(features[i])
    
    return categorical_data, numerical_data



"""Measures of Central Tendency functions"""

#Mean
def mean(data):
    _, num = cat_vs_num(data)
    for i in range(len(num)):
        print("The mean of",str(num[i]),"is",round(np.mean(data[str(num[i])]), 2))

#Weighted Mean
def feature_vals_and_weights(data, feature):

    a = data[str(feature)].value_counts()
    b = a.to_dict()
    feature_vals = []
    weights      = []

    for key, value in b.items():
        feature_vals.append(key)
        weight = value/len(data[str(feature)])
        weights.append(weight)
        print("The rounded percentage of the value", key,"is of",round(weight, 4)*100,"%")

    return feature_vals, weights

def weighted_mean(feature_vals, weights):
    ans = np.average(np.array(feature_vals), weights=np.array(weights))
    return ans

#Median
def median(data):
    _, num = cat_vs_num(data)
    for i in range(len(num)):
        print("The meadian of",str(num[i]),"is",round(np.median(data[str(num[i])]), 2))
    
#Mode
def mode(data):
    _, num = cat_vs_num(data)
    for i in range(len(num)):
        print("The mode of",str(num[i]),"is",statistics.mode(data[str(num[i])]))


"""Measures of Variability"""

#Variance
def variance(data):
    _, num = cat_vs_num(data)
    for i in range(len(num)):
        print("The variance of",str(num[i]),"is",np.var(data[str(num[i])], ddof=1))

#Standard Deviation
def std_deviation(data):
    _, num = cat_vs_num(data)
    for i in range(len(num)):
        print("The Standard Deviation of",str(num[i]),"is",np.std(data[str(num[i])], ddof=1))

#Skewness
def skewness(data):
    _, num = cat_vs_num(data)
    for i in range(len(num)):
        print("The Skewness of",str(num[i]),"is", data[str(num[i])].skew())

#Percentile
def percentiles(data):
    _, num = cat_vs_num(data)
    for i in range(len(num)):
        print("The percentiles of",str(num[i]),"is", np.percentile(data[str(num[i])], [25,50,75]))

"""Measures of Correlation Between Pairs of Data"""

def covariance_matrix(feature1, feature2):
    print("The Coviarance Matrix between this two features is: \n", np.cov(feature1, feature2))

def correlation_matrix(feature1, feature2):
    print("The Correlation Matrix between this two features is: \n", np.corrcoef(feature1, feature2))

def fts_correlation_y(data, y):
    _, num = cat_vs_num(data)
    for i in range(len(num)):
        try:
            m = np.corrcoef(data[str(num[i])], y)
            c = m[0][1]
        except:
            print("Correlation between",str(num[i]),"and the label is uncalculable")

        print("Correlation between",str(num[i]),"and the label is", c)


#---------------------------------------------MODEL EVALUATION---------------------------------------------#

"""Auxiliaries"""

def calculating_error(dict, key_error):
    error_array = dict[str(key_error)]*(-1)
    error = np.mean(error_array)
    return error

def get_errors(grid_results):
    results = pd.DataFrame.from_dict(grid_results)
    results = results.sort_values(by=["mean_test_score"], ascending=False)
    results = results.head().reset_index()
    error_cv = -results.loc[0, "mean_test_score"]
    error_tr = -results.loc[0, "mean_train_score"]
    print(f"Training RMSE Error {error_tr:0.2f}, CV RMSE Error {error_cv:0.2f}")
    print('Train/Validation: {}'.format(round(error_cv/error_tr, 1)))