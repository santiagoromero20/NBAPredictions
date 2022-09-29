import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

import warnings
warnings.filterwarnings("ignore")

"""Here you could find all the functions called on the Notebook. The main motivation for creation this new .py file was to store
the pre processing functions, and also to avoid repetition of them and gain a clearer Notebook."""


#OUTLIERS CORRECTION

def outliers(dataframe, feat):

  percentiles = np.percentile(dataframe[feat], [25,50,75])
  Q1, _, Q3  = percentiles[0], percentiles[1], percentiles[2]
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5*IQR
  upper_bound = Q3 + 1.5*IQR

  ls = dataframe.index[(dataframe[feat] < lower_bound) | (dataframe[feat] > upper_bound)]

  return ls

def fill_with_empty(dataframe, feat, indexs):
  for j in range(len(indexs)): 
    dataframe.loc[indexs[j], feat] = np.median(dataframe[feat])

  return

def detecting_filling(dataframe, numerical_features):
  for i in range(len(numerical_features)):
    outs = outliers(dataframe, str(numerical_features[i]))
    fill_with_empty(dataframe, str(numerical_features[i]), outs)

  return dataframe

#FILLING MISSING DATA

def imputing_missingdata(dataframe, all_features, numerical_features, categorical_features):

  #Numerical Features
  simp_imp = SimpleImputer(strategy="median")
  for i in range(len(all_features)):
      for j in range(len(numerical_features)):
        if all_features[i] == numerical_features[j]:
          simp_imp.fit(dataframe[str(numerical_features[j])].values.reshape(-1, 1))
          dataframe[str(numerical_features[j])] = simp_imp.transform(dataframe[str(numerical_features[j])].values.reshape(-1, 1))

  #Categorical Features 
  simp_imp_0 = SimpleImputer(strategy="most_frequent")
  for i in range(len(all_features)):
      for j in range(len(categorical_features)):
        if all_features[i] == categorical_features[j]:
          simp_imp_0.fit(dataframe[str(categorical_features[j])].values.reshape(-1, 1))
          dataframe[str(categorical_features[j])] = simp_imp_0.transform(dataframe[str(categorical_features[j])].values.reshape(-1, 1))

  return dataframe


#ENCODING

def encoding(dataframe, categorical_features):

    encoder = OneHotEncoder().fit(dataframe[categorical_features])
    dataframe = pd.concat(
        [
            dataframe,
            pd.DataFrame(
                encoder.transform(dataframe[categorical_features]).toarray(),
                index=dataframe.index,
                columns=encoder.get_feature_names(categorical_features)
            )
        ],
        axis=1
    )
    dataframe.drop(categorical_features, axis=1, inplace=True)

    return dataframe

#SCALING

def scaling(dataframe, dataframe1):
  scaler = StandardScaler()
  
  new_features = dataframe.columns
  dataframe[new_features] = scaler.fit_transform(dataframe[new_features])

  new_features_1 = dataframe1.columns
  dataframe1[new_features_1] = scaler.transform(dataframe1[new_features_1])
  
  return dataframe, dataframe1



"""Data Preprocessinf Function. This should be called when all the previous analysis (TSNE, Imbalanced Correction, Anomalys Detection) have been done. 
This return you the Training and Test Dataset preprocessed ready to be use for evaluate the Performance of some Models """

def data_preprocessing(train, test, numerical_features, categorical_features, all_features):
  #Correct Outliers
  numerical_features.remove("SALARY")
  train = detecting_filling(train, numerical_features)
  test  = detecting_filling(test, numerical_features)

  #Impute values for all columns missing data
  train = imputing_missingdata(train, all_features, numerical_features, categorical_features)
  test  = imputing_missingdata(test, all_features, numerical_features, categorical_features)

  #Encode categorical features
  train = encoding(train, categorical_features)
  test  = encoding(test, categorical_features)

  #Alignating Columns
  train, test = train.align(test, join = 'inner', axis = 1)
  
  #Feature Scaling
  if len(train.columns) == len(test.columns):
    train, test = scaling(train, test)
  else:
    A = set(train.columns)
    B = set(test.columns)
    if len(A) > len(B):
      C = A.intersection(B)
      D = A - C 
      print("You should add this columns to the test Dataframe:",D)
    elif len(B) > len(A):
      C = A.intersection(B)
      D = B - C
      print("You should add this columns Train Dataframe:",D)

  return train, test