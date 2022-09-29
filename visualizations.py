import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def hist_all_features(data):
    data_part = data.sample(frac = 0.6, random_state = 1)
    data_part.hist(figsize = (20, 20))
    plt.show()

def corr_scatterplot(data, x, y):
    sns.scatterplot(data=data, x=x, y=y)
    plt.show()

def value_counts(data, feature):
    plt.figure(figsize=(16,6))
    sns.countplot(data=data, x=str(feature), palette='Pastel1')
    plt.show()

def corr_matrix(data):
    corrmat = data.corr()
    fig = plt.figure(figsize = (12, 9))
    sns.heatmap(corrmat, vmax = .8, square = True)
    plt.show()

def boxplot_all_features(data):
    #I am making a copy to Scale this, just for the clarity of the plot
    data_c = data.copy()

    #Creating a df of just Numerical Values
    num_vars = data_c.columns[data.dtypes != object]
    data_numerical = data_c[num_vars]

    #Scaling its Features
    scaler = StandardScaler()
    scaled_num_data = scaler.fit_transform(data_numerical.to_numpy())
    scaled_num_data = pd.DataFrame(scaled_num_data, columns=num_vars)

    #Plotting it
    fig, ax = pyplot.subplots(figsize=(20,10))
    sns.boxplot(ax= ax, data=scaled_num_data,).set(title="Box Plots of all Numerical Features")
    plt.show()