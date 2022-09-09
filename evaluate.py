import seaborn as sns
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr

import sklearn.preprocessing
scaler = sklearn.preprocessing.MinMaxScaler()

# modeling methods
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures


def create_heat_map(train):
    corr_matrix=train.corr()
    
    kwargs = {'alpha':.9,'linewidth':3, 'linestyle':'-', 
            'linecolor':'k','rasterized':False, 'edgecolor':'w', 
            'capstyle':'projecting',}

    sns.heatmap(corr_matrix, cmap='Reds', annot=True,mask=np.triu(corr_matrix), **kwargs)
    plt.show()


def create_violin_chart_bedrooms(train):
     sns.violinplot(x='bedrooms', y='tax_value', data=train)
     plt.ticklabel_format(style='plain', axis='y')
     plt.title('Assessed Values trend upward with # of rooms')
     plt.show()
     r, p_value = pearsonr(train.bedrooms, train.tax_value)
     print(f'Correlation Coefficient: {r}\nP-value: {p_value}')
    
def create_violin_chart_bathrooms(train):
     sns.violinplot(x='bathrooms', y='tax_value', data=train)
     plt.ticklabel_format(style='plain', axis='y')
     plt.title('Assessed Values trend upward with # of rooms')
     plt.show()
     r, p_value = pearsonr(train.bathrooms, train.tax_value)
     print(f'Correlation Coefficient: {r}\nP-value: {p_value}')
    
def create_boxen_plot_area(train):
    sns.boxenplot(data=train, x=pd.cut(train.area,bins=6), y="tax_value")
    plt.ticklabel_format(style='plain', axis='y')
    plt.xticks(rotation = 90)
    plt.ylabel('Assessed Value')
    plt.xlabel('house sqaure footage')
    plt.title('Assessed Value generally increases with square footage' )
    plt.show()
    r, p_value = pearsonr(train.area, train.tax_value)
    print(f'Correlation Coefficient: {r}\nP-value: {p_value}')

def feature_engineer(train,validate,test):

    train['lot_living_ratio']= train.area/ train.lot_area
    validate['lot_living_ratio']= validate.area/ validate.lot_area
    test['lot_living_ratio']= test.area/ test.lot_area

    train['non_bed_bath_area']=train.area-(train['bedrooms'] * 144)+ (train['bathrooms']*150)
    validate['non_bed_bath_area']=validate.area-(validate['bedrooms'] * 144)+ (validate['bathrooms']*150)
    test['non_bed_bath_area']=test.area-(test['bedrooms'] * 144)+ (test['bathrooms']*150)
    return train, validate, test


def regplot_engineered_feat(train):
    sns.lmplot(x='non_bed_bath_area', y='tax_value', data=train, scatter_kws={"s": .5},line_kws={"color": "C1"})
    plt.ticklabel_format(style='plain', axis='y')
    plt.title('Non bed or bath sq ft trends upward with assessed value')
    plt.ylabel('Assessed Value')
    plt.xlabel('Non bed or bath square footage')
    plt.show()

    r, p_value = pearsonr(train.non_bed_bath_area, train.tax_value)
    print(f'Correlation Coefficient: {r}\nP-value: {p_value}')


def X_and_y_split(train,validate,test):
    X_train = train.drop(columns=['tax_value'])
    y_train = train.tax_value

    X_validate = validate.drop(columns=['tax_value'])
    y_validate = validate.tax_value

    X_test = test.drop(columns=['tax_value'])
    y_test = test.tax_value
    return X_train, y_train, X_validate, y_validate, X_test, y_test



def target_var_dist(y_train):

    plt.hist(y_train)
    plt.xlabel('assessed value')
    plt.ylabel("Number of properties")
    plt.show()

def scale_data(X_train,X_validate,X_test):
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    X_test_scaled = scaler.transform(X_test)

    plt.figure(figsize=(13, 6))
    plt.subplot(121)
    plt.hist(X_train, bins=25, ec='black')
    plt.title('Original')
    plt.subplot(122)
    plt.hist(X_train_scaled, bins=25, ec='black')
    plt.title('Scaled')
    plt.show()
    return X_train_scaled, X_validate_scaled, X_test


def calc_baseline(y_train,y_validate):
    # We need y_train and y_validate to be dataframes to append the new columns with predicted values. 
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)

    # 1. Predict value_pred_mean
    value_pred_mean = y_train['tax_value'].mean()
    y_train['value_pred_mean'] = value_pred_mean
    y_validate['value_pred_mean'] = value_pred_mean

    # 2. compute value_pred_median
    value_pred_median = y_train['tax_value'].median()
    y_train['value_pred_median'] = value_pred_median
    y_validate['value_pred_median'] = value_pred_median

    # 3. RMSE of value_pred_mean
    rmse_train = mean_squared_error(y_train.tax_value, y_train.value_pred_mean)**(1/2)
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.value_pred_mean)**(1/2)

    print("RMSE using Mean\nTrain/In-Sample: ", round(rmse_train, 2), 
        "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))

    # 4. RMSE of value_pred_median
    rmse_train = mean_squared_error(y_train.tax_value, y_train.value_pred_median)**(1/2)
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.value_pred_median)**(1/2)

    print("RMSE using Median\nTrain/In-Sample: ", round(rmse_train, 2), 
        "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))