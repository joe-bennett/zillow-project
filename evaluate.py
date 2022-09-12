# imports used to make the following functions work

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

from sklearn.metrics import explained_variance_score


def create_heat_map(train):
    """takes in the train dataframe and creates a correlation coefficent matrix. then takes that matrix and creates a heatmap"""
    corr_matrix=train.corr()
    
    kwargs = {'alpha':.9,'linewidth':3, 'linestyle':'-', 
            'linecolor':'k','rasterized':False, 'edgecolor':'w', 
            'capstyle':'projecting',}

    sns.heatmap(corr_matrix, cmap='Reds', annot=True,mask=np.triu(corr_matrix), **kwargs)
    plt.show()


def create_violin_chart_bedrooms(train):
    """creates violin plot of bedrooms from the train dataframe"""
    sns.violinplot(x='bedrooms', y='tax_value', data=train)
    plt.ticklabel_format(style='plain', axis='y')
    plt.title('Assessed Values trend upward with # of rooms')
    plt.show()
    r, p_value = pearsonr(train.bedrooms, train.tax_value)
    print(f'Correlation Coefficient: {r}\nP-value: {p_value}')
    
def create_violin_chart_bathrooms(train):
    """creates violin plot of bathrooms from the train dataframe"""

    sns.violinplot(x='bathrooms', y='tax_value', data=train)
    plt.ticklabel_format(style='plain', axis='y')
    plt.title('Assessed Values trend upward with # of rooms')
    plt.show()
    r, p_value = pearsonr(train.bathrooms, train.tax_value)
    print(f'Correlation Coefficient: {r}\nP-value: {p_value}')
    
def create_boxen_plot_area(train):
    """takes in the train DataFrame and creates a boxenplot"""
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
    """takes in train, validate, and test data and creates the two features area/lot ratio and non bed/bath area"""

    train['lot_living_ratio']= train.area/ train.lot_area
    validate['lot_living_ratio']= validate.area/ validate.lot_area
    test['lot_living_ratio']= test.area/ test.lot_area

    train['non_bed_bath_area']=train.area-(train['bedrooms'] * 144)+ (train['bathrooms']*150)
    validate['non_bed_bath_area']=validate.area-(validate['bedrooms'] * 144)+ (validate['bathrooms']*150)
    test['non_bed_bath_area']=test.area-(test['bedrooms'] * 144)+ (test['bathrooms']*150)
    return train, validate, test


def regplot_engineered_feat(train):
    """takes the train DataFrame and creates a regplot for the non bed/bath column"""
    sns.lmplot(x='non_bed_bath_area', y='tax_value', data=train, scatter_kws={"s": .5},line_kws={"color": "C1"})
    plt.ticklabel_format(style='plain', axis='y')
    plt.title('Non bed or bath sq ft trends upward with assessed value')
    plt.ylabel('Assessed Value')
    plt.xlabel('Non bed or bath square footage')
    plt.show()

    r, p_value = pearsonr(train.non_bed_bath_area, train.tax_value)
    print(f'Correlation Coefficient: {r}\nP-value: {p_value}')


def X_and_y_split(train,validate,test):
    """takes in train, validate, test data and splits it into X and y data sets 
    returning X_train, y_train, X_validate, y_validate, X_test, y_test"""
    X_train = train.drop(columns=['tax_value'])
    y_train = train.tax_value

    X_validate = validate.drop(columns=['tax_value'])
    y_validate = validate.tax_value

    X_test = test.drop(columns=['tax_value'])
    y_test = test.tax_value
    return X_train, y_train, X_validate, y_validate, X_test, y_test



def target_var_dist(y_train):
    """takes the y_train data and creates histogram to display target variable distribution"""
    plt.hist(y_train)
    plt.xlabel('assessed value')
    plt.ylabel("Number of properties")
    plt.show()

def scale_data(X_train,X_validate,X_test):
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    X_test_scaled = scaler.transform(X_test)
    return X_train,X_validate,X_test





def calc_baseline(y_train,y_validate):
    """takes in y_train and y_validate data and converts into DataFrames. The mean and median are both calculated and printed below along with 
    the RMSE for each"""
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


def model_eval_compare(y_train,y_validate):
    """takes in the y_train and y_validate DataFrames and returns a scatterplot and a histogram that overlays all models and compares them. 
    below that the R^2 values for each is printed"""
    plt.figure(figsize=(16,8))
    plt.axhline(label="No Error")
    plt.scatter(y_validate.tax_value, y_validate.value_pred_lm-y_validate.tax_value, 
                alpha=.7, color="red", s=10, label="Model: LinearRegression")
    plt.scatter(y_validate.tax_value, y_validate.value_pred_lars-y_validate.tax_value, 
                alpha=.7, color="yellow", s=10, label="Model: LassoLars")
    plt.scatter(y_validate.tax_value, y_validate.value_pred_lm2-y_validate.tax_value, 
                alpha=.2, color="green", s=10, label="Model 2nd degree Polynomial")
    plt.legend()
    plt.xlabel("Actual Assessed Value")
    plt.ylabel("Residual/Error: Predicted value - Actual Value")
    plt.title("Errors in predictions")
    plt.annotate("The polynomial model appears less consistent", (350000, 450000))
    plt.annotate("The OLS model (LinearRegression)\n appears to be more consistent no outliers", (20000, -400000))
    plt.show()

        # plot to visualize actual vs predicted. 
    plt.figure(figsize=(16,8))
    plt.hist(y_validate.tax_value, color='blue', alpha=.5, label="Actual Assessed Values")
    plt.hist(y_validate.value_pred_lm, color='red', alpha=.5, label="Model: LinearRegression")
    plt.hist(y_validate.value_pred_lars, color='yellow', alpha=.5, label="Model: LassoLars")
    plt.hist(y_validate.value_pred_lm2, color='green', alpha=.5, label="Model 2nd degree Polynomial")
    plt.xlabel("Assessed Value")
    plt.ylabel("Number of Properties")
    plt.title("Comparing the Distribution of Actual Assessed values to Distributions of Predicted Values for the Top Models")
    plt.legend()
    plt.show()

    evs_lm = explained_variance_score(y_train.tax_value, y_train.value_pred_lm)
    print('Explained Variance linear regression = ', round(evs_lm,3))

    evs_lars = explained_variance_score(y_train.tax_value, y_train.value_pred_lars)
    print('Explained Variance LassoLars = ', round(evs_lars,3))

    evs_poly = explained_variance_score(y_train.tax_value, y_train.value_pred_lm2)
    print('Explained Variance Polynomial regression = ', round(evs_poly,3))

def show_best_model_on_test(y_test):
    """plug in the y_test dataframe and returns a chart showing prediction error, RMSE, and R^2 using linear regression model"""
    plt.figure(figsize=(16,8))
    plt.axhline(label="No Error")
    plt.scatter(y_test.tax_value, y_test.value_pred_lm-y_test.tax_value, 
                alpha=.7, color="red", s=10, label="Model: LinearRegression")
    plt.legend()
    plt.xlabel("Actual Assessed Value")
    plt.ylabel("Residual/Error: Predicted value - Actual Value")
    plt.title("Top model performed same on final out of sample data")
    plt.annotate("The OLS model (LinearRegression)\n seems more concentrated in lower assessed values", (10000, -200000))
    plt.show()

