'''Wrangles data from Zillow Database'''

##################################################Wrangle.py###################################################
#these are all the library imports and env file needed to run the functions in this file
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from env import user, password, host

import os

#**************************************************Acquire*******************************************************

def acquire_zillow():
    ''' Acquire bed, bath, building and lot sq ft, assessed value and fips location from codeup database using credentials from env file'''
    
    url = f"mysql+pymysql://{user}:{password}@{host}/zillow"
    
    query = """
            
    SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, lotsizesquarefeet, taxvaluedollarcnt, yearbuilt, fips 
    FROM properties_2017

    LEFT JOIN propertylandusetype USING(propertylandusetypeid)
    LEFT JOIN predictions_2017 USING(id)

    WHERE propertylandusedesc IN ("Single Family Residential",                       
                                  "Inferred Single Family Residential")
                                
    AND transactiondate BETWEEN '2017-01-01' AND '2017-12-31' """

    # get dataframe of data and saving it as variable named df
    df = pd.read_sql(query, url)
    
    
    # renaming column names to easier to understand names
    df = df.rename(columns = {'bedroomcnt':'bedrooms', 
                              'bathroomcnt':'bathrooms', 
                              'calculatedfinishedsquarefeet':'area',
                              'taxvaluedollarcnt':'tax_value', 
                              'yearbuilt':'year_built',
                              'lotsizesquarefeet':'lot_area'})
    return df

#**************************************************Remove Outliers*******************************************************

def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

#**************************************************Distributions*******************************************************

def get_hist(df):
    ''' Gets histographs of acquired continuous variables from the dataset'''
    
    plt.figure(figsize=(16, 3))

    # List of columns
    cols = [col for col in df.columns if col not in ['fips', 'year_built',]]

    for i, col in enumerate(cols):

        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display histogram for column.
        df[col].hist(bins=5)

        # Hide gridlines.
        plt.grid(False)

        # turn off scientific notation
        plt.ticklabel_format(useOffset=False)

        plt.tight_layout()

    plt.show()
        
        
def get_box(df):
    ''' Gets boxplots of acquired continuous variables'''
    
    # List of columns
    cols = ['bedrooms', 'bathrooms', 'area', 'tax_value',]

    plt.figure(figsize=(16, 3))

    for i, col in enumerate(cols):

        # i starts at 0, but plot should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display boxplot for column.
        sns.boxplot(data=df[[col]])

        # Hide gridlines.
        plt.grid(False)

        # sets proper spacing between plots
        plt.tight_layout()

    plt.show()
        
#**************************************************Prepare*******************************************************

def prepare_zillow(df):
    ''' Prepare zillow data for exploration by taking in a dataframe and returns train, validate, test'''

    # removing outliers
    df = remove_outliers(df, 1.5, ['bedrooms', 'bathrooms', 'area', 'tax_value',])
    
    # get distributions of numeric data
    get_hist(df)
    get_box(df)
    
    # drop null values that are left in the lot size data since there are so few
    df=df.dropna()
    
    # converting column datatypes 
    df.fips = df.fips.astype(object)
    df.year_built = df.year_built.astype(object)
    
    # train/validate/test split and is reproducible due to random_state = 123
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    # impute year built using median from train data set and then applied to the validate and test set as well
    imputer = SimpleImputer(strategy='median')

    imputer.fit(train[['year_built']])

    train[['year_built']] = imputer.transform(train[['year_built']])
    validate[['year_built']] = imputer.transform(validate[['year_built']])
    test[['year_built']] = imputer.transform(test[['year_built']])       
    
    return train, validate, test    


#**************************************************Wrangle*******************************************************


def wrangle_zillow():
    '''Acquire and prepare data from Zillow database to explore. will search local directory for CSV files first, if not present
    will pull from SQL server and automatically save it to train,validate, and test CSV files in same directory'''
    if os.path.isfile('train.csv'):
        train=pd.read_csv('train.csv')
        get_box(train)
        get_hist(train)
        return train, pd.read_csv('validate.csv'), pd.read_csv('test.csv')
    else:
        train, validate, test = prepare_zillow(acquire_zillow())
        train.to_csv('train.csv',index=False), validate.to_csv('validate.csv', index=False), test.to_csv('test.csv', index=False)
    
    return train, validate, test