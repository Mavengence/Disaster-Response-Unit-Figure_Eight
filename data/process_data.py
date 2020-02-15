import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Arguments:
        - string : filepath for the message csv
        - string : filepath for the categories csv
        
    Output:
        - df : pandas merged DataFrame for the two csv files
    """
    # load the two datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, left_on='id', right_on='id')
    
    return df
    

def clean_data(df):
    """
    Arguments:
        - df : loaded pandas DataFrame
        
    Output:
        - df : cleaned and consistent DataFrame
    """
      # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2]).values
    
    # rename the columns of `categories`
    categories.columns = category_colnames  

    
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
    # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # drop the original categories column from `df`
    df.drop(columns=['categories', 'id', 'original'], axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df = df.drop_duplicates()
    
    # clear out inconstent values
    df.loc[df.related == 2] = 1
    
    return df


def save_data(df, database_filename):
    """
    Arguments:
        - df : cleaned and consistent pandas DataFrame
        - string : filename for the database
        
    Output:
        no output, only a db database file will be created for the input filename and DataFrame
    """
     # save to database
    try:
        engine = create_engine('sqlite:///etl-pipeline.db')
        df.to_sql('data/etl-pipeline', engine, index=False)
    except:
        print('Already saved')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
