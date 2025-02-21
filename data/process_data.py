import sys
import pandas as pd
import string as s
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads messages and categories datasets from CSV files.

    Args:
        messages_filepath (str): File path for messages.csv file.
        categories_filepath (str): File path for categories.csv file.

    Returns:
        x2 pd.DataFrame: Messages and categories dataframes.
    """
    messages = pd.DataFrame(pd.read_csv(messages_filepath))
    categories = pd.DataFrame(pd.read_csv(categories_filepath))
    return messages, categories

def clean_data(df, categories):
    """
    Cleans and processes the data.

    Args:
        df (pd.DataFrame): Messages dataframe.
        categories (pd.DataFrame): Categories dataframe.

    Returns:
        pd.DataFrame: Cleaned and merged dataframe.
    """
    # merge datasets
    df = df.merge(categories, how='outer', on=['id'])

    # create a dataframe of the 36 individual category columns
    categories = categories.categories.str.split(pat=';', expand=True)

    # select the first row of the categories dataframe
    row = tuple(categories.loc[0])

    # use this row to extract a list of new column names for categories.
    category_colnames = list(map(lambda i: i[:-2], row))

    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)

    # Leaving nulls in Original field as it is not needed for our model. Removing nulls created from joining the categories on as we cannot use them to train our model if they have no data.
    df.dropna(subset=['related'], inplace=True)
    
    return df

def save_data(df, database_filename):
    """
    Save the cleaned dataset as a SQLite database.

    Args:
        df (pd.DataFrame): Cleaned dataframe.
        database_filename (str): File path for the database file.
    """
    engine = create_engine(f'sqlite:///{database_filename}', echo=False)
    df.to_sql(name='data', con=engine, index=False, if_exists='replace')

def main():
    """
    Execute the ETL pipeline:
    - Load data
    - Clean data
    - Save data to a SQLite database
    """
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        messages, categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(messages, categories)

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
