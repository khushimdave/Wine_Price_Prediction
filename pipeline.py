import argparse
import pandas as pd
import numpy as np 
import pickle   # for extracting the model
from sklearn.preprocessing import OrdinalEncoder    # for label encoding

# col_number = 9
col_names = ['winery', 'wine', 'year', 'rating', 'num_reviews', 'region', 'type','body', 'acidity']


def drop_column(df):
    
    """This function will drop the columns that are not present in the col_name list
    If the dataset contains price column, then it will not drop that column"""
    
    if 'price' in df:
        df.drop(columns=[col for col in df.columns if col not in col_names and col != 'price'], inplace=True)
    else:
        df.drop(columns=[col for col in df.columns if col not in col_names], inplace=True) 
    return df

def merge_price_dataset(df, sav_pri):
    
    """If there is price column, then we will drop it first, predict the
    outcome and then merge this removes price back into the dataframe"""
    
    df['Original_Price'] = sav_pri
    return df

def missing_values_n_datatypes(df):
    
    """Here, missing values are handled and the datatypes 
       are changed in order to avoid further error"""
    
    df['type'] = df['type'].fillna(df['type'].mode().iloc[0])
    df['acidity'] = df['acidity'].fillna(df.acidity.median())
    df['year'] = df['year'].fillna('2011')
    df['year'] = df['year'].replace('N.V.', df['year'].interpolate)
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['year'] = df['year'].fillna(method='ffill')
    df['body'].fillna(df.body.median(), inplace=True)
    df['year'] = df['year'].astype(int)
    df['body'] = df['body'].astype(int)
    df['acidity'] = df['acidity'].astype(int)
    return df  # Return the modified DataFrame  
    
def categorization(df):
        
    """This function divides num_reviews column into 4 groups based on the descending order
    i.e. Group 4 -- more number of reviews...and os on and name that column - num_reviews_group
    After that, num_reviews_group and rating will be categorized using and & or,
    and then, a new column will be created and will be given a name combined_info and it contains - 
    Best Quality, Better Quality, Good Quality, Average Quality
    Also, the extra columns wil be dropped - rating, num_reviews, num_reviews_group"""
    
    num_reviews_groups = pd.qcut(df['num_reviews'], q=4, labels=['Group 1', 'Group 2', 'Group 3', 'Group 4'], duplicates='drop')
    df['num_reviews_group'] = num_reviews_groups
    
    conditions = [
    (df['rating'] >= 4.8) & (df['num_reviews_group'] == 'Group 4'),
    ((df['rating'] >= 4.7) | (df['rating'] <= 4.6)) & (df['num_reviews_group'] == 'Group 3'),
    ((df['rating'] >= 4.5) | (df['rating'] <= 4.4)) & (df['num_reviews_group'] == 'Group 2'),
    ((df['rating'] >= 4.3) | (df['rating'] <= 4.2)) & (df['num_reviews_group'] == 'Group 1')
    ]
    
    choices = ['Best Quality', 'Better Quality', 'Good Quality', 'Average Quality']
    df['combined_info'] = np.select(conditions, choices, default='Low Quality')
    df.drop(columns = ['rating', 'num_reviews', 'num_reviews_group'], inplace = True)
    return df

def perform_label_encoding(df):
    """This function performs label encoding on the columns like - 
    winery, wine, region, type, combined_info"""
    
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    columns_to_encode = ['winery', 'wine', 'region', 'type', 'combined_info']
    
    original_values = {}  # dictionary to store the original values
    
    for column in columns_to_encode:
        df[column] = df[column].fillna('Unknown')
        original_values[column] = df[column].copy()  # store the original values
        df[column] = encoder.fit_transform(df[[column]]).astype(int)
    
    return df, original_values
        
def transform_encoded_data(df, original_values):
    
    """This function replace the encoded values with the original values"""
    
    for column, values in original_values.items():
        df[column] = values  

    return df

def use_model(df):
    
    """This function will use the pre-trained gradient boosting machine model"""
    
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    predictions = model.predict(df)
    return predictions

def merge_prediction(df, predictions):
    
    """This function will predict the price and merge it with the dataframe"""
    
    series = pd.Series(predictions, name='Predicted_Price')
    merged_df = df.merge(series, left_index=True, right_index=True)
    return merged_df

def preprocessing(df):
    
    """Here, all the other functions come together"""
    
    df = missing_values_n_datatypes(df)
    df = categorization(df)
    df, original_values = perform_label_encoding(df)
    predictions = use_model(df)
    df = merge_prediction(df, predictions)
    df = transform_encoded_data(df, original_values)
    return df

def main(args):
    
    file_path = args.file
    df = pd.read_csv(file_path)
    column_names = df.columns.tolist()
    column_length = len(list(df.columns))

    print("[CHECK] Name of Columns...:\n", column_names)
    print("[CHECK] Number of Columns...:", column_length)
    df.head()

    # col_names = ['wine', 'winery', 'year', 'rating', 'num_reviews', 'region', 'type','body', 'acidity']

    if column_length > 9:
        drop_column(df)
        print("\nThe column names that don't match are dropped!")
        column_length = len(list(df.columns))
        if column_length != 10 and column_length != 9:
            print("\nNot enough columns!")
        else:
            print("\n[INFO] Starting Data Preprocessing...")
            if 'price' in df:
                sav_price = df.price
                sav_price = list(sav_price)
                df = df.drop(columns='price')
                df = preprocessing(df)   # assigning the processed dataframe back to the df
                merged_df = merge_price_dataset(df, sav_price)
                print("\nPreprocessed and Merged the DataFrame.\n")
                print("[INFO] Process Completed! Output csv file, saved.")
                merged_df.to_csv("Output.csv")
            else:
                df = preprocessing(df)   # assigning the processed dataframe back to the df
                print("There is no 'PRICE' column in the dataset")
                print(df)
                df.to_csv("Output.csv")
    elif column_length == 9:     # that is price column is not present
        print("[INFO] Checking if the column names are the same...")
        if column_names == col_names:
            print("[INFO] Data Preprocessing...")
            df = preprocessing(df)
            df.to_csv("Output.csv")
            print("[INFO] Process Completed! Output csv file, saved.")
        else:
            print('The name of columns does not match!')
    else:
        print("\nThe number of columns should be at least 9")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data preprocessing and prediction script")
    parser.add_argument("file", type = str, help = "Path to the input CSV file")
    args = parser.parse_args()
    main(args)