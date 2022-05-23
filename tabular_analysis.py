# %%
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from sqlalchemy import create_engine
import os
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import random
import glob
from PIL import Image
from sklearn import metrics
import time
import pickle as pkl

# %%
class ML():
    def __init__(self):
        """
        Initialize the class.
        """
        print("Initializing...")

    def get_data(self, table_name: str, the_columns: list):
        """When there's no csv, we go online to get the data.

        Args:
            - None

        Returns:
            - None
        """
        load_dotenv()
        DATABASE_TYPE = os.environ.get('DATABASE_TYPE')
        DBAPI = os.environ.get('DBAPI')
        ENDPOINT = os.environ.get('ENDPOINT')
        DBUSER = os.environ.get('DBUSER')
        DBPASSWORD = os.environ.get('DBPASSWORD')
        PORT = 5432
        DATABASE = os.environ.get('DATABASE')
        engine = create_engine(f"{DATABASE_TYPE}+{DBAPI}://{DBUSER}:"
                                    f"{DBPASSWORD}@{ENDPOINT}:"
                                    f"{PORT}/{DATABASE}")
        engine.connect()
        df = pd.read_sql_table(table_name,
                               engine,
                               columns=the_columns)
        return df

    def remove_n_a_rows(self, df: pd.DataFrame, column: str):
        """
        Scan the column for records with all N/As. Delete.

        Args:
            - column (str): The column being scanned.
        """
        # Swap N/A for the pandas nan, so we can drop them
        temp_df = df[column].replace('N/A', np.nan)
        temp_df = temp_df.dropna()
        # Create a new df with only the records without the nans
        clean_df = pd.merge(temp_df, df,
                            left_index=True, right_index=True)
        # The merge creates a duplicate column. Remove it.
        clean_df.drop(column + '_x', inplace=True, axis=1)
        # Rename the remaining category column
        clean_df.rename(columns={column + '_y': column}, inplace=True)
        # Commit the cleansed data to the dataframe
        df = clean_df
        return df

    def split_heirarchies(self, df, col: str, character: str, no_cols: int):
        """
        Takes in a column name and splits data to columns based on sep. char.

        Args:
            col (str): _description_
            character (str): _description_
            no_cols (int): _description_
        """
        df[[col+str(i) for i in range(no_cols)]] = (
            df[col].str.split(character, expand=True))
        df = df.drop(col, axis=1)
        if col == 'category':
            for i in range(no_cols):
                if i > 0:
                    df = df.drop(col+str(i), axis=1)
        return df
    
    def remove_currency_characters_from_price(self, df):
        """
        Remove the currency characters from the price column.

        Args:
            df (pandas.DataFrame): The dataframe to clean.
        """
        df['price'] = df['price'].str.replace('£', '')
        df['price'] = df['price'].str.replace(',', '')
        df = df.astype({"price": float}, errors='raise')
        df.reset_index(drop=True)
        return df

    def remove_price_outliers(self, df):
        """
        Remove the outliers from the price column.

        Args:
            df (pandas.DataFrame): The dataframe to clean.
        """
        df = df[df['price'] < 1000]
        df = df[df['price'] > 1]
        return df

    def separate_X_y_data(self, df, x_col: str, y_col: str):
        """
        Create a linear regression model.
        """
        X = df[[x_col]]
        y = df[y_col]
        X = pd.get_dummies(X, drop_first=True)
        return X, y

    def create_linear_regression_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        regr = LinearRegression()
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_test)
        # The coefficients
        print("Coefficients: \n", regr.coef_)
        # The mean squared error
        print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
        # The coefficient of determination: 1 is perfect prediction
        print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))
        return regr, y_pred, y_test

    def merge_dfs(self, df1, df2, left_on: str, right_on: str):
        return df1.merge(df2,
                         how='inner',
                         left_on=left_on,
                         right_on=right_on)



    def create_dict_of_categories(self, df: pd, column_to_index: str) -> None:
        categories_dict = set(df[column_to_index])
        categories_dict = {k: v for v, k in enumerate(categories_dict)}
        return categories_dict

    def create_df_of_image_paths(self) -> None:
        image_paths = glob.glob('resized64/*.jpg')
        image_paths = [x.split('/')[-1] for x in image_paths]
        image_paths = [x.split('.')[0] for x in image_paths]
        return image_paths

    def numberise_categories(self, df, column_to_index, categories_dict, image_paths) -> None:
        # replace the category column with the index of the category
        df[column_to_index] = df[column_to_index].map(categories_dict)
        image_df = pd.DataFrame({'image_path': image_paths})
        # merge the image_df with the df
        images_category_df = image_df.merge(df, how='inner', left_on='image_path', right_on='id_x')
        # drop all columns except the image_path and the category
        images_category_df = images_category_df[['image_path', column_to_index]]
        return images_category_df
        
    def prepare_image_category_datapoint(self, index: int, images_category_df) -> None:
        image = images_category_df['image_path'][index]
        image = Image.open('resized64/' + image + '.jpg')
        image = np.array(image)
        image = torch.from_numpy(image)
        image = torch.flatten(image)
        category = images_category_df['category'][index]
        return (image, category)

    def test_split(self, matrix, target, test_proportion):
        ratio = int(matrix.shape[0]/test_proportion) #should be int
        X_train = matrix[ratio:]
        X_test =  matrix[:ratio]
        y_train = target[ratio:]
        y_test =  target[:ratio]
        return X_train, X_test, y_train, y_test

    def save_model(self, model, path):
        torch.save(model.state_dict(), path)

#%%
if __name__ == "__main__":
    # Create instance of ML class
    fb = ML()
    # Get product data from database
    # test if local csv is present
    if os.path.isfile('data/Products.csv'):
        products_df = pd.read_csv('data/Products.csv')
    else:
        print("Downloading Product Data...")
        products_df = fb.get_data('products', ["id",
                                               "product_name",
                                               "category",
                                               "product_description",
                                               "price",
                                               "location",
                                               "page_id",
                                               "create_time"])
    # Clean the product data
    print("Cleaning Product Data...")
    print("Remove N/A rows...")
    products_df = fb.remove_n_a_rows(products_df, 'category')
    print("Split category into heirarchies...")
    num_cols = products_df['category'].str.count('/').max()+1
    products_df = fb.split_heirarchies(products_df, 'category', '/', num_cols)
    products_df.rename(columns={"category0": "category"}, inplace=True)
    print("Remove currency characters from price...")
    products_df = fb.remove_currency_characters_from_price(products_df)
    print("Remove price outliers...")
    products_df = fb.remove_price_outliers(products_df)
    
    # Give the option to display the category data distribution
    sns.boxplot(x='category', y='price', data=products_df)
    plt.show()
    
    # Create a linear regression model
    print("Creating Linear Regression Model...")
    X, y = fb.separate_X_y_data(products_df, 'category', 'price')
    # Test train split
    print("Split test and train, and then train the model...")
    fb.create_linear_regression_model(X, y)

    # Create a logistic regression model
    print("Creating Logistic Regression Model...")
    # Test if local csv is present
    if os.path.isfile('data/Images.csv'):
        image_df = pd.read_csv('data/Images.csv')
    else:
        # Get image data from database
        print("Downloading Image Data...")
        image_df = fb.get_data('images', ["id", 
                                          "product_id",
                                          "bucket_link",
                                          "image_ref",
                                          "create_time"])
    # Merge image data with product categories
    print("Merging Image Data with Product Categories...")
    df = fb.merge_dfs(image_df, products_df, 'product_id', 'id')

    cat_dict = fb.create_dict_of_categories(df, 'category')
    img_paths = fb.create_df_of_image_paths()
    img_cats_df = fb.numberise_categories(df, 'category', cat_dict, img_paths)

    # Shuffle the data
    print("Shuffling the data...")
    complete_dataset = []
    array_size = 12288
    n = len(img_cats_df)
    X = np.zeros((n, array_size))
    y = np.zeros(n)

    # create list of indexes to randomise
    indexes = list(range(n))
    random.shuffle(indexes)

    for idx in indexes:
        complete_dataset.append(fb.prepare_image_category_datapoint(idx, img_cats_df))
        
    for idx in range(n):    
        features, label = complete_dataset[idx]  
        X[idx, :] = features
        y[idx] = label
    
    # test train split
    print("Splitting test and train...")
    X_train, X_test, y_train, y_test = fb.test_split(X, y, 5)
    
# %%
    
    dump_X = pd.DataFrame(X)
    dump_y = pd.DataFrame(y)
    print("Exporting X and y to pickle files...")
    file_object = open("models/image_model_y.pkl", "wb")
    pkl.dump(dump_y, file_object)
    file_object.close()
    print("export of y complete - attempting x")
    file_object = open("models/image_model_X.pkl", "wb")
    pkl.dump(dump_X, file_object)
    file_object.close()

# %%
    print("Training the model...")
    print("start the timer...")
    start = time.time()
    model = LogisticRegression(max_iter=100)
    model.fit(X_train, y_train)
    print("stop the timer...")
    end = time.time()
    print("time taken: ", end - start)

    # make predictions
    print("Making predictions...")
    predictions = model.predict(X_test)


    # evaluate the model
    print("Evaluating the model...")
    score = model.score(X_test, y_test)
    print(score)

    # Show the confusion matrix
    print("Showing the confusion matrix...")
    cm = metrics.confusion_matrix(y_test, predictions)
    plt.figure(figsize=(13,13))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    all_sample_title = 'Accuracy Score: {0}'.format(score)
    plt.title(all_sample_title, size = 15);
    plt.show()

    # Show the categories, for comparison   
    print("Showing the categories...")
    print(cat_dict.keys())

