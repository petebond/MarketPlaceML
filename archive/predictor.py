import classes.clean_images as ci
import classes.clean_tabular as ct
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


facebook = ct.Marketplace()
if facebook.not_already_downloaded():
    print("Go online and get data")
    facebook.connect_to_RDS_engine()
    print("Remove N/A categories")
    facebook.remove_n_a_records('category')
    col = 'category'
    char = '/'
    num = facebook.main_df[col].str.count(char).max()+1
    print("Split Heirarchies")
    facebook.split_heirarchies(col, char, num)
    col = 'product_name'
    char = ' in '
    num = facebook.main_df[col].str.count(char).max()+1
    facebook.split_heirarchies(col, char, num)
    print("Get rid of extra columns")
    facebook.clean_columns(num)
    # print("Create the numbers dataframe")
    # facebook.create_num_df()
    # print("Create the categorical dataframe")
    # facebook.create_cat_df()
    facebook.num_df.to_csv(r"data/cleaned.csv")
else:
    facebook.load_all_existing_data_to_dfs()

path = './images'
output_path = './resized'
clean_images = ci.CleanImages(path, output_path)
clean_images.clean(128)

# Create the model
ohe = OneHotEncoder(sparse=False)
ohe.fit_transform(facebook.num_df[['category']])

product_X = facebook.num_df.drop(['price'], axis=1)
product_y = facebook.num_df['category']

#use sklearn model for creating the test and train data
product_X_train, product_X_test, product_y_train, product_y_test = train_test_split(product_X, product_y, test_size=0.2, random_state=0)

regr = LinearRegression()
regr.fit(product_X_train, product_y_train)

product_y_pred = regr.predict(product_X_test)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(product_y_test, product_y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(product_y_test, product_y_pred))

# Plot outputs
plt.scatter(product_X_test, product_y_test, color="black")
plt.plot(product_X_test, product_y_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()


