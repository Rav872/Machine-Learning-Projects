# House prediction using Linear Regression Machine learning implementation
import ssl
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Lets load the california house pricing dataset

from sklearn.datasets import fetch_california_housing
ssl._create_default_https_context = ssl._create_unverified_context
cali = fetch_california_housing()

# # Printing dataset information
# print(f"Type of dataset{type(cali)}") 
# print(f"keys of dataset {cali.keys()}")
# print(f"description: {cali}")
# print(f"Data {cali.data}")

# Lets create dataset

dataset = pd.DataFrame(cali.data, columns=cali.feature_names)
# print(dataset.head())  # It will give first 5 records

# Lets add the dependent feature price too
dataset['price']=cali.target

print(dataset.head())
print(dataset.info()) # will give the datatype of dataset parameters

# Summarizing the stats of the data
print(dataset.describe())





