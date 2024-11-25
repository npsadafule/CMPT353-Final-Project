"""
This script processes a dataset of property data by removing duplicate entries based on the 
'unique identifier' column ('id'). It keeps the latest occurrence of each duplicate and saves 
the cleaned dataset (clean_data/california_clean) to a new CSV file for further analysis.
"""


import pandas as pd

# Loading the dataset
file_path = 'data/california_raw.csv'
data = pd.read_csv(file_path)

# Remove duplicates based on 'id' column and keep the latest entry
cleaned_data = data.drop_duplicates(subset='id', keep='last')

# Save the cleaned dataset to a new file
cleaned_file_path = 'clean_data/california_clean.csv'
cleaned_data.to_csv(cleaned_file_path, index=False)

print("Duplicates removed based on 'id'. Cleaned file saved to:", cleaned_file_path)
