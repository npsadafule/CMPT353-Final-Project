"""
This script processes the cleaned property dataset to extract and flatten nested data from 
specific columns (e.g., 'taxAssessments', 'propertyTaxes', 'features', and 'hoa'). It uses safe 
evaluation of strings to handle nested JSON-like structures, identifies relevant monetary and 
attribute-related columns, and ensures all extracted data is properly flattened and combined. 

The resulting dataset includes the essential columns and is saved as a new CSV file (clean_data/california_filtered.csv) for further 
analysis.
"""

import pandas as pd
import ast

# Load the dataset
file_path = 'clean_data/california_clean.csv'
data = pd.read_csv(file_path)

# Initialize a list of monetary and relevant columns to extract
columns_to_extract = [
    'id',
    'county',
    'latitude',
    'longitude',
    'squareFootage',
    'lotSize',
    'bedrooms',
    'bathrooms',
    'yearBuilt',
    'propertyType',
    'lastSalePrice',
    'ownerOccupied'
]

# Function to safely evaluate strings containing Python literals
def safe_eval(x):
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return {}

# Flatten nested columns: taxAssessments, propertyTaxes, features, hoa

# Flatten 'taxAssessments'
if 'taxAssessments' in data.columns:
    tax_assessments = pd.json_normalize(
        data['taxAssessments'].dropna().apply(safe_eval)
    )
    tax_assessments.columns = [f"taxAssessments.{col}" for col in tax_assessments.columns]
    data = pd.concat([data, tax_assessments], axis=1)
    monetary_columns = [col for col in tax_assessments.columns if any(keyword in col for keyword in ['value', 'land', 'improvements'])]
    columns_to_extract.extend(monetary_columns)

# Flatten 'propertyTaxes'
if 'propertyTaxes' in data.columns:
    property_taxes = pd.json_normalize(
        data['propertyTaxes'].dropna().apply(safe_eval)
    )
    property_taxes.columns = [f"propertyTaxes.{col}" for col in property_taxes.columns]
    data = pd.concat([data, property_taxes], axis=1)
    tax_columns = [col for col in property_taxes.columns if 'total' in col]
    columns_to_extract.extend(tax_columns)

# Flatten 'features'
if 'features' in data.columns:
    features = pd.json_normalize(
        data['features'].dropna().apply(safe_eval)
    )
    # Ensure consistent column naming
    features.columns = features.columns.str.replace('.', '_')
    data = pd.concat([data, features], axis=1)
    feature_columns = [
        'floorCount',
        'heatingType',
        'roomCount',
        'unitCount',
        'garageSpaces',
        'architectureType',
        'squareFootage',  # In case it's nested
        'lotSize',        # In case it's nested
        'bedrooms',       # In case it's nested
        'bathrooms'       # In case it's nested
    ]
    columns_to_extract.extend(feature_columns)

# Flatten 'hoa'
if 'hoa' in data.columns:
    hoa = pd.json_normalize(
        data['hoa'].dropna().apply(safe_eval)
    )
    hoa.columns = [f"hoa.{col}" for col in hoa.columns]
    data = pd.concat([data, hoa], axis=1)
    hoa_columns = ['hoa.fee']
    columns_to_extract.extend(hoa_columns)

# Remove duplicates and ensure columns exist
columns_to_extract = list(set(columns_to_extract))
existing_columns = [col for col in columns_to_extract if col in data.columns]

# Select only the relevant columns
cleaned_data = data[existing_columns].copy()

# Save the cleaned dataset to a new CSV file
output_file_path = 'clean_data/california_filtered.csv'
cleaned_data.to_csv(output_file_path, index=False)

print(f"Cleaned dataset with relevant columns saved to: {output_file_path}")
