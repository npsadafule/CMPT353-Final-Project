"""
This script is designed to fetch property data from the RentCast API for a specified state 
(California) in batches, starting from a user-defined offset. It handles API requests, 
pagination, and data processing. Fetched data is appended to an existing CSV file, or a new 
CSV file is created if one does not already exist. It ensures that data is stored 
in a structured format (data/california_raw).
"""


import requests
import pandas as pd
from api import api

# Configuration
API_KEY = api  # API key from the `api` module
BASE_URL = "https://api.rentcast.io/v1/properties"
STATE = "CA"  # Replace with the state abbreviation ("CA" for California)
RECORDS_PER_REQUEST = 500  # Maximum records allowed per request
MAX_REQUESTS = 50  # Number of requests to make
CSV_FILE = "data/california_raw.csv"  # Output file

def fetch_property_data(offset):
    """
    Fetch property data for the specified state and offset using the API.
    """
    headers = {
        "accept": "application/json",
        "X-Api-Key": API_KEY
    }
    params = {
        "state": STATE,
        "limit": RECORDS_PER_REQUEST,
        "offset": offset
    }
    response = requests.get(BASE_URL, headers=headers, params=params)
    if response.status_code == 200:
        try:
            response_data = response.json()
            # Ensure the response is a list or adapt to the actual structure
            if isinstance(response_data, list):
                return response_data  # Directly return the list if that's the format
            elif isinstance(response_data, dict):
                return response_data.get("data", [])  # Adapt to dictionary structure
            else:
                print(f"Unexpected response format: {type(response_data)}")
                return []
        except ValueError:
            print(f"Invalid JSON response at offset {offset}: {response.text}")
            return []
    else:
        print(f"Error fetching data at offset {offset}: {response.status_code}")
        print(f"Response: {response.text}")
        return []

def main(start_offset):
    """
    Fetch the next batch of property data and append to an existing CSV file.
    """
    all_data = []
    print(f"Fetching data for state: {STATE} starting from offset {start_offset}")
    for request_num in range(MAX_REQUESTS):
        offset = start_offset + request_num * RECORDS_PER_REQUEST
        print(f"Requesting records with offset {offset}...")
        properties = fetch_property_data(offset)
        if not properties:  # Stop if no data is returned
            print(f"No more data available at offset {offset}.")
            break
        all_data.extend(properties)
        print(f"Fetched {len(properties)} records (offset: {offset}).")
    
    # Append new data to the existing CSV
    if all_data:
        df = pd.DataFrame(all_data)
        try:
            existing_df = pd.read_csv(CSV_FILE) # Load existing data
            updated_df = pd.concat([existing_df, df], ignore_index=True)
        except FileNotFoundError:
            print(f"{CSV_FILE} does not exist. Creating a new file.")
            updated_df = df
        updated_df.to_csv(CSV_FILE, index=False)
        print(f"Data successfully appended to {CSV_FILE}.")
        return df
    else:
        print("No new data fetched.")
        return pd.DataFrame()

if __name__ == "__main__":
    # Set the starting offset for the new batch 
    START_OFFSET = 75000
    raw_dataframe = main(START_OFFSET)
    print(f"Total new records fetched: {len(raw_dataframe)}")
