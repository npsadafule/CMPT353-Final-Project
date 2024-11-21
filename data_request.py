import requests
import pandas as pd
from api import api1, api2

# Configuration
API_KEY = api2 # Using the second API key
BASE_URL = "https://api.rentcast.io/v1/properties"
STATE = "TX"  # Fetching for Texas
RECORDS_PER_REQUEST = 500  # Max records allowed per request
MAX_REQUESTS = 49  # Limit per API restrictions (lost 1 request while testing again)
CSV_FILE = "texas_property_data.csv"  # Output file

def fetch_property_data(offset):
    """
    Fetch property data for Texas with the specified offset using the API.
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
            # Mking sure the response is a list, or adapt to the actual structure
            if isinstance(response_data, list):
                return response_data  # Directly return the list if that is the format
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

def main():
    """
    Main function to fetch property data for Texas and save to CSV.
    """
    all_data = []
    print(f"Fetching data for state: {STATE}")
    for request_num in range(MAX_REQUESTS):
        offset = request_num * RECORDS_PER_REQUEST
        print(f"Requesting records with offset {offset}...")
        properties = fetch_property_data(offset)
        if not properties:  # Stop if no data is returned
            print(f"No more data available at offset {offset}.")
            break
        all_data.extend(properties)
        print(f"Fetched {len(properties)} records (offset: {offset}).")
    
    # Save fetched data to CSV
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(CSV_FILE, index=False)
        print(f"Data successfully saved to {CSV_FILE}.")
        return df
    else:
        print("No data fetched.")
        return pd.DataFrame()

if __name__ == "__main__":
    raw_dataframe = main()
    print(f"Total records fetched: {len(raw_dataframe)}")
