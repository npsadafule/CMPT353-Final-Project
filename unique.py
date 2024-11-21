import pandas as pd
import sys

def main():
    if len(sys.argv) != 2:
        print("Format: python3 unique.py <csv_file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]

    try:
        # Load the CSV file into a Pandas DataFrame
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: File '{file_path}' is empty.")
        sys.exit(1)
    except pd.errors.ParserError:
        print(f"Error: File '{file_path}' is not a valid CSV.")
        sys.exit(1)

    # Count the number of unique IDs
    unique_ids = data['id'].nunique()

    # Count the total number of records
    total_records = len(data)

    print(f"Total number of records: {total_records}")
    print(f"Number of unique IDs: {unique_ids}")

if __name__ == "__main__":
    main()
