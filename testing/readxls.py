import pandas as pd

def read_xls_file(file_path):
    """
    Reads an Excel .xls file and returns a pandas DataFrame.
    
    Note: You may need to install dependencies:
    pip install pandas xlrd
    """
    try:
        # Read the .xls file
        # xlrd engine is used for the older .xls format
        df = pd.read_excel(file_path, engine='xlrd')
        
        print(f"Successfully loaded {file_path}")
        return df
    except FileNotFoundError:
        print("Error: The file was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    file_path = "Details.xlsx" 
    data = read_xls_file(file_path)
    
    if data is not None:
        # Display the first few rows
        print(data.head())
