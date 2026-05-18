import json
import os

def read_json_file(file_path):
    """
    Reads a JSON file and returns the parsed data.
    """
    try:
        if not os.path.exists(file_path):
            print(f"Error: The file '{file_path}' was not found.")
            return None
            
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            print(f"Successfully loaded {file_path}")
            return data
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON. {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    file_path = "../email_extract/email_data.json" 
    content = read_json_file(file_path)
    
    if content:
        # Display a snippet of the data
        print(json.dumps(content, indent=2)[:500] + "...")