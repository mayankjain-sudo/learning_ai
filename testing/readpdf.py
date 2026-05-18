import PyPDF2

def read_pdf_file(file_path):
    """
    Reads a PDF file and returns the extracted text.
    
    Note: You need to install the dependency:
    pip install PyPDF2
    """
    try:
        with open(file_path, 'rb') as file:
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            
            text = ""
            # Iterate through all the pages and extract text
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
            
            print(f"Successfully loaded {file_path}")
            return text
    except FileNotFoundError:
        print("Error: The file was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    file_path = "your_file.pdf" 
    content = read_pdf_file(file_path)
    
    if content:
        # Display the first 500 characters
        print(content[:500])
