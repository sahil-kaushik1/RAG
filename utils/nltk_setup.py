"""
This module ensures that all required NLTK data is downloaded
"""
import nltk


def download_nltk_data():
    """
    Downloads necessary NLTK data packages
    """
    try:
        # Download commonly used NLTK data packages
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('stopwords')
        nltk.download('wordnet')
        print("NLTK data downloaded successfully")
    except Exception as e:
        print(f"Error downloading NLTK data: {str(e)}")

if __name__ == "__main__":
    download_nltk_data()