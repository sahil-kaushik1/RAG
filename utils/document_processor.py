import os
import io
import re
import yt_dlp
import re
import torch
import pandas as pd
from pydub import AudioSegment
import speech_recognition as sr
import pdfplumber
import whisper
from trafilatura import fetch_url, extract
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_together import Together
import ssl
import certifi
ssl_context = ssl.create_default_context(cafile=certifi.where())

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def initialize_langchain_components():
    # Vector store
    vector_store = FAISS

    # Together.ai LLM
    together_llm = Together(
        model="togethercomputer/llama-2-7b-chat",
        temperature=0.7,
        max_tokens=512,
        together_api_key=os.getenv("TOGETHER_API_KEY")  # Set your API key in environment variables
    )

    return vector_store, together_llm

import pdfplumber
import re
import requests
from pdf2image import convert_from_path
import pytesseract

import re
import pdfplumber
import pytesseract
import requests
from pdf2image import convert_from_path
from bs4 import BeautifulSoup

def process_pdf(file_path):
    """
    Extracts text and hyperlinks from a PDF, applies OCR to images, and fetches content from detected links (one level deep).
    """
    text_output = ""
    links = set()  # Store unique links

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""

            # Extract hyperlinks from annotations (embedded links)
            if "annots" in page.to_dict():
                for annot in page.annots:
                    if "uri" in annot:
                        links.add(annot["uri"])

            # Extract URLs from text using regex
            found_links = re.findall(r'(https?://[^\s]+)', page_text)
            links.update(found_links)

            # Extract images and apply OCR
            images = page.images
            if images:
                pil_images = convert_from_path(file_path, first_page=page_num+1, last_page=page_num+1)
                for img in pil_images:
                    ocr_text = pytesseract.image_to_string(img)
                    page_text += f"\n[OCR Extracted Text]: {ocr_text}\n"

                    # Extract URLs from OCR text
                    found_ocr_links = re.findall(r'(https?://[^\s]+)', ocr_text)
                    links.update(found_ocr_links)

            text_output += page_text + "\n"

    print("Extracted Links:", links)  # Debugging: Check if links are being extracted

    # Process extracted links (one level deep)
    for link in links:
        extracted_content = extract_content(link)
        if extracted_content:
            text_output += f"\n\n---\n\nContent from {link}:\n\n{extracted_content}"

    text_output = re.sub(r'\s+', ' ', text_output).strip()  # Clean excessive spaces
    return text_output

def extract_content(url):
    """
    Extracts content from a given URL using requests and BeautifulSoup.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")

            for tag in soup(["script", "style", "meta", "noscript", "header", "footer", "nav"]):
                tag.extract()

            text = soup.get_text(separator="\n")
            text = re.sub(r"\n+", "\n", text).strip()
            text = re.sub(r" +", " ", text)

            return text

        return ""

    except Exception as e:
        print(f"Error extracting content from {url}: {str(e)}")
        return ""
# import pdfplumber
# import re
# import pdfplumber
# import re
# from pdf2image import convert_from_path
# import pytesseract

# def process_pdf(file_path):
#     text_output = ""

#     with pdfplumber.open(file_path) as pdf:
#         for page_num, page in enumerate(pdf.pages):
#             print(f"Processing Page {page_num + 1}...")
            
#             # Extract normal text from the page
#             page_text = page.extract_text() or ""

#             # Extract hyperlinks
#             links = extract_links_from_page(page)
#             print(f"Extracted Links: {links}")
            
#             if links:
#                 page_text += "\n[Extracted Links]: " + " ".join(links) + "\n"

#                 # Fetch and append linked content
#                 for link in links:
#                     if link.startswith(("http://", "https://", "./", "/")):  # External link
#                         link_content = extract_external_link_content(link)
#                     else:  # Internal link
#                         link_content = extract_internal_link_content(pdf, link)
#                     page_text += f"\n[Content from {link}]: {link_content}\n"

#             # Extract images and apply OCR
#             if page.images:
#                 ocr_text = extract_ocr_text(file_path, page_num)
#                 page_text += f"\n[OCR Extracted Text]: {ocr_text}\n"

#             text_output += page_text + "\n"

#     # Clean up excessive spaces
#     text_output = re.sub(r'\s+', ' ', text_output).strip()
#     return text_output
# import requests

# def extract_external_link_content(destination):
#     """
#     Extract content from an external link destination.
    
#     Args:
#         destination (str): The external destination (e.g., a URL or file path).
    
#     Returns:
#         str: Extracted content from the destination.
#     """
#     try:
#         # If the destination is a URL
#         if destination.startswith(("http://", "https://")):
#             response = requests.get(destination)
#             if response.status_code == 200:
#                 return response.text  # Return the raw content
#             else:
#                 return f"Failed to fetch URL: {destination}"
        
#         # If the destination is a file path
#         elif destination.startswith(("./", "/")):
#             with open(destination, "r") as file:
#                 return file.read()
        
#         return ""  # Unsupported destination
#     except Exception as e:
#         print(f"Error extracting external link content: {e}")
#         return ""
    
# def extract_internal_link_content(pdf, destination):
#     """
#     Extract content from an internal link destination.
    
#     Args:
#         pdf: The pdfplumber PDF object.
#         destination (str): The internal destination (e.g., "Link-01" or "2").
    
#     Returns:
#         str: Extracted content from the destination.
#     """
#     try:
#         # If the destination is a page number
#         if destination.isdigit():
#             page_num = int(destination) - 1  # Convert to 0-based index
#             if 0 <= page_num < len(pdf.pages):
#                 return pdf.pages[page_num].extract_text() or ""
        
#         # If the destination is an ID
#         else:
#             # Iterate through all pages to find the ID
#             for page in pdf.pages:
#                 text = page.extract_text() or ""
#                 if destination in text:
#                     return text
        
#         return ""  # Destination not found
#     except Exception as e:
#         print(f"Error extracting internal link content: {e}")
#         return ""

# def extract_ocr_text(file_path, page_num):
#     """
#     Extract text from images in a PDF page using OCR.
    
#     Args:
#         file_path (str): Path to the PDF file.
#         page_num (int): Page number to process.
    
#     Returns:
#         str: Extracted OCR text.
#     """
#     ocr_text = ""
#     try:
#         # Convert the specific page to images
#         pil_images = convert_from_path(file_path, first_page=page_num + 1, last_page=page_num + 1)
#         for img in pil_images:
#             ocr_text += pytesseract.image_to_string(img) + "\n"
#     except Exception as e:
#         print(f"Error during OCR processing: {e}")
    return ocr_text.strip()

def fetch_link_content(link):
    """
    Fetch content from a hyperlink.
    
    Args:
        link (str): The URL to fetch content from.
    
    Returns:
        str: Extracted content from the link.
    """
    try:
        # Your existing fetch_link_content implementation
        # Example: Use requests or trafilatura to fetch content
        return "Fetched content from link"  # Replace with actual implementation
    except Exception as e:
        print(f"Error fetching content from {link}: {e}")
        return ""





# def process_csv(file_path):
#     loader = CSVLoader(file_path)
#     documents = loader.load()
#     text = "\n".join([doc.page_content for doc in documents])  # Extract text
#     return text
def process_csv(file_path):
    """
    Process a CSV file and convert it to text
    """
    try:
        # Try different encodings if default fails
        encodings = ['utf-8', 'latin-1', 'iso-8859-1']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            print(f"Failed to read CSV with any encoding: {file_path}")
            return ""
            
        # Get column descriptions
        columns_text = "Columns: " + ", ".join(df.columns) + "\n\n"
        
        # Get data overview
        rows_count = len(df)
        overview = f"This CSV file contains {rows_count} rows with the following columns: {', '.join(df.columns)}.\n\n"
        
        # Add data type information
        dtypes_info = "Column data types:\n"
        for col, dtype in df.dtypes.items():
            dtypes_info += f"- {col}: {dtype}\n"
        
        # Add basic statistics for numeric columns
        stats_info = "\nNumeric column statistics:\n"
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                stats_info += f"- {col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.2f}\n"
        
        # Convert dataframe to text (limit rows for large files)
        max_rows = min(50, len(df))
        text_buffer = io.StringIO()
        df.head(max_rows).to_string(text_buffer, index=False)
        data_text = text_buffer.getvalue()
        
        # Combine all text
        return overview + columns_text + dtypes_info + stats_info + "\nSample Data:\n" + data_text
    except Exception as e:
        print(f"Error processing CSV: {str(e)}")
        return ""

def sanitize_filename(name):
    """Removes special characters and emojis from the filename."""
    return re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')

def get_next_filename(output_dir, base_name, output_format):
    """Finds the next available 'vidX.wav' filename before downloading."""
    index = 1
    while True:
        filename = f"{base_name}{index}.{output_format}"
        file_path = os.path.join(output_dir, filename)
        if not os.path.exists(file_path):
            return file_path
        index += 1

def process_youtube(url, output_format="wav", output_dir="downloads"):
    try:
        os.makedirs(output_dir, exist_ok=True)

        # Generate final filename before downloading
        final_file = get_next_filename(output_dir, "vid", output_format)
        base_name = os.path.splitext(os.path.basename(final_file))[0]  # Extract 'vidX' part

        # yt-dlp options with controlled filename
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(output_dir, base_name) + ".%(ext)s",  # Match base_name
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': output_format,
                'preferredquality': '192',
            }],
        }

        # Download and process
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Find the converted file (should be 'vidX.wav')
        expected_file = os.path.join(output_dir, f"{base_name}.{output_format}")

        if not os.path.exists(expected_file):
            print(f"Error: Expected audio file '{expected_file}' not found.")
            return None

        print(f"Audio extracted successfully: {expected_file}")
        return process_audio(expected_file)

    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

def process_audio(file_path):
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    """
    Process an audio file and convert it to text using speech recognition
    """
    model = whisper.load_model("small")  # Load the Whisper model
    print("Processing audio...")
    result = model.transcribe(file_path,fp16=False)
    
    text = result["text"]
    print("Processed audio.")
    return text
import re
import requests
from bs4 import BeautifulSoup
from trafilatura import fetch_url, extract

import requests
import re
from bs4 import BeautifulSoup
from trafilatura import fetch_url, extract
from urllib.parse import urljoin, urlparse

def process_website(url):
    """
    Process a website URL, extract the main content, and follow all internal links (one level deep).
    """
    try:
        main_content = extract_content(url)

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract all internal links
            base_domain = urlparse(url).netloc
            links = set()  # Use a set to avoid duplicates

            for a in soup.find_all('a', href=True):
                link = urljoin(url, a['href'])  # Convert relative URLs to absolute
                if urlparse(link).netloc == base_domain:  # Keep only internal links
                    links.add(link)

            # Process all internal links (one level deep)
            for link in links:
                sub_content = extract_content(link)
                if sub_content:
                    main_content += f"\n\n---\n\nContent from {link}:\n\n{sub_content}"

        return main_content.strip()

    except Exception as e:
        print(f"Error processing website: {str(e)}")
        return ""

def extract_content(url):
    """
    Extract content from a single web page.
    """
    try:
        downloaded = fetch_url(url)
        if downloaded:
            text = extract(downloaded)
            if text and len(text.strip()) > 0:
                return text

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            for tag in soup(["script", "style", "meta", "noscript", "header", "footer", "nav"]):
                tag.extract()

            text = soup.get_text(separator='\n')
            text = re.sub(r'\n+', '\n', text).strip()
            text = re.sub(r' +', ' ', text)

            return text

        return ""

    except Exception as e:
        print(f"Error extracting content from {url}: {str(e)}")
        return ""


def preprocess_text(text):
    """
    Clean and preprocess text
    """
    if not text:
        return ""
    
    # Replace multiple newlines and spaces
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^\w\s\.,;:!?\'\"()-]', '', text)
    
    # Optional: Remove very short lines (often headers/footers)
    lines = text.split('\n')
    cleaned_lines = [line for line in lines if len(line.strip()) > 10]
    
    return '\n'.join(cleaned_lines).strip()

from nltk.tokenize import sent_tokenize

# def chunk_text(text, chunk_size=1000, overlap=200):
#     """
#     Split text into overlapping chunks while ensuring words are not split.
#     """
#     if not text:
#         return []
    
#     chunks = []
#     try:
#         # Use NLTK's sentence tokenizer
#         sentences = sent_tokenize(text)
        
#         current_chunk = []
#         current_length = 0
        
#         for sentence in sentences:
#             words = sentence.split()
#             sentence_length = sum(len(word) for word in words) + len(words) - 1  # Account for spaces
            
#             # If adding this sentence exceeds chunk size, store the current chunk and start a new one
#             if current_length + sentence_length > chunk_size and current_chunk:
#                 chunks.append(" ".join(current_chunk))
                
#                 # Keep overlap without breaking words
#                 overlap_text = " ".join(current_chunk[-(overlap // 5):])  # Approximate word-based overlap
#                 current_chunk = [overlap_text] if overlap_text else []
#                 current_length = sum(len(word) for word in current_chunk) + len(current_chunk) - 1
            
#             current_chunk.extend(words)
#             current_length += sentence_length
        
#         # Add the last chunk if not empty
#         if current_chunk:
#             chunks.append(" ".join(current_chunk))
    
#     except Exception as e:
#         print(f"Error using NLTK for chunking: {str(e)}. Falling back to simple chunking.")
#         words = text.split()
        
#         current_chunk = []
#         current_length = 0
        
#         for word in words:
#             word_length = len(word)
            
#             # If adding this word exceeds the chunk size, store the current chunk
#             if current_length + word_length > chunk_size and current_chunk:
#                 chunks.append(" ".join(current_chunk))
                
#                 # Keep overlap without breaking words
#                 overlap_text = " ".join(current_chunk[-(overlap // 5):])
#                 current_chunk = [overlap_text] if overlap_text else []
#                 current_length = sum(len(w) for w in current_chunk) + len(current_chunk) - 1
            
#             current_chunk.append(word)
#             current_length += word_length + 1  # Account for space
        
#         if current_chunk:
#             chunks.append(" ".join(current_chunk))
    
#     return chunks
from langchain_experimental.text_splitter import SemanticChunker
from nltk.tokenize import sent_tokenize
import logging
from langchain_huggingface import HuggingFaceEmbeddings

# Example with HuggingFace embeddings
em_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def chunk_text(text, embedding_model=em_model, min_chunk_size=200, max_chunk_size=400, overlap=200, breakpoint_percentile=90):
    """
    Split text into semantically meaningful chunks using LangChain's SemanticChunker.
    If LangChain fails, fallback to sentence-based chunking.
    
    Args:
        text (str): The text to be chunked
        embedding_model: The embedding model to use for semantic chunking (e.g., HuggingFaceEmbeddings)
        min_chunk_size (int): Minimum size of each chunk in characters
        max_chunk_size (int): Maximum size of each chunk in characters
        overlap (int): Number of characters to overlap between chunks
        breakpoint_percentile (int): Percentile threshold for determining breakpoints (1-100)
        
    Returns:
        list: List of text chunks
    """
    if not text:
        return []
    
    try:
        # Use LangChain's SemanticChunker with the provided embedding model
        chunker = SemanticChunker(
            embedding_model,
            min_chunk_size=min_chunk_size,
            breakpoint_threshold_type='percentile',
            breakpoint_threshold_amount=breakpoint_percentile
        )
        chunks = chunker.split_text(text)
        
        # Verify we actually got chunks
        if chunks and (len(chunks) > 1 or len(chunks[0]) <= max_chunk_size):
            return chunks
        else:
            print("Semantic chunking didn't properly split the text. Falling back to sentence-based chunking.")
    
    except Exception as e:
        print(f"Error using LangChain for chunking: {str(e)}. Falling back to sentence-based chunking.")
    
    # Fallback: Sentence-based chunking
    try:
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size and we already have content,
            # finalize the current chunk
            if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
                chunks.append(current_chunk)
                
                # Create overlap by including the last portion of the previous chunk
                if len(current_chunk) > overlap:
                    current_chunk = current_chunk[-overlap:]
                else:
                    current_chunk = ""
            
            # Add the sentence to the current chunk (with a space if needed)
            if current_chunk and not current_chunk.endswith(" "):
                current_chunk += " "
            current_chunk += sentence
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
        
    except Exception as e:
        print(f"Error with sentence-based chunking: {str(e)}. Falling back to character-based chunking.")
    
    # Final fallback: Character-based chunking
    chunks = []
    for i in range(0, len(text), max_chunk_size - overlap):
        end = min(i + max_chunk_size, len(text))
        chunk = text[i:end]
        chunks.append(chunk)
    print(chunks)
    return chunks


import nltk
from nltk.tokenize import word_tokenize
from rake_nltk import Rake

# Ensure NLTK dependencies are available
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

from rake_nltk import Rake
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

# Download necessary NLTK data (only required once)
from rake_nltk import Rake
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

# Download necessary NLTK data (only required once)
nltk.download('stopwords')
nltk.download('punkt')

def extract_keywords(text, min_length=4, max_words=2, min_score=3.0, max_keywords=10):
    """
    Extracts only the most meaningful keywords from the text.
    
    Args:
        text (str): The input text.
        min_length (int): Minimum length of a keyword.
        max_words (int): Maximum words in a keyword phrase.
        min_score (float): Minimum score required to consider a keyword.
        max_keywords (int): Maximum number of keywords to return.
    
    Returns:
        list: A list of top, most meaningful keywords.
    """
    if not isinstance(text, str):
        print(f"Warning: Expected string, got {type(text)} - {text}")
        return []

    # Initialize RAKE
    rake = Rake()
    rake.extract_keywords_from_text(text)
    
    # Get ranked phrases with scores
    ranked_phrases_with_scores = rake.get_ranked_phrases_with_scores()

    # Extract meaningful words (filtering stopwords and short words)
    stop_words = set(stopwords.words('english'))
    meaningful_keywords = []

    for score, phrase in ranked_phrases_with_scores:
        clean_phrase = re.sub(r'[^a-zA-Z0-9 ]', '', phrase).strip().lower()
        words = word_tokenize(clean_phrase)

        # Filter words: remove stopwords, short words, and keep only the best ones
        filtered_words = [
            word for word in words
            if word not in stop_words and len(word) >= min_length
        ]

        # Join back into a phrase, limit to max_words
        keyword = " ".join(filtered_words[:max_words])

        # Add only high-score keywords
        if keyword and score >= min_score:
            meaningful_keywords.append((keyword, score))

    # Sort by score in descending order and pick top max_keywords
    meaningful_keywords = sorted(meaningful_keywords, key=lambda x: x[1], reverse=True)
    top_keywords = [kw[0] for kw in meaningful_keywords[:max_keywords]]

    return top_keywords

def clean_keywords(word_list):
    """Remove special characters from a list of words."""
    clean_words = [re.sub(r'[^a-zA-Z0-9\s]', '', word) for word in word_list]
    return [word for word in clean_words if word.strip()]  # Remove empty strings