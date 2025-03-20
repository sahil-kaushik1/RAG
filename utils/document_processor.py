import os
import io
import re
import PyPDF2
import pandas as pd
import speech_recognition as sr
import pdfplumber
from trafilatura import fetch_url, extract
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import numpy as np

def process_pdf(file_path):
    """
    Process a PDF file and extract text
    Uses PyPDF2, pdfplumber, and OCR (Tesseract) for image-based PDFs
    """
    text = ""
    debug_log = []  # Track processing steps for debugging
    
    # Create a debug log file
    debug_file = os.path.join("debug_logs", f"pdf_processing_{os.path.basename(file_path)}.log")
    
    debug_log.append(f"Starting PDF processing for {file_path}")
    
    # Try with PyPDF2 first
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            debug_log.append(f"PyPDF2: Found {len(reader.pages)} pages")
            
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                page_text = page.extract_text() or ""
                text += page_text + "\n\n"
                
            debug_log.append(f"PyPDF2: Extracted {len(text.strip())} characters")
    except Exception as e:
        error_msg = f"Error with PyPDF2: {str(e)}"
        debug_log.append(error_msg)
        print(error_msg)
    
    # If PyPDF2 didn't extract much, try pdfplumber
    if len(text.strip()) < 100:
        debug_log.append("Text too short, trying pdfplumber")
        try:
            with pdfplumber.open(file_path) as pdf:
                debug_log.append(f"pdfplumber: Found {len(pdf.pages)} pages")
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n\n"
                    
                debug_log.append(f"pdfplumber: Extracted {len(text.strip())} characters")
        except Exception as e:
            error_msg = f"Error with pdfplumber: {str(e)}"
            debug_log.append(error_msg)
            print(error_msg)
    
    # If still not much text, it might be an image-based PDF, try OCR
    if len(text.strip()) < 100:
        debug_log.append("Text still too short, attempting OCR on image-based PDF")
        print("Attempting OCR on image-based PDF...")
        
        try:
            # Create a temporary directory for images
            temp_dir = os.path.join("debug_logs", "temp_ocr")
            os.makedirs(temp_dir, exist_ok=True)
            debug_log.append(f"Created temp directory: {temp_dir}")
            
            # Convert PDF to images
            debug_log.append("Converting PDF to images...")
            try:
                images = convert_from_path(file_path, output_folder=temp_dir)
                debug_log.append(f"Conversion successful, got {len(images)} images")
            except Exception as e:
                error_msg = f"Error converting PDF to images: {str(e)}"
                debug_log.append(error_msg)
                print(error_msg)
                # Try with alternate parameters
                try:
                    debug_log.append("Trying with alternate parameters...")
                    images = convert_from_path(
                        file_path, 
                        output_folder=temp_dir,
                        dpi=200,  # Lower DPI
                        fmt='jpeg',  # Use JPEG format
                        thread_count=1,  # Single thread
                        use_pdftocairo=True  # Use pdftocairo instead of pdftoppm
                    )
                    debug_log.append(f"Alternate conversion successful, got {len(images)} images")
                except Exception as inner_e:
                    error_msg = f"Error with alternate conversion: {str(inner_e)}"
                    debug_log.append(error_msg)
                    print(error_msg)
                    images = []
            
            # Check if we have images to process
            if images:
                # Perform OCR on each image
                ocr_text = ""
                debug_log.append("Starting OCR on images...")
                
                for i, image in enumerate(images):
                    debug_log.append(f"Processing image {i+1}...")
                    
                    try:
                        # Save image for debugging
                        image_path = os.path.join(temp_dir, f"page_{i+1}.jpg")
                        image.save(image_path, "JPEG")
                        debug_log.append(f"Saved image to {image_path}")
                        
                        # Try to enhance image for better OCR
                        try:
                            # Convert to grayscale
                            gray_image = image.convert('L')
                            
                            # Increase contrast
                            enhanced_image = Image.fromarray(
                                255 * (np.array(gray_image) > 128).astype(np.uint8)
                            )
                            
                            # Save enhanced image
                            enhanced_path = os.path.join(temp_dir, f"enhanced_{i+1}.jpg")
                            enhanced_image.save(enhanced_path)
                            debug_log.append(f"Enhanced image saved to {enhanced_path}")
                            
                            # Use enhanced image for OCR
                            page_text = pytesseract.image_to_string(enhanced_image, lang='eng')
                        except:
                            # Fallback to original image if enhancement fails
                            debug_log.append("Image enhancement failed, using original image")
                            page_text = pytesseract.image_to_string(image, lang='eng')
                        
                        debug_log.append(f"OCR extracted {len(page_text)} characters from image {i+1}")
                        ocr_text += f"Page {i+1}:\n{page_text}\n\n"
                    except Exception as e:
                        error_msg = f"Error processing image {i+1}: {str(e)}"
                        debug_log.append(error_msg)
                        print(error_msg)
                
                # If OCR extracted text, use it
                if len(ocr_text.strip()) > 0:
                    text = ocr_text
                    success_msg = f"OCR extraction successful: {len(ocr_text)} characters"
                    debug_log.append(success_msg)
                    print(success_msg)
                else:
                    debug_log.append("OCR extraction produced no text")
                
                # We'll keep the temp files for debugging
                debug_log.append("Keeping temporary files for debugging")
            else:
                debug_log.append("No images were generated from PDF")
                
        except Exception as e:
            error_msg = f"Error with OCR: {str(e)}"
            debug_log.append(error_msg)
            print(error_msg)
    
    # Write debug log
    with open(debug_file, 'w') as f:
        f.write('\n'.join(debug_log))
    
    print(f"PDF processing complete: {len(text.strip())} characters extracted")
    return text.strip()

def process_csv(file_path):
    """
    Process a CSV file and convert it to text
    """
    try:
        df = pd.read_csv(file_path)
        # Get column descriptions
        columns_text = "Columns: " + ", ".join(df.columns) + "\n\n"
        
        # Get data overview
        rows_count = len(df)
        overview = f"This CSV file contains {rows_count} rows with the following columns: {', '.join(df.columns)}.\n\n"
        
        # Convert dataframe to text
        text_buffer = io.StringIO()
        df.to_string(text_buffer, index=False)
        data_text = text_buffer.getvalue()
        
        # Combine all text
        return overview + columns_text + data_text
    except Exception as e:
        print(f"Error processing CSV: {str(e)}")
        return ""

def process_audio(file_path):
    """
    Process an audio file and convert it to text using speech recognition
    """
    recognizer = sr.Recognizer()
    text = ""
    
    try:
        with sr.AudioFile(file_path) as source:
            # Adjust for ambient noise and record
            recognizer.adjust_for_ambient_noise(source)
            audio_data = recognizer.record(source)
            
            # Convert speech to text
            text = recognizer.recognize_google(audio_data)
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        return ""
    
    return text

def process_website(url):
    """
    Process a website URL and extract main content
    """
    try:
        # Download the content
        downloaded = fetch_url(url)
        if downloaded:
            # Extract the main content
            text = extract(downloaded)
            return text or ""
        return ""
    except Exception as e:
        print(f"Error processing website: {str(e)}")
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
    
    return text.strip()

def chunk_text(text, chunk_size=1000, overlap=200):
    """
    Split text into overlapping chunks for better embedding and retrieval
    Uses simple splitting method to avoid NLTK issues
    """
    if not text:
        return []
    
    chunks = []
    
    # Simple approach: split text by newlines and periods
    # This is less accurate than NLTK but more reliable without dependencies
    simple_sentences = []
    # Split by newlines first
    paragraphs = text.split('\n')
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
        # Then split by periods (approximate sentences)
        paragraph_sentences = paragraph.split('.')
        for sentence in paragraph_sentences:
            if sentence.strip():
                simple_sentences.append(sentence.strip() + '.')
    
    # Now chunk these simple sentences
    current_chunk = ""
    for sentence in simple_sentences:
        # If adding this sentence exceeds the chunk size, store the current chunk and start a new one
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Keep some overlap for context
            current_chunk = current_chunk[-overlap:] if overlap > 0 else ""
        
        current_chunk += " " + sentence
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # If we couldn't get proper sentences, fall back to character chunking
    if not chunks:
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if chunk:
                chunks.append(chunk)
    
    return chunks
