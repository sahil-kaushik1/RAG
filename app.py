import os
import tempfile
import pandas as pd
import re
import threading
import time
import subprocess
import sys
import pickle
import subprocess
import sys

# Try to import yt-dlp
try:
    import yt_dlp
except ImportError:
    print("yt-dlp not found, installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install",  "yt-dlp"])

# Your remaining app code...

from utils.document_processor import (
    process_pdf, process_csv, process_audio, 
    process_website, preprocess_text, chunk_text, process_youtube, extract_keywords, clean_keywords
)
from utils.vector_store import VectorStore
from utils.rag_engine import RAGEngine
from utils.file_monitor import start_file_monitoring
from utils.web_scraper import get_website_text_content
from utils.nltk_setup import download_nltk_data
# import asyncio
# asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# import torch

# torch.cuda.set_per_process_memory_fraction(0.1, 0)  # Limit to 50% of GPU 0 memory
import shutil

import asyncio
import os
import subprocess

# Try installing yt-dlp if not already installed

# Ensure an event loop exists before running Streamlit
import asyncio
import sys
import sys
import asyncio

# os.system("streamlit run app.py --server.port 8501 --server.address 0.0.0.0")

if sys.platform == "linux":
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

import sys
from unittest.mock import MagicMock

# Block Streamlit from inspecting torch._classes
sys.modules["torch.classes"] = MagicMock()
sys.modules["torch._classes"] = MagicMock()

# Now import Streamlit and other modules
import streamlit as st
import torch  # Safe to import after patching
import streamlit as st


try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Page configuration
st.set_page_config(
    page_title="Multimodal RAG System",
    page_icon="📚",
    layout="wide",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)


st.title("Your RAG Chatbot")



# Download required NLTK data
download_nltk_data()


# Initialize or load data from disk
DATA_DIR = "data"
VECTOR_STORE_PATH = os.path.join(DATA_DIR, "vector_store.pkl")
COLLECTIONS_PATH = os.path.join(DATA_DIR, "collections.pkl")

# Create data directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Initialize or load vector store
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()
    # Try to load from disk if exists
    if os.path.exists(VECTOR_STORE_PATH):
        try:
            st.session_state.vector_store.load(VECTOR_STORE_PATH)
            st.sidebar.success("Loaded existing vector store from disk")
        except Exception as e:
            st.sidebar.error(f"Error loading vector store: {str(e)}")

# Initialize RAG engine
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGEngine(st.session_state.vector_store)

# Initialize or load collections
if "collections" not in st.session_state:
    # Default empty collections
    st.session_state.collections = {}  
    # Try to load from disk if exists
    if os.path.exists(COLLECTIONS_PATH):
        try:
            with open(COLLECTIONS_PATH, 'rb') as f:
                st.session_state.collections = pickle.load(f)
            st.sidebar.success("Loaded existing collections from disk")
        except Exception as e:
            st.sidebar.error(f"Error loading collections: {str(e)}")

# Other session state variables
if "current_collection" not in st.session_state:
    # Set to first collection if available
    if st.session_state.collections:
        st.session_state.current_collection = next(iter(st.session_state.collections))
    else:
        st.session_state.current_collection = None
        
if "file_monitor_thread" not in st.session_state:
    st.session_state.file_monitor_thread = None
    
if "temp_dir" not in st.session_state:
    st.session_state.temp_dir = os.path.join(DATA_DIR, "temp")
    if not os.path.exists(st.session_state.temp_dir):
        os.makedirs(st.session_state.temp_dir)

# Function to save data
def save_data():
    try:
        # Save vector store
        st.session_state.vector_store.save(VECTOR_STORE_PATH)
        
        # Save collections
        with open(COLLECTIONS_PATH, 'wb') as f:
            pickle.dump(st.session_state.collections, f)
            
        return True
    except Exception as e:
        print(f"Error saving data: {str(e)}")
        return False
    
# Title and description
st.title("Multimodal RAG System")
st.write("""
This application allows you to upload various document types (PDFs, CSVs, voice files, web content),
process them to create vector embeddings, and then query the content using a RAG (Retrieval-Augmented Generation) approach.
""")

# Sidebar for collection management
with st.sidebar:
    st.header("Collection Management")
    
    # Create a new collection
    new_collection = st.text_input("Create New Collection", placeholder="Collection name")
    if st.button("Create Collection") and new_collection:
        if new_collection not in st.session_state.collections:
            st.session_state.collections[new_collection] = []
            st.session_state.current_collection = new_collection
            save_data()  # Save to disk
            st.success(f"Collection '{new_collection}' created!")
        else:
            st.error(f"Collection '{new_collection}' already exists!")
    
    # Select an existing collection
    if st.session_state.collections:
        select_collection = st.selectbox(
            "Select Collection", 
            list(st.session_state.collections.keys()),
            index=list(st.session_state.collections.keys()).index(st.session_state.current_collection) 
            if st.session_state.current_collection in st.session_state.collections else 0
        )
        if st.button("Switch to Collection"):
            st.session_state.current_collection = select_collection
            st.success(f"Switched to collection '{select_collection}'")
    
    # Delete a collection
    if st.session_state.collections:
        delete_collection = st.selectbox(
            "Delete Collection", 
            list(st.session_state.collections.keys()),
            key="delete_collection"
        )
        if st.button("Delete"):
            if delete_collection in st.session_state.collections:
                del st.session_state.collections[delete_collection]
                # Reset current collection if it was the deleted one
                if st.session_state.current_collection == delete_collection:
                    st.session_state.current_collection = next(iter(st.session_state.collections)) if st.session_state.collections else None
                save_data()  # Save to disk
                st.success(f"Collection '{delete_collection}' deleted!")
                st.rerun()
    
    # Show collection stats
    if st.session_state.current_collection:
        st.subheader(f"Collection: {st.session_state.current_collection}")
        docs_count = len(st.session_state.collections.get(st.session_state.current_collection, []))
        st.write(f"Documents: {docs_count}")

# Main content area with tabs
tab1, tab2, tab3 = st.tabs(["Document Upload", "Query System", "Collection Contents"])

# Move cleanup function and button to the top, outside tabs
def cleanup_temp_dir():
    if os.path.exists(st.session_state.temp_dir):
        shutil.rmtree(st.session_state.temp_dir)
        os.makedirs(st.session_state.temp_dir)

# Place this before the tabs, after session state initialization
st.button("Clean Temporary Files", on_click=cleanup_temp_dir, help="Remove all temporary files")
st.success("Temporary files cleaned!", icon="✅")  # This will only show briefly when clicked

# Document Upload Tab
with tab1:
    if not st.session_state.current_collection:
        st.warning("Please create or select a collection first.")
    else:
        st.header("Upload and Process Documents")
        
        # Select document type
        doc_type = st.radio(
            "Select document type to upload",
            ["PDF", "CSV", "Audio File", "Website URL", "YouTube Video"]
        )
        
        if doc_type == "PDF":
            uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
            if uploaded_file is not None:
                # Save the uploaded file to a temporary location
                temp_file_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                if st.button("Process PDF"):
                    with st.spinner("Processing PDF..."):
                        progress_container = st.empty()
                        for i in range(100):  # Simulate progress
                            progress_container.progress(i + 1)
                            time.sleep(0.01)
                        # Process the PDF and get chunks
                        text = process_pdf(temp_file_path)
                        dis_text = text
                        if isinstance(text, list):
                            text = "\n".join([doc.page_content if hasattr(doc, "page_content") else str(doc) for doc in text])

                        # print("hi")
                        # print(text)
                        # preprocessed_text = preprocess_text(text)
                        chunks = chunk_text(text)
                        
                        # Add chunks to vector store with metadata
                        doc_id = f"pdf_{uploaded_file.name}_{int(time.time())}"
                        metadata = {
                            "filename": uploaded_file.name,
                            "type": "pdf",
                            "collection": st.session_state.current_collection,
                            "document_id": doc_id
                        }
                        
                        st.session_state.vector_store.add_documents(chunks, metadata)
                        
                        # Add document info to collection
                        st.session_state.collections[st.session_state.current_collection].append({
                            "id": doc_id,
                            "filename": uploaded_file.name,
                            "type": "pdf",
                            "chunks": len(chunks),
                            "path": temp_file_path
                        })
                        
                        # Start file monitoring for automatic updates if not already running
                        if not st.session_state.file_monitor_thread or not st.session_state.file_monitor_thread.is_alive():
                            st.session_state.file_monitor_thread = threading.Thread(
                                target=start_file_monitoring,
                                args=(st.session_state.temp_dir, st.session_state.vector_store),
                                daemon=True
                            )
                            st.session_state.file_monitor_thread.start()
                        
                        save_data()  # Save to disk
                        st.success(f"PDF processed and added to collection '{st.session_state.current_collection}'!")
                        
                        # Display processed content
                        with st.expander("View Processed PDF Content", expanded=True):
                            st.text_area("Extracted Text", dis_text, height=300)
        
        elif doc_type == "CSV":
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded_file is not None:
                # Save the uploaded file to a temporary location
                temp_file_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Preview the CSV
                df = pd.read_csv(temp_file_path)
                st.write("CSV Preview:")
                st.dataframe(df.head())
                
                if st.button("Process CSV"):
                    with st.spinner("Processing CSV..."):
                        progress_container = st.empty()
                        for i in range(100):  # Simulate progress
                            progress_container.progress(i + 1)
                            time.sleep(0.01)
                        # Process the CSV and get chunks
                        text = process_csv(temp_file_path)
                        dis_text = text
                        if isinstance(text, list):
                            text = "\n".join([doc.page_content if hasattr(doc, "page_content") else str(doc) for doc in text])
                        # preprocessed_text = preprocess_text(text)
                        chunks = chunk_text(text)
                        
                        # Add chunks to vector store with metadata
                        doc_id = f"csv_{uploaded_file.name}_{int(time.time())}"
                        metadata = {
                            "filename": uploaded_file.name,
                            "type": "csv",
                            "collection": st.session_state.current_collection,
                            "document_id": doc_id
                        }
                        
                        st.session_state.vector_store.add_documents(chunks, metadata)
                        
                        # Add document info to collection
                        st.session_state.collections[st.session_state.current_collection].append({
                            "id": doc_id,
                            "filename": uploaded_file.name,
                            "type": "csv",
                            "chunks": len(chunks),
                            "path": temp_file_path
                        })
                        
                        save_data()  # Save to disk
                        st.success(f"CSV processed and added to collection '{st.session_state.current_collection}'!")
                        
                        # Display processed content
                        with st.expander("View Processed CSV Content", expanded=True):
                            st.text_area("Extracted Text", text, height=300)
        
        elif doc_type == "Audio File":
            uploaded_file = st.file_uploader("Upload Audio File", type=["mp3", "wav", "m4a"])
            if uploaded_file is not None:
                # Save the uploaded file to a temporary location
                temp_file_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                if st.button("Process Audio"):
                    with st.spinner("Processing Audio... This might take a while."):
                        progress_container = st.empty()
                        for i in range(100):  # Simulate progress
                            progress_container.progress(i + 1)
                            time.sleep(0.01)
                        try:
                            # Process the audio and get text
                            text = process_audio(temp_file_path)
                            if text:
                                # preprocessed_text = preprocess_text(text)
                                chunks = chunk_text(text)
                                
                                # Add chunks to vector store with metadata
                                doc_id = f"audio_{uploaded_file.name}_{int(time.time())}"
                                metadata = {
                                    "filename": uploaded_file.name,
                                    "type": "audio",
                                    "collection": st.session_state.current_collection,
                                    "document_id": doc_id
                                }
                                
                                st.session_state.vector_store.add_documents(chunks, metadata)
                                
                                # Add document info to collection
                                st.session_state.collections[st.session_state.current_collection].append({
                                    "id": doc_id,
                                    "filename": uploaded_file.name,
                                    "type": "audio",
                                    "chunks": len(chunks),
                                    "path": temp_file_path
                                })
                                
                                save_data()  # Save to disk
                                st.success(f"Audio processed and added to collection '{st.session_state.current_collection}'!")
                                
                                # Display processed content
                                with st.expander("View Processed Audio Content", expanded=True):
                                    st.text_area("Extracted Text", text, height=300)
                            else:
                                st.error("Failed to extract text from the audio file.")
                        except Exception as e:
                            st.error(f"Error processing audio file: {str(e)}")
        
        elif doc_type == "Website URL":
            url = st.text_input("Enter Website URL", placeholder="https://example.com")
            
            if st.button("Process Website") and url:
                with st.spinner("Processing Website..."):
                    progress_container = st.empty()
                    for i in range(100):  # Simulate progress
                        progress_container.progress(i + 1)
                        time.sleep(0.01)
                    try:
                        # Extract content from the website
                        text = process_website(url)
                        if text:
                            # preprocessed_text = preprocess_text(text)
                            chunks = chunk_text(text)
                            
                            # Add chunks to vector store with metadata
                            doc_id = f"web_{url.replace('://', '_').replace('/', '_')}_{int(time.time())}"
                            metadata = {
                                "url": url,
                                "type": "website",
                                "collection": st.session_state.current_collection,
                                "document_id": doc_id
                            }
                            
                            st.session_state.vector_store.add_documents(chunks, metadata)
                            
                            # Add document info to collection
                            st.session_state.collections[st.session_state.current_collection].append({
                                "id": doc_id,
                                "url": url,
                                "type": "website",
                                "chunks": len(chunks)
                            })
                            
                            save_data()  # Save to disk
                            st.success(f"Website content processed and added to collection '{st.session_state.current_collection}'!")
                            
                            # Display processed content
                            with st.expander("View Processed Website Content", expanded=True):
                                st.text_area("Extracted Text", text, height=300)
                        else:
                            st.error("Failed to extract content from the website.")
                    except Exception as e:
                        st.error(f"Error processing website: {str(e)}")
        elif doc_type == "YouTube Video":  # Consistent naming
            url = st.text_input("Enter YouTube URL", placeholder="https://www.youtube.com/watch?v=VIDEO_ID")
            
            if st.button("Process Video") and url:
                with st.spinner("Processing video..."):
                    progress_container = st.empty()
                    for i in range(100):  # Simulate progress
                        progress_container.progress(i + 1)
                        time.sleep(0.01)
                    try:
                        # Extract audio and transcribe text from the YouTube video
                        text = process_youtube(url)
                        if text:
                            # preprocessed_text = preprocess_text(text)
                            chunks = chunk_text(text)
                            
                            # Add chunks to vector store with metadata
                            doc_id = f"video_{url.replace('://', '_').replace('/', '_')}_{int(time.time())}"
                            metadata = {
                                "url": url,
                                "type": "youtube_video",  # Consistent naming
                                "collection": st.session_state.current_collection,
                                "document_id": doc_id
                            }
                            
                            st.session_state.vector_store.add_documents(chunks, metadata)
                            
                            # Add document info to collection
                            st.session_state.collections[st.session_state.current_collection].append({
                                "id": doc_id,
                                "url": url,
                                "type": "youtube_video",  # Consistent naming
                                "chunks": len(chunks)
                            })
                            
                            save_data()  # Save to disk
                            st.success(f"YouTube video content processed and added to collection '{st.session_state.current_collection}'!")
                            
                            # Display processed content
                            with st.expander("View Processed YouTube Content", expanded=True):
                                st.text_area("Extracted Text", text, height=800)
                        else:
                            st.error("Failed to extract content from the YouTube video.")
                    except Exception as e:
                        st.error(f"Error processing YouTube video: {str(e)}")

# Query System Tab
with tab2:
    if not st.session_state.current_collection or not st.session_state.collections.get(st.session_state.current_collection, []):
        st.warning("Please select a collection with documents to query.")
    else:
        st.header("Query Your Documents")
        st.write("Ask questions about your documents, and the system will retrieve relevant information.")
        
        # Query parameters
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=3)
        with col2:
            min_score = st.slider("Minimum similarity score (0-1)", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
        
        # Query input
        query = st.text_input("Enter your query", placeholder="What information are you looking for?")
        
        if st.button("Search") and query:
            with st.spinner("Searching and generating response..."):
                # Get the RAG response
                response, retrieved_chunks = st.session_state.rag_engine.query(
                    query, 
                    collection=st.session_state.current_collection, 
                    top_k=top_k,
                    min_score=min_score
                )
                
                # Display the response
                st.subheader("AI Response")
                st.write(response)
                
                # Display retrieved chunks for transparency
                st.subheader("Retrieved Information")
                if retrieved_chunks:
                    for i, chunk_info in enumerate(retrieved_chunks):
                        with st.expander(f"Chunk {i+1} (Similarity: {chunk_info['score']:.4f})"):
                            st.write(f"*Source*: {chunk_info['metadata'].get('filename', chunk_info['metadata'].get('url', 'Unknown'))}")
                            st.write(f"*Type*: {chunk_info['metadata'].get('type', 'Unknown')}")

                            keywords = extract_keywords(chunk_info.get('text', ''))
                            keywords = clean_keywords(keywords)

                            text = chunk_info['text']

                            for keyword in sorted(keywords, key=len, reverse=True):
                                pattern = rf'(?<!\w)({re.escape(keyword)})(?!\w)'  # Match whole words only
                                text = re.sub(pattern, rf'<span style="color:#FF6F61; font-weight:bold;">\1</span>', text, flags=re.IGNORECASE)

                            st.markdown(f'<div style="line-height:1.6">{text}</div>', unsafe_allow_html=True)  

                            if keywords:
                                st.markdown("")
                                st.markdown("###### Keywords - ")
                                st.markdown(", ".join([f'<span style="font-weight:bold;">{kw}</span>' for kw in keywords]), unsafe_allow_html=True)
                else:
                    st.info("No relevant information found.")

# Collection Contents Tab
with tab3:
    if not st.session_state.current_collection:
        st.warning("Please select a collection to view its contents.")
    else:
        st.header(f"Contents of Collection: {st.session_state.current_collection}")
        
        collection_docs = st.session_state.collections.get(st.session_state.current_collection, [])
        if not collection_docs:
            st.info("This collection is empty. Add documents in the 'Document Upload' tab.")
        else:
            # Display table of documents
            st.subheader("Documents in Collection")
            
            # Create a DataFrame for display
            docs_data = []
            for doc in collection_docs:
                docs_data.append({
                    "ID": doc.get("id", ""),
                    "Name": doc.get("filename", doc.get("url", "Unknown")),
                    "Type": doc.get("type", "Unknown"),
                    "Chunks": doc.get("chunks", 0)
                })
            
            docs_df = pd.DataFrame(docs_data)
            st.dataframe(docs_df)
            
            # Option to remove a document
            if docs_data:
                doc_to_remove = st.selectbox(
                    "Select document to remove", 
                    options=[d["ID"] for d in docs_data],
                    key="remove_doc"
                )
                
                if st.button("Remove Document"):
                    # Remove from collections
                    st.session_state.collections[st.session_state.current_collection] = [
                        doc for doc in collection_docs if doc.get("id") != doc_to_remove
                    ]
                    
                    # Remove from vector store
                    st.session_state.vector_store.delete_documents(
                        document_id=doc_to_remove,
                        collection=st.session_state.current_collection
                    )
                    
                    save_data()  # Save to disk
                    st.success(f"Document removed from collection!")
                    st.rerun()