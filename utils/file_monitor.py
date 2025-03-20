import time
import os
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import hashlib
from .document_processor import (
    process_pdf, process_csv, process_audio, 
    preprocess_text, chunk_text
)

class DocumentChangeHandler(FileSystemEventHandler):
    def __init__(self, vector_store):
        super().__init__()
        self.vector_store = vector_store
        self.file_hashes = {}  # {file_path: hash}
        self.processing_lock = threading.Lock()
    
    def _get_file_hash(self, file_path):
        """Get MD5 hash of a file"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()
    
    def _get_file_extension(self, file_path):
        """Get the file extension"""
        _, ext = os.path.splitext(file_path)
        return ext.lower()
    
    def _process_file(self, file_path):
        """Process a file and update the vector store if needed"""
        if not os.path.exists(file_path):
            return
        
        # Get file hash
        current_hash = self._get_file_hash(file_path)
        
        # Skip if file hasn't changed
        if file_path in self.file_hashes and self.file_hashes[file_path] == current_hash:
            return
        
        # Update hash
        self.file_hashes[file_path] = current_hash
        
        # Process file based on extension
        ext = self._get_file_extension(file_path)
        
        # Find document ID from path (if it was processed before)
        file_name = os.path.basename(file_path)
        doc_id = None
        collection = None
        
        # Search for existing document with this path
        for d_id, info in self.vector_store.documents.items():
            metadata = info.get("metadata", {})
            if metadata.get("path") == file_path:
                doc_id = d_id
                collection = info.get("collection")
                break
        
        # If document exists, delete it first
        if doc_id:
            self.vector_store.delete_documents(document_id=doc_id)
        
        # Skip if we can't determine collection
        if not collection:
            return
        
        # Process the file
        text = None
        if ext == '.pdf':
            text = process_pdf(file_path)
        elif ext == '.csv':
            text = process_csv(file_path)
        elif ext in ['.mp3', '.wav', '.m4a']:
            text = process_audio(file_path)
        
        if text:
            # Preprocess and chunk the text
            preprocessed_text = preprocess_text(text)
            chunks = chunk_text(preprocessed_text)
            
            # Create metadata
            metadata = {
                "filename": file_name,
                "type": ext[1:],  # Remove the dot
                "collection": collection,
                "document_id": doc_id or f"{ext[1:]}_{file_name}_{int(time.time())}",
                "path": file_path
            }
            
            # Add to vector store
            self.vector_store.add_documents(chunks, metadata)
            print(f"Updated document in vector store: {file_name}")
    
    def on_modified(self, event):
        if not event.is_directory:
            # Use a lock to prevent multiple threads from processing the same file
            with self.processing_lock:
                # Process the file in a separate thread to avoid blocking
                threading.Thread(target=self._process_file, args=(event.src_path,)).start()
    
    def on_created(self, event):
        if not event.is_directory:
            with self.processing_lock:
                threading.Thread(target=self._process_file, args=(event.src_path,)).start()

def start_file_monitoring(directory, vector_store):
    """
    Start monitoring a directory for file changes
    
    Args:
        directory: The directory to monitor
        vector_store: The vector store to update
    """
    event_handler = DocumentChangeHandler(vector_store)
    observer = Observer()
    observer.schedule(event_handler, directory, recursive=True)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
