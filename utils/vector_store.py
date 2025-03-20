import os
import pickle
import uuid
import time
import random

class VectorStore:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Simple vector store implementation (simplified version without ML dependencies)
        """
        # Document store to keep track of documents and their metadata
        self.documents = {}  # {doc_id: {"text": text, "metadata": metadata, "collection": collection}}
        
        # Basic collection tracking
        self.collections = {}  # {collection_name: [doc_id, ...]}
        
        # Counter for assigning unique IDs
        self.next_idx = 0
    
    def add_documents(self, texts, metadata=None):
        """
        Add documents to the vector store
        Args:
            texts: List of text chunks or a single text chunk
            metadata: Metadata to associate with the documents
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return
        
        # Extract collection from metadata, default to "default"
        collection = metadata.get("collection", "default") if metadata else "default"
        document_id = metadata.get("document_id", f"doc_{uuid.uuid4()}") if metadata else f"doc_{uuid.uuid4()}"
        
        # Initialize collection if it doesn't exist
        if collection not in self.collections:
            self.collections[collection] = []
        
        # Add document to collection
        self.collections[collection].append(document_id)
        
        # Store document information
        self.documents[document_id] = {
            "text": texts,
            "metadata": metadata.copy() if metadata else {},
            "collection": collection
        }
        
        return document_id
    
    def search(self, query, collection=None, top_k=5, min_score=0.0):
        """
        Basic keyword search (simplified version without ML)
        Args:
            query: The search query
            collection: The collection to search in (if None, search all)
            top_k: Number of results to return
            min_score: Minimum similarity score (0-1)
        Returns:
            List of document information
        """
        collections_to_search = [collection] if collection else list(self.collections.keys())
        
        all_results = []
        
        # Split query into keywords
        keywords = query.lower().split()
        
        for collection_name in collections_to_search:
            if collection_name not in self.collections:
                continue
            
            doc_ids = self.collections[collection_name]
            
            for doc_id in doc_ids:
                doc_info = self.documents.get(doc_id)
                if not doc_info:
                    continue
                
                # For each text chunk in the document
                for text in doc_info["text"]:
                    text_lower = text.lower()
                    
                    # Basic matching score - count keyword occurrences
                    keyword_matches = sum(1 for keyword in keywords if keyword in text_lower)
                    if not keywords:
                        score = 0
                    else:
                        # Simple scoring mechanism
                        score = min(1.0, keyword_matches / len(keywords))
                    
                    if score >= min_score:
                        all_results.append({
                            "text": text,
                            "metadata": doc_info["metadata"],
                            "score": score,
                            "doc_id": doc_id
                        })
        
        # Sort by score and take top_k
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        # If no good matches, provide random documents as fallback for demo purposes
        if not all_results and self.documents:
            # Get random documents for demonstration
            doc_ids = list(self.documents.keys())
            sample_size = min(top_k, len(doc_ids))
            random_doc_ids = random.sample(doc_ids, sample_size)
            
            for doc_id in random_doc_ids:
                doc_info = self.documents.get(doc_id)
                if doc_info and doc_info["text"]:
                    text = doc_info["text"][0] if isinstance(doc_info["text"], list) else doc_info["text"]
                    all_results.append({
                        "text": text,
                        "metadata": doc_info["metadata"],
                        "score": 0.1,  # Low score indicating fallback
                        "doc_id": doc_id
                    })
        
        return all_results[:top_k]
    
    def delete_documents(self, document_id=None, collection=None):
        """
        Delete documents from the vector store
        Args:
            document_id: The document ID to delete (if None, delete all in collection)
            collection: The collection to delete from (if None, check all collections)
        """
        if document_id and document_id in self.documents:
            # Find the collection for this document
            doc_collection = self.documents.get(document_id, {}).get("collection", None)
            
            # If collection is specified, only delete if it matches
            if collection and doc_collection != collection:
                return False
            
            # Remove from collection
            if doc_collection and doc_collection in self.collections:
                if document_id in self.collections[doc_collection]:
                    self.collections[doc_collection].remove(document_id)
            
            # Remove from documents
            del self.documents[document_id]
            return True
        
        elif collection and collection in self.collections:
            # Get all document IDs in this collection
            doc_ids = self.collections[collection].copy()
            
            # Delete each document
            for doc_id in doc_ids:
                self.delete_documents(document_id=doc_id)
            
            # Clear the collection
            self.collections[collection] = []
            return True
        
        return False
    
    def get_collection_stats(self, collection=None):
        """
        Get statistics about a collection or all collections
        """
        stats = {}
        
        collections_to_check = [collection] if collection else list(self.collections.keys())
        
        for coll_name in collections_to_check:
            if coll_name in self.collections:
                doc_count = len(self.collections[coll_name])
                
                # Count all chunks across all documents in this collection
                chunk_count = 0
                for doc_id in self.collections[coll_name]:
                    doc_info = self.documents.get(doc_id, {})
                    texts = doc_info.get("text", [])
                    if isinstance(texts, list):
                        chunk_count += len(texts)
                    else:
                        chunk_count += 1
                
                stats[coll_name] = {
                    "document_count": doc_count,
                    "chunk_count": chunk_count
                }
        
        return stats
    
    def save(self, path="vector_store.pkl"):
        """
        Save the vector store to disk
        """
        state = {
            "documents": self.documents,
            "collections": self.collections,
            "next_idx": self.next_idx
        }
        
        # Save the state
        with open(path, "wb") as f:
            pickle.dump(state, f)
    
    def load(self, path="vector_store.pkl"):
        """
        Load the vector store from disk
        """
        if os.path.exists(path):
            with open(path, "rb") as f:
                state = pickle.load(f)
            
            self.documents = state["documents"]
            self.collections = state["collections"]
            self.next_idx = state["next_idx"]
            
            return True
        
        return False
