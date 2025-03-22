import os
import pickle
import uuid
import time
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple


class VectorStore:
    def __init__(self, model_name="all-MiniLM-L6-v2", use_embeddings=False):
        """
        Vector store implementation with better search functionality
        
        Args:
            model_name: Name of the embedding model (placeholder for future integration)
            use_embeddings: Whether to use actual embeddings or keyword search
        """
        # Document store to keep track of documents and their metadata
        self.documents = {}  # {doc_id: {"text": text, "metadata": metadata, "collection": collection, "embedding": embedding}}
        
        # Basic collection tracking
        self.collections = {}  # {collection_name: [doc_id, ...]}
        
        # Counter for assigning unique IDs
        self.next_idx = 0
        
        # Whether to use embeddings or keyword search
        self.use_embeddings = use_embeddings
        
        # Cache for commonly used queries
        self.search_cache = {}  # {(query, collection, top_k): [results]}
        self.cache_ttl = 3600  # Cache TTL in seconds (1 hour)
        self.cache_hits = 0
        self.cache_misses = 0
        
        try:
            # Import embedding model if available
            if use_embeddings:
                try:
                    from sentence_transformers import SentenceTransformer
                    self.embedding_model = SentenceTransformer(model_name)
                    print(f"Loaded embedding model: {model_name}")
                except ImportError:
                    print("sentence-transformers not installed. Falling back to keyword search.")
                    self.use_embeddings = False
                except Exception as e:
                    print(f"Error loading embedding model: {str(e)}. Falling back to keyword search.")
                    self.use_embeddings = False
        except Exception as e:
            print(f"Error initializing vector store: {str(e)}")
            self.use_embeddings = False
    
    def _compute_embedding(self, text):
        """Compute embedding for text if embedding model is available"""
        if self.use_embeddings:
            try:
                return self.embedding_model.encode(text)
            except:
                return None
        return None
    
    def _compute_similarity(self, query_embedding, document_embedding):
        """Compute cosine similarity between embeddings"""
        if query_embedding is None or document_embedding is None:
            return 0.0
            
        # Ensure embeddings are numpy arrays
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)
        if not isinstance(document_embedding, np.ndarray):
            document_embedding = np.array(document_embedding)
            
        # Normalize embeddings for cosine similarity
        query_norm = np.linalg.norm(query_embedding)
        doc_norm = np.linalg.norm(document_embedding)
        
        if query_norm == 0 or doc_norm == 0:
            return 0.0
            
        return np.dot(query_embedding, document_embedding) / (query_norm * doc_norm)
    
    def add_documents(self, texts, metadata=None):
        """
        Add documents to the vector store
        
        Args:
            texts: List of text chunks or a single text chunk
            metadata: Metadata to associate with the documents
            
        Returns:
            Document ID
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return None
        
        # Extract collection from metadata, default to "default"
        collection = metadata.get("collection", "default") if metadata else "default"
        document_id = metadata.get("document_id", f"doc_{uuid.uuid4()}") if metadata else f"doc_{uuid.uuid4()}"
        
        # Initialize collection if it doesn't exist
        if collection not in self.collections:
            self.collections[collection] = []
        
        # Add document to collection
        self.collections[collection].append(document_id)
        
        # Compute embeddings for each text chunk if available
        embeddings = []
        if self.use_embeddings:
            try:
                embeddings = [self._compute_embedding(chunk) for chunk in texts]
            except Exception as e:
                print(f"Error computing embeddings: {str(e)}")
                embeddings = [None] * len(texts)
        else:
            embeddings = [None] * len(texts)
        
        # Structure document chunks with their embeddings
        doc_chunks = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            doc_chunks.append({
                "text": text,
                "embedding": embedding,
                "chunk_id": i
            })
        
        # Store document information
        self.documents[document_id] = {
            "chunks": doc_chunks,
            "text": texts,  # Keep original list for backward compatibility
            "metadata": metadata.copy() if metadata else {},
            "collection": collection,
            "timestamp": time.time()
        }
        
        # Clear search cache as data has changed
        self.search_cache = {}
        
        return document_id
    
    def search(self, query, collection=None, top_k=5, min_score=0.0):
        """
        Search the vector store for relevant documents
        
        Args:
            query: The search query
            collection: The collection to search in (if None, search all)
            top_k: Number of results to return
            min_score: Minimum similarity score (0-1)
            
        Returns:
            List of document information
        """
        # Check cache first
        cache_key = (query, collection, top_k, min_score)
        cache_entry = self.search_cache.get(cache_key)
        
        if cache_entry:
            cache_time, results = cache_entry
            if time.time() - cache_time < self.cache_ttl:
                self.cache_hits += 1
                return results
        
        self.cache_misses += 1
        
        collections_to_search = [collection] if collection else list(self.collections.keys())
        
        all_results = []
        
        # Compute query embedding if using embeddings
        query_embedding = None
        if self.use_embeddings:
            try:
                query_embedding = self._compute_embedding(query)
            except Exception as e:
                print(f"Error computing query embedding: {str(e)}")
        
        # If embeddings are used and successfully computed
        if self.use_embeddings and query_embedding is not None:
            for collection_name in collections_to_search:
                if collection_name not in self.collections:
                    continue
                
                doc_ids = self.collections[collection_name]
                
                for doc_id in doc_ids:
                    doc_info = self.documents.get(doc_id)
                    if not doc_info or "chunks" not in doc_info:
                        continue
                    
                    # Process each chunk
                    for chunk in doc_info["chunks"]:
                        if "embedding" not in chunk or chunk["embedding"] is None:
                            continue
                            
                        # Compute similarity score
                        score = self._compute_similarity(query_embedding, chunk["embedding"])
                        
                        if score >= min_score:
                            all_results.append({
                                "text": chunk["text"],
                                "metadata": doc_info["metadata"],
                                "score": float(score),  # Convert to Python float
                                "doc_id": doc_id,
                                "chunk_id": chunk.get("chunk_id", 0)
                            })
        
        # Fall back to keyword search if no embeddings or no results from embeddings
        if not self.use_embeddings or not all_results:
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
                    chunks = doc_info.get("chunks", [])
                    texts = [chunk.get("text") for chunk in chunks]
                    
                    # If no chunks available, use the legacy format
                    if not texts and "text" in doc_info:
                        texts = doc_info["text"] if isinstance(doc_info["text"], list) else [doc_info["text"]]
                    
                    for i, text in enumerate(texts):
                        if not text:
                            continue
                            
                        text_lower = text.lower()
                        
                        # Advanced keyword scoring with TF-IDF-like weighting
                        # Count keyword occurrences and weight by importance
                        total_score = 0
                        matches = 0
                        
                        for keyword in keywords:
                            if keyword in text_lower:
                                # Count occurrences
                                occurrences = text_lower.count(keyword)
                                # Weight by keyword length (longer keywords more important)
                                keyword_weight = len(keyword) / 5  # normalize by typical word length
                                # Weight by frequency
                                frequency_weight = min(1.0, occurrences / 5)  # cap at 1.0
                                keyword_score = keyword_weight * frequency_weight
                                total_score += keyword_score
                                matches += 1
                        
                        # Normalize score
                        if not keywords:
                            score = 0
                        else:
                            # Consider both match ratio and weighted score
                            match_ratio = matches / len(keywords)
                            # Combine metrics, emphasizing match ratio
                            score = 0.7 * match_ratio + 0.3 * min(1.0, total_score / len(keywords))
                        
                        if score >= min_score:
                            # Check if this result is already in all_results (from embeddings)
                            duplicate = False
                            for existing in all_results:
                                if existing["doc_id"] == doc_id and existing.get("chunk_id", -1) == i:
                                    duplicate = True
                                    # Take the higher score
                                    existing["score"] = max(existing["score"], score)
                                    break
                            
                            if not duplicate:
                                all_results.append({
                                    "text": text,
                                    "metadata": doc_info["metadata"],
                                    "score": score,
                                    "doc_id": doc_id,
                                    "chunk_id": i
                                })
        
        # Sort by score and take top_k
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Filter out results below minimum score if specified
        if min_score > 0:
            all_results = [result for result in all_results if result["score"] >= min_score]
            results = all_results[:top_k]
        else:
            results = all_results[:top_k]
        
        # If no good matches AND no minimum score filter is set, provide random documents as fallback
        if not results and self.documents and min_score == 0:
            # Get random documents for demonstration
            doc_ids = list(self.documents.keys())
            sample_size = min(top_k, len(doc_ids))
            random_doc_ids = random.sample(doc_ids, sample_size)
            
            for doc_id in random_doc_ids:
                doc_info = self.documents.get(doc_id)
                if doc_info:
                    chunks = doc_info.get("chunks", [])
                    if chunks and chunks[0].get("text"):
                        text = chunks[0]["text"]
                    elif doc_info.get("text"):
                        text = doc_info["text"][0] if isinstance(doc_info["text"], list) and doc_info["text"] else doc_info["text"]
                    else:
                        continue
                        
                    results.append({
                        "text": text,
                        "metadata": doc_info["metadata"],
                        "score": 0.1,  # Low score indicating fallback
                        "doc_id": doc_id,
                        "chunk_id": 0
                    })
        
        # Store in cache
        self.search_cache[cache_key] = (time.time(), results)
        
        return results
    
    def delete_documents(self, document_id=None, collection=None):
        """
        Delete documents from the vector store
        
        Args:
            document_id: The document ID to delete (if None, delete all in collection)
            collection: The collection to delete from (if None, check all collections)
            
        Returns:
            Whether the deletion was successful
        """
        # Clear cache as data will change
        self.search_cache = {}
        
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
        
        Returns:
            Dictionary with collection statistics
        """
        stats = {}
        
        collections_to_check = [collection] if collection else list(self.collections.keys())
        
        for coll_name in collections_to_check:
            if coll_name in self.collections:
                doc_count = len(self.collections[coll_name])
                
                # Count all chunks across all documents in this collection
                chunk_count = 0
                avg_chunks_per_doc = 0
                total_text_size = 0
                
                for doc_id in self.collections[coll_name]:
                    doc_info = self.documents.get(doc_id, {})
                    
                    # Count chunks
                    chunks = doc_info.get("chunks", [])
                    if chunks:
                        chunk_count += len(chunks)
                        # Calculate total text size
                        for chunk in chunks:
                            if "text" in chunk:
                                total_text_size += len(chunk["text"])
                    # Fallback to legacy format
                    elif "text" in doc_info:
                        texts = doc_info["text"]
                        if isinstance(texts, list):
                            chunk_count += len(texts)
                            total_text_size += sum(len(t) for t in texts if t)
                        else:
                            chunk_count += 1
                            total_text_size += len(texts) if texts else 0
                
                if doc_count > 0:
                    avg_chunks_per_doc = chunk_count / doc_count
                
                # Calculate average chunk size
                avg_chunk_size = 0
                if chunk_count > 0:
                    avg_chunk_size = total_text_size / chunk_count
                
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
