import os

class RAGEngine:
    def __init__(self, vector_store, model_name=None):
        """
        Initialize a simplified RAG engine (without ML dependencies)
        
        Args:
            vector_store: Instance of VectorStore for document retrieval
            model_name: Placeholder for compatibility
        """
        self.vector_store = vector_store
    
    def query(self, query, collection=None, top_k=3, min_score=0.3):
        """
        Process a query using basic retrieval
        
        Args:
            query: The user query string
            collection: The specific collection to search in (if None, search all)
            top_k: Number of documents to retrieve
            min_score: Minimum similarity score threshold
            
        Returns:
            Tuple of (generated_answer, retrieved_context)
        """
        # Retrieve relevant documents
        retrieved_chunks = self.vector_store.search(
            query=query,
            collection=collection,
            top_k=top_k,
            min_score=min_score
        )
        
        if not retrieved_chunks:
            return "I couldn't find any relevant information to answer your question.", []
        
        # Simple answer generation from retrieved chunks
        answer = self._generate_answer_from_chunks(query, retrieved_chunks)
        
        return answer, retrieved_chunks
    
    def _generate_answer_from_chunks(self, query, chunks):
        """
        Generate a simple answer from retrieved chunks
        """
        if not chunks:
            return "No relevant information found."
            
        # Use multiple chunks for a more comprehensive answer
        # Sort the chunks by score
        sorted_chunks = sorted(chunks, key=lambda x: x["score"], reverse=True)
        
        # Extract key information from top chunks
        combined_text = ""
        for i, chunk in enumerate(sorted_chunks[:3]):  # Use top 3 chunks
            if chunk["score"] > 0.2:  # Only use chunks with reasonable relevance
                combined_text += chunk["text"] + " "
        
        # Extract relevant sentences that might answer the query
        query_words = set(query.lower().split())
        sentences = []
        
        # Split combined text into sentences (simple approach)
        text_sentences = combined_text.split('.')
        
        for sentence in text_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Count query words in the sentence
            word_matches = sum(1 for word in query_words if word in sentence.lower())
            
            # If the sentence has at least 2 query words, include it
            if word_matches >= min(2, len(query_words)/2):
                sentences.append(sentence + '.')
        
        # Combine the most relevant sentences into an answer
        if sentences:
            # Limit to 5 most relevant sentences to keep the answer concise
            relevant_info = ' '.join(sentences[:5])
            
            response = (
                f"Based on the information in your documents, here's what I found about your query:\n\n"
                f"{relevant_info}\n\n"
                f"This information was retrieved from your document collection. "
                f"You can see the source chunks in the 'Retrieved Information' section below."
            )
        else:
            # Fallback if no sentences are found
            response = (
                f"Based on the information I found, here's what might help answer your question:\n\n"
                f"{sorted_chunks[0]['text'][:350]}...\n\n"
                f"This information was retrieved from your document collection. "
                f"You can see more details in the 'Retrieved Information' section below."
            )
        
        return response
    
    def summarize(self, document_id, max_length=150):
        """
        Provide a simple excerpt from a document
        
        Args:
            document_id: The ID of the document to summarize
            max_length: Maximum length of the text to return
            
        Returns:
            Document excerpt
        """
        if document_id not in self.vector_store.documents:
            return "Document not found."
        
        # Get the document text
        doc_info = self.vector_store.documents[document_id]
        text = "\n\n".join(doc_info["text"]) if isinstance(doc_info["text"], list) else doc_info["text"]
        
        # Return a simple excerpt
        if len(text) > max_length:
            summary = text[:max_length] + "..."
        else:
            summary = text
            
        return f"Document excerpt:\n\n{summary}"
