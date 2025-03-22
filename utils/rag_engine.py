import os
from together import Together

class RAGEngine:
    def __init__(self, vector_store, model_name=None, api_key=None):
        """
        Initialize a RAG engine with Together AI integration
        
        Args:
            vector_store: Instance of VectorStore for document retrieval
            model_name: Placeholder for compatibility
            api_key: API key for Together AI
        """
        self.vector_store = vector_store
        self.api_key = api_key
        self.client = Together(api_key=api_key)
    
    def query(self, query, collection=None, top_k=3, min_score=0.3):
        """
        Process a query using retrieval and Together AI for answer generation
        
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
        
        # Generate an answer from retrieved chunks
        answer = self._generate_answer_from_chunks(query, retrieved_chunks)
        
        return answer, retrieved_chunks
    
    def _generate_answer_from_chunks(self, query, chunks):
        """
        Generate an answer using Together AI based on retrieved document chunks.
        """
        if not chunks:
            return "No relevant information found."

        # Sort and extract top chunks
        sorted_chunks = sorted(chunks, key=lambda x: x["score"], reverse=True)
        context = "\n".join([chunk["text"] for chunk in sorted_chunks[:3]])

        stream = self.client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant that provides accurate and concise answers based on the given context. If the context is insufficient, state that explicitly."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nProvide a well-structured answer based on the given context. If there is not enough information, say: 'The provided context does not contain sufficient information to answer this question.'"}
            ],
            stream=True,
        )

        answer = ""
        for chunk in stream:
            answer += chunk.choices[0].delta.content or ""

        return answer.strip()

    
    def summarize(self, document_id, max_length=150):
        """
        Provide a simple excerpt from a document
        
        Args:
            document_id: The ID of the document to summarize
            max_length: Maximum length of the text to return
            
        Returns:
            Document excerpt
        """
        doc_info = self.vector_store.documents.get(document_id)
        if not doc_info:
            return "Document not found."
        
        text = "\n\n".join(doc_info["text"]) if isinstance(doc_info["text"], list) else doc_info["text"]
        
        summary = text[:max_length] + "..." if len(text) > max_length else text
        
        return f"Document excerpt:\n\n{summary}"
