# from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# import os
# import tempfile
# from typing import List, Optional
# import json
# from utils.document_processor import (
#     process_pdf, process_csv, process_audio, 
#     process_website, preprocess_text, chunk_text
# )
# from utils.vector_store import VectorStore
# from utils.rag_engine import RAGEngine
# from utils.web_scraper import get_website_text_content

# app = FastAPI(title="Multimodal RAG API")

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Initialize vector store and RAG engine
# vector_store = VectorStore()
# rag_engine = RAGEngine(vector_store)

# # Create a temporary directory for file uploads
# temp_dir = tempfile.mkdtemp()

# @app.post("/ingest/pdf")
# async def ingest_pdf(
#     file: UploadFile = File(...),
#     collection: str = Form("default")
# ):
#     """
#     Ingest a PDF document and add it to the vector store
#     """
#     if not file.filename.lower().endswith('.pdf'):
#         raise HTTPException(status_code=400, detail="File must be a PDF")
    
#     # Save the uploaded file
#     file_path = os.path.join(temp_dir, file.filename)
#     with open(file_path, "wb") as f:
#         f.write(await file.read())
    
#     # Process the PDF
#     text = process_pdf(file_path)
#     preprocessed_text = preprocess_text(text)
#     chunks = chunk_text(preprocessed_text)
    
#     # Add to vector store
#     doc_id = f"pdf_{file.filename}_{int(os.path.getmtime(file_path))}"
#     metadata = {
#         "filename": file.filename,
#         "type": "pdf",
#         "collection": collection,
#         "document_id": doc_id,
#         "path": file_path
#     }
    
#     vector_store.add_documents(chunks, metadata)
    
#     return {
#         "status": "success",
#         "document_id": doc_id,
#         "collection": collection,
#         "chunks": len(chunks)
#     }

# @app.post("/ingest/csv")
# async def ingest_csv(
#     file: UploadFile = File(...),
#     collection: str = Form("default")
# ):
#     """
#     Ingest a CSV document and add it to the vector store
#     """
#     if not file.filename.lower().endswith('.csv'):
#         raise HTTPException(status_code=400, detail="File must be a CSV")
    
#     # Save the uploaded file
#     file_path = os.path.join(temp_dir, file.filename)
#     with open(file_path, "wb") as f:
#         f.write(await file.read())
    
#     # Process the CSV
#     text = process_csv(file_path)
#     preprocessed_text = preprocess_text(text)
#     chunks = chunk_text(preprocessed_text)
    
#     # Add to vector store
#     doc_id = f"csv_{file.filename}_{int(os.path.getmtime(file_path))}"
#     metadata = {
#         "filename": file.filename,
#         "type": "csv",
#         "collection": collection,
#         "document_id": doc_id,
#         "path": file_path
#     }
    
#     vector_store.add_documents(chunks, metadata)
    
#     return {
#         "status": "success",
#         "document_id": doc_id,
#         "collection": collection,
#         "chunks": len(chunks)
#     }

# @app.post("/ingest/audio")
# async def ingest_audio(
#     file: UploadFile = File(...),
#     collection: str = Form("default")
# ):
#     """
#     Ingest an audio file and add it to the vector store
#     """
#     if not file.filename.lower().endswith(('.mp3', '.wav', '.m4a')):
#         raise HTTPException(status_code=400, detail="File must be an audio file (MP3, WAV, M4A)")
    
#     # Save the uploaded file
#     file_path = os.path.join(temp_dir, file.filename)
#     with open(file_path, "wb") as f:
#         f.write(await file.read())
    
#     # Process the audio
#     text = process_audio(file_path)
#     if not text:
#         raise HTTPException(status_code=500, detail="Failed to extract text from audio")
    
#     preprocessed_text = preprocess_text(text)
#     chunks = chunk_text(preprocessed_text)
    
#     # Add to vector store
#     doc_id = f"audio_{file.filename}_{int(os.path.getmtime(file_path))}"
#     metadata = {
#         "filename": file.filename,
#         "type": "audio",
#         "collection": collection,
#         "document_id": doc_id,
#         "path": file_path
#     }
    
#     vector_store.add_documents(chunks, metadata)
    
#     return {
#         "status": "success",
#         "document_id": doc_id,
#         "collection": collection,
#         "chunks": len(chunks)
#     }

# @app.post("/ingest/website")
# async def ingest_website(url: str, collection: str = "default"):
#     """
#     Ingest content from a website URL and add it to the vector store
#     """
#     # Process the website
#     text = get_website_text_content(url)
#     if not text:
#         raise HTTPException(status_code=500, detail="Failed to extract content from website")
    
#     preprocessed_text = preprocess_text(text)
#     chunks = chunk_text(preprocessed_text)
    
#     # Add to vector store
#     doc_id = f"web_{url.replace('://', '_').replace('/', '_')}_{int(time.time())}"
#     metadata = {
#         "url": url,
#         "type": "website",
#         "collection": collection,
#         "document_id": doc_id
#     }
    
#     vector_store.add_documents(chunks, metadata)
    
#     return {
#         "status": "success",
#         "document_id": doc_id,
#         "collection": collection,
#         "chunks": len(chunks)
#     }

# @app.post("/query")
# async def query(
#     query: str,
#     collection: Optional[str] = None,
#     top_k: int = 3,
#     min_score: float = 0.3
# ):
#     """
#     Query the RAG system for an answer
#     """
#     response, retrieved_chunks = rag_engine.query(
#         query=query,
#         collection=collection,
#         top_k=top_k,
#         min_score=min_score
#     )
    
#     return {
#         "answer": response,
#         "retrieved_chunks": [
#             {
#                 "text": chunk["text"],
#                 "score": chunk["score"],
#                 "metadata": chunk["metadata"],
#                 "document_id": chunk["doc_id"]
#             }
#             for chunk in retrieved_chunks
#         ]
#     }

# @app.get("/collections")
# async def get_collections():
#     """
#     Get all available collections and their statistics
#     """
#     stats = vector_store.get_collection_stats()
    
#     collections = []
#     for coll_name, coll_stats in stats.items():
#         collections.append({
#             "name": coll_name,
#             "document_count": coll_stats["document_count"],
#             "chunk_count": coll_stats["chunk_count"]
#         })
    
#     return {"collections": collections}

# @app.delete("/collections/{collection}")
# async def delete_collection(collection: str):
#     """
#     Delete a collection and all its documents
#     """
#     success = vector_store.delete_documents(collection=collection)
    
#     if success:
#         return {"status": "success", "message": f"Collection '{collection}' deleted"}
#     else:
#         raise HTTPException(status_code=404, detail=f"Collection '{collection}' not found")

# @app.delete("/documents/{document_id}")
# async def delete_document(document_id: str):
#     """
#     Delete a specific document
#     """
#     success = vector_store.delete_documents(document_id=document_id)
    
#     if success:
#         return {"status": "success", "message": f"Document '{document_id}' deleted"}
#     else:
#         raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("api.endpoints:app", host="0.0.0.0", port=8000, reload=True)
