# RAG-Document-Q-A-With-Groq-And-LLaMA3    Link - http://192.168.1.11:8506/

# Business Summary: RAG Document Q&A Chatbot with Groq and LLaMA3
# Purpose:
Build an intelligent chatbot that answers questions based only on a set of internal research documents (PDFs), using LLM and document retrieval.

# Document Ingestion & Embedding
Loads research PDFs from a specified folder (research_papers/).
Splits documents into smaller overlapping chunks for better processing and relevance.
Converts these chunks into vector embeddings using a Hugging Face model (all-MiniLM-L6-v2).
Stores the vectors in a FAISS vector database for fast and efficient similarity search.

# LLM Integration
Uses Groq’s ultra-fast LLaMA3-8B model as the large language model to generate responses.
Connects to the model via the ChatGroq interface, enabling low-latency responses.

# Prompt Engineering
Creates a structured prompt instructing the model to answer strictly from the provided document context.
Encourages accurate and context-aware responses while avoiding hallucination.

# RAG Workflow (Retrieval-Augmented Generation)
When a user asks a question:
The system retrieves the most relevant document chunks from the vector database.
Passes these chunks as context to the LLM.
The LLM generates a precise answer based only on the retrieved documents.

# Streamlit Frontend (UI)
Clean interface where:
Users input questions.
Can trigger document embedding with a button.
View generated answers and supporting context (matching docs).
Tracks and displays response time for performance transparency.

## Benefits
Ensures answers are grounded in internal company knowledge (not internet-based).
Fast, reliable performance using Groq’s inference engine.
Scalable for many types of documents and business domains (research, compliance, manuals, etc.).
Useful for knowledge management, customer support, and internal advisory tools.
