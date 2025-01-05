import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def get_pdf_text(pdf_path):
    """Extract text from a PDF"""
    text = ""
    try:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text += page.extract_text()
    except Exception as e:
        print(f"Error while reading the PDF: {e}")
    return text

def get_text_chunks(text):
    """Split text into chunks for processing"""
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(text)

def process_pdf_and_store(pdf_path):
    """Process a PDF, create vector store, and save it"""
    # Check if the Backend folder exists; if not, create it
    backend_dir = "Vector"
    vectorstore_dir = os.path.join(backend_dir, "hospital_vectorstore")

    if not os.path.exists(backend_dir):
        os.makedirs(backend_dir)

    if not os.path.exists(vectorstore_dir):
        os.makedirs(vectorstore_dir)
    
    # Extract text from the PDF
    raw_text = get_pdf_text(pdf_path)
    
    if not raw_text:
        print(f"Error: No text extracted from {pdf_path}")
        return

    # Split the extracted text into smaller chunks
    text_chunks = get_text_chunks(raw_text)
    
    # Create embeddings for the chunks
    embeddings = OpenAIEmbeddings()

    # Create the FAISS vector store using the chunks and embeddings
    try:
        vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    except Exception as e:
        print(f"Error creating FAISS vector store: {e}")
        return

    # Save the vector store to a local directory
    try:
        vectorstore.save_local(vectorstore_dir)
        print(f"Vector store saved successfully to {vectorstore_dir}")
    except Exception as e:
        print(f"Error saving vector store: {e}")

