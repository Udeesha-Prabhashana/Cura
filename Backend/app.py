import json
from flask import Flask, request, jsonify, session
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import os
from flask_cors import CORS
from pdf_processing import process_pdf_and_store
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from operator import itemgetter
from langchain_core.tracers.log_stream import LogEntry, LogStreamCallbackHandler

app = Flask(__name__)
CORS(app)

conversation_chain = None
initialized = False 

app.secret_key = 'csccscscsc'

@app.before_request
def initialize_pdf_processing():
    """Process PDFs in the folder and initialize the conversation chain."""
    global conversation_chain, initialized
    if not initialized: 
        pdf_folder = "pdfs/"
        pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

        if pdf_files:
            for pdf_path in pdf_files:
                process_pdf_and_store(pdf_path)  # Process and store vectors
            vectorstore = get_vectorstore_from_file()
            conversation_chain = get_conversation_chain(vectorstore)
            print("PDFs processed and conversation chain initialized.")
        else:
            print("No PDFs found in the Backend/pdfs/ folder.")
        initialized = True

@app.before_request
def initialize_chat_history():
    """Initialize chat history at the beginning of each request."""
    if 'chat_history' not in session:
        session['chat_history'] = []

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle user questions."""
    global conversation_chain
    if not conversation_chain:
        return jsonify({'error': 'Conversation chain not initialized'}), 400

    data = request.json
    question = data.get('question', '')
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    # Append the current question to chat history
    session['chat_history'].append({"role": "user", "content": question})

    # Get response from the conversation chain, passing chat history as context
    response = get_response(question, session['chat_history'], conversation_chain)  # Pass chat history

    return jsonify({'answer': response['answer']}), 200

def get_response(question, chat_history, conversation_chain):
    """Get response from the conversation chain."""

    retriever_input = {
        'input': question,
        'chat_history': chat_history,  # Include chat history in the input
    }

    # Invoke the conversation chain with proper input
    response = conversation_chain.invoke(retriever_input)

    # Extract only the assistant's answer (assuming it's part of the response)
    answer = response.get('answer', 'Sorry, I don\'t know the answer.')  # Default fallback if answer is missing

    # Append the assistant's response to chat history
    chat_history.append({"role": "assistant", "content": answer})

    print(answer)
    return {'answer': answer}

def get_vectorstore_from_file():
    """Load the vectorstore from file"""
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(
        "Vector/hospital_vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

def create_chain(history_aware_retriever, llm, qa_prompt):
    """Create the chain for Question Answering"""
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

def get_conversation_chain(vectorstore):
    """Set up the conversational chain with a behavior prompt to limit the scope to hospital-related PDF questions."""
    # Initialize the LLM
    llm = ChatOpenAI()

    # Initialize the memory
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    # Define the prompt template
    system_prompt = (
        "You are Cura, the CareFlow Hospital Assistant. "
        "Your task is to answer questions based only on the hospital data provided in the PDF documents. "
        "Please use the retrieved context from the PDF to answer the question. If you don't know the answer, say that you don't know. "
        "If the question is not related to hospital data, respond with: 'Please ask questions related to CareFlow Hospital.' "
        "You should only answer questions that are relevant to the hospital information such as hospital services, doctors, facilities, and policies. "
        "Give your answer clearly and concisely."
        "\n\n"
        "{context}\n{chat_history}"
    )

    # Define the prompt template
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),  # Include chat history in the prompt
            ("human", "{input}"),
        ]
    )

    # Create the retrieval chain
    conversation_chain = create_chain(vectorstore.as_retriever(), llm, qa_prompt)

    return conversation_chain

def custom_jsonify(data):
    """Custom jsonify function to handle circular references"""
    try:
        return json.dumps(data, default=str) 
    except (TypeError, ValueError):
        return json.dumps({'error': 'Could not serialize the object to JSON.'})

if __name__ == '__main__':
    app.run(debug=True)
