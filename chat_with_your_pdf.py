import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpointEmbeddings 
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv, find_dotenv

import logging
logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv(find_dotenv())

# embeddings
embedding_function = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY")
    )
logging.info(f"Embedding function: {embedding_function}")

# Initialize chromadb, delete ids from db from previous run
persistent_db_path = "db_pdf_chatbot"
logging.info(f"Persistent DB Path: {persistent_db_path}")
db = Chroma(persist_directory=persistent_db_path, embedding_function=embedding_function)
if db.get()["ids"]:
    logging.info(f"Deleting ids from db: {db.get()['ids']}")
    db.delete(ids=db.get()["ids"])
    logging.info("Deleted ids from db.")

# Get the current working directoryrrent working directory
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
logging.info(f"Current Directory: {current_dir}")

# Specify System-Prompt # TODO
system_prompt = """
Du bist ein KI-gestützter Assistent, der Informationen aus einem hochgeladenen PDF-Dokument extrahiert und präzise Antworten auf Fragen liefert. PDF-Dokument extrahiert und präzise Antworten auf Fragen liefert. 
Gib nur Inhalte zurück, die direkt im Dokument stehen, und füge keine eigenen Informationen hinzu. Falls die Frage nicht durch das Dokument beantwortet werden kann, antworte entsprechend.Gib nur Inhalte zurück, die direkt im Dokument stehen, und füge keine eigenen Informationen hinzu. Falls die Frage nicht durch das Dokument beantwortet werden kann, antworte entsprechend.
"""
# Load chat
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)
logging.info(f"LLM: {llm}")

# Set up Streamlit app
st.title("📄 Interaktiver PDF-Chatbot")
logging.info("Streamlit App gestartet")

# Upload PDF-Datei
uploaded_file = st.file_uploader("Lade deine PDF-Datei hier hoch:", type="pdf")
logging.info(f"Uploaded files: {uploaded_file}")

# Read the uploaded file as bytes
uploaded_file_bytes_data = uploaded_file.read()
st.write("filename:", uploaded_file.name)
st.write(uploaded_file_bytes_data)
logging.info(f"Uploaded file: {uploaded_file.name}, bytes_data: {uploaded_file_bytes_data}, length: {len(uploaded_file_bytes_data)}")

# Check if a file is uploaded
if uploaded_file is not None:
    # Save the uploaded file to a temp file
    with uploaded_file.name(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Now use the path with PyPDFLoader
    pdf_loader = PyPDFLoader(tmp_path)
    documents = pdf_loader.load()

    pdf_loader = PyPDFLoader(uploaded_file)
    docs = pdf_loader.load()
    total_length = sum(len(doc.page_content) for doc in docs)
    chunk_size = min(1000, total_length // 100)
    # Specify params
    chunk_overlap=200
    if chunk_overlap > chunk_size:
        chunk_overlap = chunk_size / 10        
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                              chunk_overlap=chunk_overlap,
                                              separators=["\n\n", "\n"," ", ".", ","])
    chunks = splitter.split_documents(docs)

    # Write data to db to db
    st.info("Daten mit Sprachmodell verarbeiten ...", icon="ℹ️")
    db.add_documents(chunks)

# Retrieval-QA erstellen mit System-Prompt
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 2}),
    verbose=True
)
logging.info("Retrieval-QA erstellt")
# qa_chain.combine_documents_chain.llm_chain.prompt.messages.insert(0, {"role": "system", "content": system_prompt}) # TODO

query = st.text_input("Stelle eine Frage an die PDF:")
logging.info(f"User query: {query}")
response = None
if uploaded_file is None:
    st.warning("Bitte PDF hochladen.")
    logging.warning("No PDF uploaded.")

if query is None:
    st.success("Der Chatbot ist bereit! Stelle jetzt deine Fragen.", icon="✨")
    logging.info("Chatbot is ready for questions.")
else:
    logging.info(f"User query: {query}")
    response = qa_chain.run(query)
    logging.info(f"Response: {response}")
    st.write("### Antwort:")
    st.write(response)
    logging.info("Response displayed")
