import os
import streamlit as st
import shutil
# from langchain.document_loaders import PyPDFLoader
# from langchain.vectorstores import Chroma
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

# Initialize chromadb, delete ids from db from previous run
persistent_db_path = "db_pdf_chatbot"
db = Chroma(persist_directory=persistent_db_path, embedding_function=embedding_function)
if db.get()["ids"]:
    db.delete(ids=db.get()["ids"])

# Get the current working directoryrrent working directory
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)

# System-Prompt definieren # TODO
system_prompt = """
Du bist ein KI-gestützter Assistent, der Informationen aus einem hochgeladenen PDF-Dokument extrahiert und präzise Antworten auf Fragen liefert. PDF-Dokument extrahiert und präzise Antworten auf Fragen liefert. 
Gib nur Inhalte zurück, die direkt im Dokument stehen, und füge keine eigenen Informationen hinzu. Falls die Frage nicht durch das Dokument beantwortet werden kann, antworte entsprechend.Gib nur Inhalte zurück, die direkt im Dokument stehen, und füge keine eigenen Informationen hinzu. Falls die Frage nicht durch das Dokument beantwortet werden kann, antworte entsprechend.
"""
# Chat laden
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)

# Streamlit App starten# Streamlit App starten
st.title("📄 Interaktiver PDF-Chatbot")

uploaded_file = st.file_uploader("Lade deine PDF-Datei hier hoch:", type="pdf")

# Specify params
chunk_overlap=200

uploaded_files = st.file_uploader(
    "Choose a CSV file", accept_multiple_files=True
)
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    st.write("filename:", uploaded_file.name)
    st.write(bytes_data)
if uploaded_file is not None:
    # Save the uploaded file to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Now use the path with PyPDFLoader
    pdf_loader = PyPDFLoader(tmp_path)
    documents = pdf_loader.load()
    # if uploaded_file:

    #     # PDF hochladen und Daten aus PDF lesenesen
    #     st.info("Lese Daten aus PDF ...", icon="📝")
    #     with open(uploaded_file, "wb") as temp_pdf:
    #         temp_pdf.write(uploaded_file.read())
    pdf_loader = PyPDFLoader(uploaded_file)
    docs = pdf_loader.load()
    total_length = sum(len(doc.page_content) for doc in docs)
    chunk_size = min(1000, total_length // 100)
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
# qa_chain.combine_documents_chain.llm_chain.prompt.messages.insert(0, {"role": "system", "content": system_prompt}) # TODO

query = st.text_input("Stelle eine Frage an die PDF:")
response = None
if uploaded_file is None:
    st.warning("Bitte PDF hochladen.")

if query is None:
    st.success("Der Chatbot ist bereit! Stelle jetzt deine Fragen.", icon="✨")
else:
    response = qa_chain.run(query)
    st.write("### Antwort:")
    st.write(response)
