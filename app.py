import os
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpointEmbeddings 
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv, find_dotenv

import logging
logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv(find_dotenv())

# Streamlit App
st.set_page_config(page_title="PDF-Chatbot", page_icon=":robot_face:", layout="wide")

# Temperaturwerte vorbereiten (0.0 bis 1.0 in 0.1er-Schritten)
temperature_options = [round(i * 0.1, 1) for i in range(11)]  # [0.0, 0.1, ..., 1.0]

# Initialize Flag indicating Chatbot is ready
chatbot_ready = False

# embeddings
embedding_function = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY")
    )
logging.info(f"Embedding function: {embedding_function}")

# Initialize chroma db, delete ids from db from previous run
persistent_db_path = "db_pdf_chatbot"
logging.info(f"Persistent DB Path: {persistent_db_path}")
db = Chroma(persist_directory=persistent_db_path, embedding_function=embedding_function)
if db.get()["ids"]:
    logging.info(f"Deleting ids from db: {db.get()['ids']}")
    db.delete(ids=db.get()["ids"])
    logging.info("Deleted ids from db.")

# Get the current working directory and file path
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
logging.info(f"Current Directory: {current_dir}")

# Define Prompt-Template
system_prompt = """
Du bist ein hilfreicher, KI-gestützter Assistent. 
Dir wird ein PDF-Dokument zur Verfügung gestellt, das vom Nutzer hochgeladen wurde.
Deine Aufgabe ist es, ausschließlich auf Basis der Informationen aus diesem Dokument präzise und verständliche Antworten auf Fragen zu geben.
Wenn eine Information nicht eindeutig im Dokument vorhanden ist, gib keine Vermutungen ab. 
Antworte stattdessen höflich, dass diese Information nicht im Dokument enthalten ist.
Wenn du Tabellen, Listen oder Daten findest, gib diese möglichst klar strukturiert zurück.
Füge keine eigenen Interpretationen hinzu, verwende keine externen Quellen und spekuliere nicht.
Sprich den Nutzer direkt an und versuche, komplexe Inhalte verständlich zusammenzufassen.
Sprich sachlich, klar und auf der Sprache, auf der dir Fragen gestellt werden, in der Regel Deutsch.
"""

# Title
st.title("📄 Interaktiver PDF-Chatbot")
logging.info("Streamlit App gestartet")

# Set Slider for model temperature
model_temperature = st.select_slider(
    "Temperatur des Sprachmodells",
    options=temperature_options,
    value=0.5,
    format_func=lambda x: f"{x:.1f}",
    key="chat_model_temperature",
    help="Temperatur des Sprachmodells",
    label_visibility="collapsed",
)
st.write("Temperatur des Sprachmodells:", model_temperature)
logging.info(f"LLM model temperature: {model_temperature}")

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", 
                 temperature=model_temperature,
                # openai_api_key=os.getenv("OPENAI_API_KEY"),
                )
logging.info(f"LLM: {llm.name}")

# Upload PDF file
uploaded_file = st.file_uploader("Lade deine PDF-Datei hier hoch:", type="pdf")
logging.info(f"Uploaded files: {uploaded_file}")

# Check if a file is uploaded
if uploaded_file is not None:
    # Create temp folder and save uploaded pdf in it
    temp_dir = "tmp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success(f"PDF erfolgreich hochgeladen")
    logging.info(f"PDF gespeichert unter: {temp_file_path}")

    # Now use the path with PyPDFLoader
    pdf_loader = PyPDFLoader(temp_file_path)
    docs = pdf_loader.load()
    total_length = sum(len(doc.page_content) for doc in docs)
    chunk_size = min(1000, total_length // 100)

    # Split documents into chunks
    chunk_overlap=200
    if chunk_overlap > chunk_size:
        chunk_overlap = chunk_size / 10
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                              chunk_overlap=chunk_overlap,
                                              separators=["\n\n", "\n"," ", ".", ","])
    chunks = splitter.split_documents(docs)
    # Write data to db to db
    db.add_documents(chunks)
    st.info("Text aus PDF erfolgreich gelesen", icon="📖")

# Define Prompt template
system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt)
human_message_prompt = HumanMessagePromptTemplate.from_template(
    "Kontext:\n{context}\n\nFrage:\n{question}"
)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
logging.info("Prompt-Template erstellt")

# LLMChain mit dem custom Prompt
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    chain_type_kwargs={"prompt": chat_prompt},
    retriever=db.as_retriever(search_kwargs={"k": 2}),
    verbose=True
)
logging.info("Retrieval-QA erstellt")

query = st.text_input("Stelle eine Frage an die PDF:")
logging.info(f"User query: {query}")
response = None
if uploaded_file is None:
    st.warning("Bitte PDF hochladen.")
    logging.warning("No PDF uploaded.")

if not query:
    st.success("Der Chatbot ist bereit! Stelle jetzt deine Fragen.", icon="✨")
    logging.info("Chatbot is ready for questions.")
else:
    try:
        response = qa_chain.run(query)
        logging.info(f"Response: {response}")
        st.write("### Antwort:")
        st.write(response)
        logging.info("Response displayed")
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        st.error("Es gab ein Problem bei der Verarbeitung deiner Anfrage. Bitte versuche es erneut.")
        st.write(response)