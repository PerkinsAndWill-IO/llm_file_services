import os, sys
import os.path
import json
import datetime

from PyPDF2 import PdfReader

from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, OpenAI
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA

from dotenv import load_dotenv
load_dotenv()

def create_vector_db():
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(text_chunks, embeddings)
    db.save_local(DB_FAISS_PATH)

DATA_PATH = "files"
DB_FAISS_PATH = "vectorstore"
create_vector_db()