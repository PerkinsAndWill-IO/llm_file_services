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
# create_vector_db()

custom_prompt_template = """
Use the following pieces of information to answer the user's question. 
If you do not know the answer, please just say that you do not know the answer. Do not try to make up the answer.
Context: {context}
Question: {question}
Only returns the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vector stores
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm = OpenAI(),
        chain_type = "stuff",
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt':prompt}
    )
    return qa_chain

def qa_bot():
    # embeddings = HuggingFaceEmbeddings(model='sentence-transformers/all-MiniLM-L6-v', model_kwargs={'device': 'cpu'}) 
    # embeddings = model.encode(text_chunks)
    embeddings = OpenAIEmbeddings()
    db= FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = OpenAI()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain.invoke(llm, qa_prompt, db) # qa_chain
    return qa


def run_qa_bot():
    qa_result = qa_bot() # qa
    while True:
        query = input("please enter your query $")
        if query == "q" or query == "Q":
            break
        else:
            response = qa_result({'query': query}) # qa_chain with inputs query
            print(response)

# run_qa_bot()


embeddings = OpenAIEmbeddings()
db= FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
llm = OpenAI()
qa_prompt = set_custom_prompt()

qa_chain = RetrievalQA.from_chain_type(
        llm = OpenAI(),
        chain_type = "stuff",
        retriever=db.as_retriever(search_kwargs={'k': 10}),
        return_source_documents=True,
        chain_type_kwargs={'prompt':qa_prompt}
    )

# qa_result = qa_bot()
response =  qa_chain.invoke("summarize data") # qa_chain

# print(response)
print(f"\nquery:\n{response['query']}")
print(f"\nresult:\n{response['result']}")


docs = response['source_documents']
for i, doc in enumerate(docs):
    print(f"\nsource {i}\n{doc}")

