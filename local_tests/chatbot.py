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


