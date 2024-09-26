import os.path
import json
import datetime
import pandas as pd
import streamlit as st
from streamlit_gsheets import GSheetsConnection

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

from dotenv import load_dotenv
load_dotenv()

import random


import gspread
import gspread_dataframe as gd
from google.oauth2 import service_account
from google.oauth2.service_account import Credentials


def update_worksheet_entry(user_question="null", bot_answer="null"):
    d = datetime.datetime.now()
    dt = f'{d.strftime("%I")}:{d.strftime("%M")} {d.strftime("%p")} {d.strftime("%B")} {d.strftime("%Y")}'

    # using getlogin() returning username
    login="username"
    try:
        login = os.environ.get('USERNAME')
        print(login)
    except:
        login="null"

    x={"time": [dt], "login": login, "user_question":[user_question] , "bot_answer":[bot_answer], "filenames":[st.session_state.filenames]}
    with open("info.json", "r+") as f:
        file_data= json.load(f)
        file_data["data"].append(x)
        f.seek(0)
        json.dump(file_data, f, indent=4)

    creds_dict = st.secrets["google_service_account"]
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    client = gspread.authorize(creds)
    spreadsheet = client.open_by_key(st.secrets["sheet_id"])
    sheet = spreadsheet.sheet1
    x=[dt, login, user_question , bot_answer, st.session_state.filenames]
    sheet.append_row(x)



def handle_userinput(user_question):
    try:
        response = st.session_state.conversation.invoke({'question': user_question})
        st.session_state.chat_history = response['chat_history']
        user_query="user_query is null"
        bot_answer="bot_answer is null"
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
                user_query=message.content
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
                bot_answer=message.content
    except:
        st.write(bot_template.replace("{{MSG}}", "Please ensure pdfs are uploaded and processed completely before trying."), unsafe_allow_html=True)
    update_worksheet_entry(user_query, bot_answer)


def run_main():
    st.set_page_config(page_title="Perkins&Will", page_icon=":flag-pw:")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    #
    st.session_state.conn = st.connection("gsheets", type=GSheetsConnection)
    st.session_state.data=[]
    #
    filenames=[]

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                if pdf_docs:
                    for uploaded_file in pdf_docs:
                        filenames.append(uploaded_file.name)
                    st.session_state.filenames =  ", ".join(filenames)



if __name__ == '__main__':
    run_main()


    