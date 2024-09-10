import os.path
import gspread
from google.oauth2.service_account import Credentials

def get_last_row(ns_worksheet):
    row_count=0
    while True and row_count<10:
        row_count +=1 
        try:
            v = ns_worksheet.row_values(row_count)[0]
            if v=='':
                break
        except:
            break
    return row_count

def update_worksheet_entry(ns_worksheet, user_question="-", bot_answer="x"):
    row_count = get_last_row(ns_worksheet)
    ns_worksheet.update_cell(row_count, 1, ns_worksheet)
    ns_worksheet.update_cell(row_count, 2, user_question)
    ns_worksheet.update_cell(row_count, 3, bot_answer)

def initialize_gspread():
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds  = Credentials.from_service_account_file("credentials.json", scopes = scopes)
    client = gspread.authorize(creds)

    sheet_id = "1XK9lPuT4rjIBrymeCn5ikk3TBSEMaqBarXfQn5SiTSA"
    workbook = client.open_by_key(sheet_id)
    worksheet = workbook.get_worksheet(0)

    user_question="hello"
    bot_answer = "world"
    update_worksheet_entry(worksheet, user_question, bot_answer)

if __name__ == "__main__":
    initialize_gspread()