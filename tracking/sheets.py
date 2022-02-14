#!/usr/bin/env python3
import pathlib

import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials

# define the scope
# SCOPES = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
SCOPES = ['https://www.googleapis.com/auth/drive']
SPREADSHEET_ID = '1qT3gS0brhu2wj59cyeZYP3AywGErROJCqR2wYks6Hcw'

def authenticate():
    credentials_path = pathlib.Path(__file__).parent.joinpath("credentials.json")
    creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, SCOPES)
    client = gspread.authorize(creds)
    return client

def main():
    client = authenticate()
    sheet = client.open_by_key(SPREADSHEET_ID)

    for worksheet in sheet.worksheets():
        df = pd.DataFrame.from_dict(worksheet.get_all_records())
        print(df)

if __name__ == "__main__":
    main()