#!/usr/bin/env python3
import pathlib
import sys

import numpy as np
import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
from rich import print
import git

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config

def authenticate():
    credentials_path = pathlib.Path(__file__).parent.joinpath("credentials.json")
    creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, config.SCOPES)
    client = gspread.authorize(creds)
    return client
    
def _get_sheet():
    client = authenticate()
    sheet = client.open_by_key(config.SPREADSHEET_ID)
    return sheet.get_worksheet_by_id(config.SHEET_ID)

def load_csv():
    worksheet = _get_sheet()
    records = worksheet.get_all_records()
    return pd.DataFrame.from_dict(records)

def _header(worksheet):
    n_headers = len(worksheet.row_values(1))
    return f"{gspread.utils.rowcol_to_a1(1, 1)}:{gspread.utils.rowcol_to_a1(1, n_headers)}"

def _resize_request(df):
    output = { "autoResizeDimensions": 
            {
                "dimensions": 
                    {
                    "sheetId": config.SHEET_ID,
                    "dimension": "COLUMNS",
                    "startIndex": 0,
                    "endIndex": len(df.columns)
                    }
            }
    }
    return output

def _rgba(*args):
    def _to_rgba_dict(rgba):
        return {"red": rgba[0], "green": rgba[1], "blue": rgba[2], "alpha": rgba[3]}
    if len(args) == 1:
        c = args[0]
        return _to_rgba_dict((c, c, c, 1))
    elif len(args) == 2:
        c, a = args[0], args[1]
        return _to_rgba_dict((c, c, c, a))
    elif len(args) == 3:
        return _to_rgba_dict((args[0], args[1], args[2], 1))
    elif len(args) == 4:
        return _to_rgba_dict((args[0], args[1], args[2], args[3]))
    else:
        raise Exception(f"Unexpected number of arguments to sheets._rgba {len(args)}")

def _sheet_dtype(value):
    if type(value) == str:
        return "stringValue", str(value)
    elif type(value) in [np.int32, np.int64, int, float, np.float64, np.float32]:
        return "numberValue", str(value)
    elif type(value) in [bool, np.bool8]:
        return "boolValue", bool(value)
    else:
        return "stringValue", str(value)

def _value_batch_update(df, client, spreadsheet, worksheet, order=[], col_order=[]):
    def _val(val):
        dtype, value = _sheet_dtype(val)
        return {"userEnteredValue": {dtype: value}}

    def _row_values(row):
        vals = []
        for col in col_order:
            vals.append(_val(row[col]))
        return vals

    if order is not None and order != []:
        df = df.sort_values(by=order, axis=0, ascending=False, na_position="last", ignore_index=False)
    
    rows = [{"values": [_val(col) for col in col_order]}]
    for idx in df.index:
        row = {"values": _row_values(df.loc[idx])}
        rows.append(row)

    output = {
        "updateCells": {
            "rows": rows,
            "range": {
                "sheetId": config.SHEET_ID,
                "startRowIndex": 0,
                "endRowIndex": len(df) + 1, # +1 due to header/column names row also added
                "startColumnIndex": 0,
                "endColumnIndex": len(df.columns)
            },
            "fields": "userEnteredValue"
        }
    }
    return output

def _delete_formatting(banded_range_id):
    return {"deleteBanding": {"bandedRangeId": banded_range_id}}

def _formatting(df, command="addBanding"):
    output = {
        command: 
        {
            'bandedRange':
            {
                'bandedRangeId': 1,
                'range': 
                {
                    'sheetId': config.SHEET_ID,
                    'startRowIndex': 0,
                    'endRowIndex': len(df) + 1,
                    'startColumnIndex': 0,
                    'endColumnIndex': len(df.columns),
                },
                'rowProperties': {
                    'headerColor': config.HEADER_BACKGROUND_COLOR,
                    'firstBandColor': config.ODD_ROW_BACKGROUND_COLOR,
                    'secondBandColor': config.EVEN_ROW_BACKGROUND_COLOR
                },
            }
        }
    }
    return output

def _get_brange_ids(spreadsheet):
    banded_range_ids = []
    spreadsheet_get_info = spreadsheet._spreadsheets_get()
    if "sheets" in spreadsheet_get_info.keys():
        for sheet in spreadsheet_get_info["sheets"]:
            if "properties" in sheet.keys() and "sheetId" in sheet["properties"].keys():
                if sheet["properties"]["sheetId"] == config.SHEET_ID:
                    if "bandedRanges" in sheet.keys():
                        for brange in sheet["bandedRanges"]:
                            if "bandedRangeId" in brange.keys():
                                banded_range_ids.append(brange["bandedRangeId"])
    return banded_range_ids

def _set_requests(df, client, spreadsheet, worksheet, order=[], col_order=[]):
    requests = [
        _value_batch_update(df, client, spreadsheet, worksheet, order=order, col_order=col_order),
        _resize_request(df),
    ]
    requests += [_delete_formatting(BRANGE_ID) for BRANGE_ID in _get_brange_ids(spreadsheet)]
    requests.append(_formatting(df, command="addBanding"))
    return requests

def _batch_update(df, order=[], col_order=[]):
    client = authenticate()
    spreadsheet = client.open_by_key(config.SPREADSHEET_ID)
    worksheet =  spreadsheet.get_worksheet_by_id(config.SHEET_ID)
    body = {"requests": _set_requests(df, client, spreadsheet, worksheet, order=order, col_order=col_order)}
    res = spreadsheet.batch_update(body)

def save_csv(df, order=[], col_order=[]):
    cleaned_df = df.fillna("")
    if col_order == [] or col_order is None:
        col_order = cleaned_df.columns
    _batch_update(cleaned_df, order=order, col_order=col_order)