#!/usr/bin/env python3
import pathlib
import sys
from typing import Iterable, Mapping, Tuple, Union

import gspread
from gspread.utils import a1_range_to_grid_range, rowcol_to_a1
import pandas as pd
import numpy as np
from oauth2client.service_account import ServiceAccountCredentials
from rich import print
import git

REPO_DIR = pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)
sys.path.insert(0, str(REPO_DIR))
import config
from logger import ILogger, Logger

class SheetClient:
    def __init__(self, logger: ILogger = Logger()) -> None:
        self.logger = logger
        self._client = self._authenticate()
        self.logger.log(f"Opening spreadsheet by key: {config.SPREADSHEET_ID}")
        self._spreadsheet = self._client.open_by_key(config.SPREADSHEET_ID)
        self.logger.log(f"Opening sheet by SHEET ID {config.SHEET_ID}")
        self._sheet = self._spreadsheet.get_worksheet_by_id(config.SHEET_ID)

    def _authenticate(self) -> gspread.Client:
        credentials_path = REPO_DIR.joinpath("credentials.json")
        self.logger.log(f"Reading Google Sheets credentials from: {credentials_path}")
        creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, config.SCOPES)
        client = gspread.authorize(creds)
        return client

    def get_column_names(self) -> Iterable[str]:
        rows = self._sheet.get_values()
        if len(rows) > 0:
            return rows[0]
        return []

    def add_columns(self, columns: Iterable[str]) -> None:
        cols = self.get_column_names()
        if len(cols) + len(columns) >= self._sheet.col_count:
            num_new_cols = len(columns)
            self._sheet.add_cols(num_new_cols)

        first_new_col_idx = len(cols) + 1
        last_new_col_idx = first_new_col_idx + len(columns)
        data = [{
            "range": f"{rowcol_to_a1(1, first_new_col_idx)}:{rowcol_to_a1(1, last_new_col_idx)}",
            "values": [columns]
        }]
        self._sheet.batch_update(data)
        
    def _order_row_values(self, flattened_new_row: Mapping[str, any]) -> list:
        existing_columns = self.get_column_names()
        output = [flattened_new_row[col] for col in existing_columns]
        return output

    def add_row(self, row: Mapping[str, any]) -> None:
        if len(row) == 0:
            raise Exception(f"SheetClient could not write empty row: {row}")
        
        # Flatten nested dicts in the row
        flattened_row = pd.json_normalize(row).to_dict(orient="records")[0]

        existing_columns = self.get_column_names()
        cols_in_new_row = flattened_row.keys()
        
        # Ensure that the new row has keys/columns for all columns that already exist in the sheet
        missing_columns = list(set(existing_columns) - set(cols_in_new_row)) # Initialize these columns in the new row with missing value None/0/np.nan 
        for col in missing_columns:
            flattened_row[col] = None

        # Add any new columns, e.g. keys/columns in the new row that are not already present in the sheet, to the sheet.
        new_columns = list(set(cols_in_new_row) - set(existing_columns)) # add these columns to the sheet
        if len(new_columns) > 0:
            self.add_columns(new_columns)

        # Ensure that the values in new row is ordered correctly, according to columns and append it to the worksheet
        row_values_to_append = self._order_row_values(flattened_row)
        for idx, value in enumerate(row_values_to_append):
            if type(value) not in [float, int, str] and value is not None:
                row_values_to_append[idx] = str(value)
        self._sheet.append_row(row_values_to_append, value_input_option="RAW", insert_data_option="INSERT_ROWS")

    def sort(self, order_by: Iterable[str] = []) -> None:
        cols = self.get_column_names()
        _order_by = [col for col in order_by if col in cols]
        order_by = [(cols.index(col) + 1, "des") for col in _order_by]
        rows = self._sheet.get_values()
        value_range = f"A2:{rowcol_to_a1(len(rows), len(cols))}" # cells not containing header row
        self._sheet.sort(*order_by, range=value_range)

    def _move_request(self, from_col_idx: int, to_col_idx: int) -> dict:
        json_request = {
                    "moveDimension": {
                        "source": {
                            "sheetId": config.SHEET_ID,
                            "dimension": "COLUMNS",
                            "startIndex": from_col_idx,
                            "endIndex": from_col_idx+1
                        },
                        "destinationIndex": to_col_idx
                    }
                }
        return json_request

    def set_col_order(self, col_order: Iterable[str] = []) -> None:
        columns = self.get_column_names()
        cols_to_move = []
        for index, column in enumerate(col_order):
            frm = columns.index(column)
            to = index
            if frm != to:
                r = self._move_request(frm, index)
                
                del columns[frm]
                columns.insert(index, column)

                cols_to_move.append(r)
        
        if len(cols_to_move) > 0:
            body = {
                "requests": cols_to_move
            }
            self._spreadsheet.batch_update(body)

    def _get_sheet_metadata(self):
        spreadsheet_metadata = self._spreadsheet.fetch_sheet_metadata()
        
        if "sheets" in spreadsheet_metadata.keys():
            for sheet in spreadsheet_metadata["sheets"]:
                if "properties" in sheet:
                    properties = sheet["properties"]
                    if "sheetId" in properties:
                        if properties["sheetId"] == config.SHEET_ID:
                            return sheet
        return {}

    def _batchupdate_bandedrange(self, command: str = "addBanding", banded_range_id: int = None):
        if command not in ["addBanding", "updateBanding"]:
            raise Exception(f"Command argument {command} is invalid, must be 'addBanding' or 'updateBanding'")
        
        rows = self._sheet.get_values()
        cols = self.get_column_names()
        banded_range = {
            'range': {
                'sheetId': config.SHEET_ID,
                'startRowIndex': 0,
                'endRowIndex': len(rows),
                'startColumnIndex': 0,
                'endColumnIndex': len(cols),
            },
            'rowProperties': {
                'headerColor': config.HEADER_BACKGROUND_COLOR,
                'firstBandColor': config.ODD_ROW_BACKGROUND_COLOR,
                'secondBandColor': config.EVEN_ROW_BACKGROUND_COLOR
            },
        }
        if banded_range_id is not None:
            banded_range["bandedRangeId"] = banded_range_id
        
        cmd = {command: {"bandedRange": banded_range}}
        if command == "updateBanding": 
            cmd[command]["fields"] = "range"

        body = {"requests": [cmd]}
        self._spreadsheet.batch_update(body)

    def set_alternating_colors(self):
        metadata = self._get_sheet_metadata()
        
        banded_range_id = None
        command = "addBanding"
        if "bandedRanges" in metadata:
            for bandedRange in metadata["bandedRanges"]:
                if "bandedRangeId" in bandedRange:
                    banded_range_id = bandedRange["bandedRangeId"]
                    command = "updateBanding"

        self._batchupdate_bandedrange(command=command, banded_range_id=banded_range_id)

    def autosize_columns(self):
        cols = self.get_column_names()
        resize_request = {
            "autoResizeDimensions": {
                "dimensions": {
                    "sheetId": config.SHEET_ID,
                    "dimension": "COLUMNS",
                    "startIndex": 0,
                    "endIndex": len(cols)
                }
            }
        }
        body = {
            "requests": [resize_request]
        }
        self._spreadsheet.batch_update(body)
    
    def format(self, order_by: Iterable[Tuple[str, str]] = [], col_order: Iterable[str] = []) -> None:
        self.sort(order_by=order_by)
        self.set_col_order(col_order=col_order)
        self.set_alternating_colors()
        self.autosize_columns()

def main():
    from datetime import datetime
    client = SheetClient()
    data = {
        "created_at": datetime.now().strftime(config.DATETIME_FORMAT),
        "a_new_column": {
            "nested": "value"
        }
    }
    client.add_row(data)
    col_order = ["created_at"]
    
    client.format(order_by=[("created_at", "des")], col_order=col_order)

if __name__ == "__main__":
    main()