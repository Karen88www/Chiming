from google.oauth2.service_account import Credentials
import gspread
from gspread_dataframe import set_with_dataframe, get_as_dataframe
import pandas as pd
import warnings

def sheetAuth(authFilePath):
    scopes = ['https://spreadsheets.google.com/feeds']
    credentials = Credentials.from_service_account_file(authFilePath, scopes=scopes)
    gc = gspread.authorize(credentials)
    return gc


def sheetUpdate(authFilePath, fileId, sheet, df):
    warnings.warn("will be deprecated, use update_df instead", DeprecationWarning)
    gc = sheetAuth(authFilePath)
    file = gc.open_by_key(fileId)
    file.values_clear(f"{sheet}!A:Z")
    worksheet = file.worksheet(sheet)
    set_with_dataframe(worksheet, df)


def disConUpdate(authFilePath, fileId, sheet, df, toCols, **startat):
    gc = sheetAuth(authFilePath) 
    file = gc.open_by_key(fileId)
    worksheet = file.worksheet(sheet)
    if len(startat) == 0:
        startat = 1
    else:
        startat = startat['startat']
    for i,j in enumerate(toCols):
        colName = df.columns[i]
        worksheet.update(f'{j}{startat}:{j}{startat+len(df)}', [[df.columns.tolist()[i]]]+[[k] for k in df[colName].to_list()])


def sheetAppend(authFilePath, fileId, sheet, df, startat, header=True):
    warnings.warn("will be deprecated, use update_df instead", DeprecationWarning)
    gc = sheetAuth(authFilePath)
    file = gc.open_by_key(fileId)
    worksheet = file.worksheet(sheet)
    if len(startat) == 0:
        startat = "A1"
    data = df if type(df) == list else df.to_dict(orient='records')
    data = [[i[j] for j in i] for i in data]
    if header:
        header = df.columns.tolist()
        data = [header] + data
    worksheet.update(startat, data)


def cellUpdate(authFilePath, fileId, sheet, fromCell, func, toCell):
    gc = sheetAuth(authFilePath)
    file = gc.open_by_key(fileId)
    worksheet = file.worksheet(sheet)
    target = worksheet.acell(fromCell).value
    if fromCell == None or func == None:
        return target
    value = func(target)
    if toCell == None:
        return value
    else:
        worksheet.update(toCell, value)


def insert_sheet_df(authFilePath, fileId, sheet, df):
    gc = sheetAuth(authFilePath)
    file = gc.open_by_key(fileId)
    rows = len(df)+10
    cols = len(df.columns)+2
    try:
        file.add_worksheet(title=f"{sheet}", rows=rows, cols=cols)
    except:
        file.add_worksheet(title=f"{sheet}_1", rows=rows, cols=cols)
        sheet = f"{sheet}_1"
    worksheet = file.worksheet(sheet)
    set_with_dataframe(worksheet, df)


def insert_sheet(authFilePath, fileId, sheet):
    gc = sheetAuth(authFilePath)
    file = gc.open_by_key(fileId)
    rows = 100
    cols = 26
    try:
        file.add_worksheet(title=f"{sheet}", rows=rows, cols=cols)
    except:
        file.add_worksheet(title=f"{sheet}_1", rows=rows, cols=cols)
    return file.worksheet(sheet)


def update_acell(authFilePath, fileId, sheet, targetCell, value):
    gc = sheetAuth(authFilePath)
    file = gc.open_by_key(fileId)
    worksheet = file.worksheet(sheet)
    worksheet.update(targetCell, value)


def get_cell(authFilePath, fileId, sheet, targetCell):
    gc = sheetAuth(authFilePath)
    file = gc.open_by_key(fileId)
    worksheet = file.worksheet(sheet)
    return worksheet.acell(targetCell).value


def worksheet(authFilePath, fileId, sheet):
    gc = sheetAuth(authFilePath)
    file = gc.open_by_key(fileId)
    worksheet = file.worksheet(sheet)
    return worksheet

def update_df(authFilePath, fileId, sheet, update_type, ranges, content, header=True):
    gc = sheetAuth(authFilePath)
    file = gc.open_by_key(fileId)
    worksheet = file.worksheet(sheet)
    if header:
        ranges = ranges if ranges != '' else 'A:Z'
    else:
        ranges = ranges if range != '' else 'A2:Z'
    if isinstance(content, pd.DataFrame):
        if update_type == 'update':
            file.values_clear(f"{sheet}!{ranges}")
            set_with_dataframe(worksheet, content)
            return
        else:
            content = content.to_dict(orient='records')
            content = [[i[j] for j in i] for i in content]
    if len(content) > 1:
        content = content if header else content[1:]
    
    worksheet.append_rows(content)


# def update_df(authFilePath, fileId, sheet, update_type, ranges, content, header=True):
#     gc = sheetAuth(authFilePath)
#     file = gc.open_by_key(fileId)
#     worksheet = file.worksheet(sheet)
#     if isinstance(content, pd.DataFrame):
#         content = content.to_dict(orient='records')
#         content = [[i[j] for j in i] for i in content]
#     if len(content) > 1:
#         content = content if header else content[1:]
#     if update_type == 'append': 
#         worksheet.append_rows(content)
#     elif update_type == 'update':
#         ranges = ranges if ranges != '' else 'A:Z'
#         file.values_clear(f"{sheet}!{ranges}")
#         set_with_dataframe(worksheet, content)
        

def get_last_row(sheet):
    values = sheet.get_all_values()
    last_row_number = None
    for i, row in enumerate(reversed(values), start=1):
        if any(cell for cell in row if cell):
            last_row_number = len(values) - i + 1
            break
    return last_row_number


def sheet_clear_values(authFilePath, fileId, sheet, ranges):
    gc = sheetAuth(authFilePath)
    file = gc.open_by_key(fileId)
    file.values_clear(f"{sheet}!{ranges}")
