import pandas as pd 
import numpy as np
from dotenv import load_dotenv
import yfinance as yf
from scipy.io import savemat, loadmat
import os
import csv
from tqdm import tqdm

def downloadAllData(tickers):
    industrys = {}
    data = {}
    len_max = 0
    sheet_len_max = 0
    for ticker_name in tqdm(tickers):
        ticker = yf.Ticker(ticker_name)
        if 'industry' not in ticker.info:
            continue

        industry = ticker.info['industry']
        high_values, low_values, volume_values, balance_sheet, flow_sheet = downloadTickerData(ticker)
        if balance_sheet is False or flow_sheet is False:
            continue

        if len(high_values) <= 0:
            continue

        if industry in industrys:
            industrys[industry].append([high_values, low_values, volume_values, balance_sheet, flow_sheet])
        else:
            industrys[industry] = [[high_values, low_values, volume_values, balance_sheet, flow_sheet]]

        len_max = max(len(high_values),len_max)
        len_max = max(len(low_values),len_max)
        len_max = max(len(high_values),len_max)

        sheet_len_max = max(len(balance_sheet),sheet_len_max)
        sheet_len_max = max(len(flow_sheet),sheet_len_max)
    
    for key in tqdm(industrys):
        high_values_av = None
        low_values_av = None
        volume_values_av = None
        for i in range(len(industrys[key])):
            high_values, low_values, volume_values, balance_sheet, flow_sheet = industrys[key][i]

            high_values = np.append(np.zeros(len_max-len(high_values)),high_values)
            low_values = np.append(np.zeros(len_max-len(low_values)),low_values)
            volume_values = np.append(np.zeros(len_max-len(volume_values)),volume_values)

            balance_sheet = np.append(np.zeros((sheet_len_max-len(balance_sheet),balance_sheet.shape[1])),
                                      balance_sheet)
            flow_sheet = np.append(np.zeros((sheet_len_max-len(flow_sheet),flow_sheet.shape[1])),
                                      flow_sheet)

            if high_values_av is None:
                high_values_av = high_values * volume_values
                low_values_av = low_values * volume_values
                volume_values_av = volume_values
            else:
                high_values_av += high_values * volume_values
                low_values_av += high_values * volume_values
                volume_values_av += volume_values
            
            high_values /= np.max(high_values)
            low_values /= np.max(high_values)
            volume_values /= np.max(volume_values)

            industrys[key][i] = [
                high_values, low_values, volume_values, balance_sheet, flow_sheet
            ]
        
        high_values_av /= volume_values_av
        low_values_av /= volume_values_av

        high_values_av /= np.max(high_values_av)
        low_values /= np.max(high_values_av)
        volume_values_av /= np.max(volume_values_av)

        data[key] = {
            'tickrs': industrys[key],
            'av': [
                high_values_av,
                low_values_av,
                volume_values_av
            ]
        }
    
    return data

def replace_missing(sheet, freq='QE'):
    dates = sheet.columns
    start_date = dates.min()
    end_date = dates.max()
    if start_date == 'NaT' or end_date == 'NaT':
        return False
    expected_dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    missing_dates = expected_dates.difference(dates)

    if len(missing_dates) == 0:
        return sheet

    missing_data_df = pd.DataFrame(index=missing_dates, columns=sheet.index)

    for column in sheet.index:
        column_avg = sheet.loc[column].mean()
        missing_data_df.loc[:, column] = column_avg
    missing_data_df = missing_data_df.T
    combined_df = sheet.join(missing_data_df)

    combined_df = combined_df.reindex(sorted(combined_df.columns, key=pd.to_datetime), axis=1)

    return combined_df


def normilzeSheet(sheet_main):

    sheet_main = replace_missing(sheet_main)
    if sheet_main is False:
        return False

    sheet = sheet_main.to_numpy().astype(np.float32)

    col_mean = np.nanmean(sheet, axis=0)
    inds = np.where(np.isnan(sheet))
    sheet[inds] = np.take(col_mean, inds[1])
    sheet = (sheet - np.min(sheet)) / (np.max(sheet) - np.min(sheet))

    return sheet


def downloadTickerData(ticker: yf.Ticker):
    stock_data = ticker.history(period="max")
    
    #dates = ticker['']
    high_values = stock_data['High'].to_numpy()
    low_values = stock_data['Low'].to_numpy()
    volume_values = stock_data['Volume'].to_numpy()

    balance_sheet = normilzeSheet(ticker.quarterly_balance_sheet)
    
    flow_sheet = normilzeSheet(ticker.quarterly_cash_flow)

    return high_values, low_values, volume_values, balance_sheet, flow_sheet

def getData(tickers, file):
    if os.path.isfile(file):
        return loadmat(file)
    else:
        data = downloadAllData(tickers)
        savemat(file,data)
        return data

def get_sp500_list():
    is_first = True
    out = []
    with open('constituents.csv', mode ='r')as file:
        csvFile = csv.reader(file)
        for line in csvFile:
            if is_first:
                is_first = False
                continue
            out.append(line[0])
    return out

print(getData(get_sp500_list(), 'S&P.mat'))