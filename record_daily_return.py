import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
from datetime import datetime
import yfinance as yfin
from datetime import date, timedelta
yfin.pdr_override()

symbols = ['AMZN','MSFT', 'QQQ']

def fetch_data(symbols, start_str, end_str):
    result = []
    # set date string format
    date_format = "%Y-%m-%d"
    # get start time and end time
    start_date = datetime.strptime(start_str, date_format)
    end_date = datetime.strptime(end_str, date_format)
    for symbol in symbols:
        print(f'Now fetching {symbol} data from Yahoo...')
        # get data from Yahoo finance
        data = pdr.get_data_yahoo(symbol, start = start_date, end = end_date)
        print(f'{symbol} data fetched successfully.')
        # add data to result
        result.append(data)
    return result

def get_adj_close(df):
  return df[['Adj Close']]

def get_merged_df(result):
  # only keep 'Adj Close'
  result_ac = [get_adj_close(x) for x in result]
  # change columns name
  for i in range(len(result_ac)):
    result_ac[i].columns =[symbols[i]]
  # merge into one dataframe
  merged_df = pd.concat(result_ac, axis=1)
  # print(f'original shape\t\t{merged_df.shape}')
  # drop NA rows, in case some stock did not trade on some day
  merged_df.dropna(inplace=True)
  # print(f'shape after drop N.A.\t{merged_df.shape}')
  return merged_df


# run before market open every day
def run_before_open():
  model_MSFT = tf.keras.models.load_model('./model_MSFT')
  model_AMZN = tf.keras.models.load_model('./model_AMZN')
  # input last 3 days data
  today = date.today()
  ten_days_before = today - timedelta(days=10)
  current_data = fetch_data(symbols, ten_days_before.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d"))
  merged_df = get_merged_df(current_data)
  # print(merged_df.tail())
  change_df = merged_df.copy()
  change_df['AMZN'] = merged_df['AMZN'] / merged_df['AMZN'].shift(1)
  change_df['MSFT'] = merged_df['MSFT'] / merged_df['MSFT'].shift(1)
  change_df['QQQ'] = merged_df['QQQ'] / merged_df['QQQ'].shift(1)
  change_df.dropna(inplace=True)
  change_df['AMZN_DIFF_QQQ'] = change_df['AMZN'] - change_df['QQQ']
  change_df['MSFT_DIFF_QQQ'] = change_df['MSFT'] - change_df['QQQ']
  # print(change_df.tail())
  AMZN_DIFF_QQQ = change_df['AMZN_DIFF_QQQ'].values.reshape(-1,1)
  MSFT_DIFF_QQQ = change_df['MSFT_DIFF_QQQ'].values.reshape(-1,1)
  AMZN_DIFF_QQQ = AMZN_DIFF_QQQ[-3:].reshape(1,3,1)
  MSFT_DIFF_QQQ = MSFT_DIFF_QQQ[-3:].reshape(1,3,1)
  X_AMZN = tf.convert_to_tensor(AMZN_DIFF_QQQ, dtype=tf.float32)
  X_MSFT = tf.convert_to_tensor(MSFT_DIFF_QQQ, dtype=tf.float32)
  mean_diff = 0.0011170707
  pred_AMZN = model_AMZN.predict(X_AMZN).reshape(-1)
  # print(f'pred_AMZN:\n{pred_AMZN}')
  pred_MSFT = model_MSFT.predict(X_MSFT).reshape(-1)
  print(f'pred_MSFT:\n{pred_MSFT}')
  print(f'pred_AMZN - pred_MSFT > mean_diff is: {pred_AMZN - pred_MSFT > mean_diff}')

  record = pd.DataFrame({'date': today, 'pred_AMZN': pred_AMZN, 'pred_MSFT': pred_MSFT\
                       ,'BUY AMZN/SHORT MSFT': pred_AMZN - pred_MSFT > mean_diff, 'BUY MSFT/SHORT AMZN': pred_AMZN - pred_MSFT < mean_diff\
                       ,'return': 0})
  # show record
  print(record)
  # save record
  record.to_csv('./before_open.csv',index=False)
  
def run_after_close():
  record = pd.read_csv('./before_open.csv', index_col=False)
  end = date.today() + timedelta(days=1)
  start = date.today() - timedelta(days=1)
  current_data = fetch_data(symbols, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
  merged_df = get_merged_df(current_data)
  print(merged_df)
  AMZN = merged_df['AMZN'].values
  MSFT = merged_df['MSFT'].values
  if record['BUY AMZN/SHORT MSFT'].values[-1]:
    return_rate = AMZN[-1]/AMZN[-2] - MSFT[-1]/MSFT[-2]
  else:
    return_rate = -AMZN[-1]/AMZN[-2] + MSFT[-1]/MSFT[-2]
  print(return_rate)
  # update today's actual return
  record['return'] = [return_rate]
  print(record)
  # append to record file
  record.to_csv('./record.csv', mode='a', header= False, index=False)


run_before_open()

run_after_close()