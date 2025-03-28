# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas_datareader.data as web
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
#yf.pdr_override()
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
import itertools

import sys
#株価取得------------------------------------------------------------------------------------------
def data_get(stock,start,end):
    #df = web.get_data_yahoo(stock,  start, end)
    df = yf.download(stock, start=start, end=end)
    df.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
    df["SMA60"] = df["Close"].rolling(window=60).mean()
    df["Deviation"] = (df["Close"] / df["SMA60"])
    df = df.drop(["Open","High","Low","Volume","SMA60"], axis=1, inplace=False)
    df.insert(0, 'Date', df.index)
    df = df.reset_index(drop=True)
    #index, Date, Close, Deviation
    return df

#SP波動法------------------------------------------------------------------------------------------
def wave_method(df,SP):
    nadir_arr = [] #天底
    #初期上昇の場合
    if df['Close'][0]<=df['Close'][1]:
        nadir_arr.append([df['Date'][0], df['Close'][0],0])
        bot = [df['Date'][0], df['Close'][0]]
        mode = 0
    #初期下降の場合
    else :
        nadir_arr.append([df['Date'][0], df['Close'][0],1])
        top = [df['Date'][0], df['Close'][0]]
        mode = 1

    for index, row in df.loc[:,['Date','Close']].iterrows():
        row.iloc[0] = pd.Timestamp(row.iloc[0])
        '''
        index : 0~株価の終わり日まで自然数
        row : リスト型、[0]にはpd.timestamp型の日付、[1]にはint型の株価終値
        '''
        #底を使って天井候補の条件を返す関数
        def SP_a():
            return bot[1] * (200+SP) / (200-SP)
        #天井を使って底候補の条件を返す関数
        def SP_b():
            return top[1] * (200-SP) / (200+SP)
        #mode0と1で分けるとうまくコードが書けた
        if mode == 0 :
            #SP天井を超えた
            if row.iloc[1] >= SP_a():
                top = row #最新の天井
                nadir_arr.append([row.iloc[0],row.iloc[1],1]) #天井の決定、配列にメモ
                mode = 1
            #底を超える
            elif row.iloc[1] < bot[1]:
                bot = row #最新の底
                #新たなの底を更新
                nadir_arr.pop() #配列にメモしていた底を消す
                nadir_arr.append([row.iloc[0],row.iloc[1],0]) #新たな底をメモ
        elif mode == 1 :
            #SP底を超えた
            if row.iloc[1] < SP_b():
                bot = row #最新の底
                nadir_arr.append([row.iloc[0],row.iloc[1],0]) #底の決定、配列にメモ
                mode = 0
            #天井を超える
            elif row.iloc[1] > top[1]:
                top = row  #最新の天井
                nadir_arr.pop()
                nadir_arr.append([row.iloc[0],row.iloc[1],1])
    return nadir_arr


#上昇度------------------------------------------------------------------------------------------
def increasefunc(df,SP):
    nadir_arr = wave_method(df,SP)
    increase_rate_arr = []
    display_nikkei_rise = [] #上昇を表示した日経データ表示用
    display_nikkei_fall = [] #下降を表示した日経データ表示用
    for i in range(len(nadir_arr)):
        #天底最初
        if i == 0:
            section_df = df[df['Date'] <= nadir_arr[i][0]].reset_index(drop=True)
            if nadir_arr[i][2] == 1:
                increase_rate = 1
                display_nikkei_rise.append([section_df['Date'], section_df['Close']])
            else:
                increase_rate = 0
                display_nikkei_fall.append([section_df['Date'], section_df['Close']])

            for j in range(len(section_df)-1):
                increase_rate_arr.append(increase_rate)

        #天底最後
        elif i == len(nadir_arr)-1:
            section_df = df[(df['Date'] >= nadir_arr[i-1][0]) & (df['Date'] <= nadir_arr[i][0])].reset_index(drop=True)
            if nadir_arr[i][2] == 1:
                increase_rate = 1
                display_nikkei_rise.append([section_df['Date'], section_df['Close']])

            else:
                increase_rate = 0
                display_nikkei_fall.append([section_df['Date'], section_df['Close']])

            for j in range(len(section_df)-1):
                increase_rate_arr.append(increase_rate)


            section_df = df[df['Date'] >= nadir_arr[i][0]].reset_index(drop=True)
            if nadir_arr[i][2] == 1:
                increase_rate = 0
                display_nikkei_fall.append([section_df['Date'], section_df['Close']])
            else:
                increase_rate = 1
                display_nikkei_rise.append([section_df['Date'], section_df['Close']])

            for j in range(len(section_df)):
                increase_rate_arr.append(increase_rate)

        #天底
        else:
            section_df = df[(df['Date'] >= nadir_arr[i-1][0]) & (df['Date'] <= nadir_arr[i][0])].reset_index(drop=True)
            if nadir_arr[i][2] == 1:
                increase_rate = 1
                display_nikkei_rise.append([section_df['Date'], section_df['Close']])
            else:
                increase_rate = 0
                display_nikkei_fall.append([section_df['Date'], section_df['Close']])

            for j in range(len(section_df)-1):
                    increase_rate_arr.append(increase_rate)
        
    df['Increase Rate'] = increase_rate_arr

    return nadir_arr