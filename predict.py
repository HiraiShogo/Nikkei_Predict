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
import itertools

import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Activation, BatchNormalization
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from callbacks import EarlyStopping

import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())

import function as fc

#株価取得---------------------------------------------------------------------------------------
'''
start = datetime(2001, 1, 1)
end = datetime.today()
stock = '^N225'
df = fc.data_get(stock,start,end)
df.to_csv('Stock_Price.csv')


df = pd.read_csv('Stock_Price.csv')
df['Date'] = pd.to_datetime(df['Date'])
'''


df = pd.read_csv('Stock_Price.csv')
df['Date'] = pd.to_datetime(df['Date'])
date = pd.to_datetime(df["Date"])   

#SP波動法のパラメータ-------------------------------------------------------
# 日経平均の日足なら1~2が目安
SP = 1


#上昇度を求める-----------------------------------------------------------------------
nadir_arr = fc.increasefunc(df, SP)

#機械学習----------------------------------------------------------------------------------
# class名 RNN に意味はない。詳解ディープラーニングのプログラムに書いてあったのをそのまま使っているだけ。
class RNN(Model):
    def __init__(self, hidden_dim):
        super().__init__()
        '''
        1層:LSTM(60ニューロン),バッチ正規化,活性化,ドロップアウト
        2層:全結合層(1ニューロン)
        '''
        self.l1 = LSTM(hidden_dim, 
                       recurrent_activation='sigmoid',
                       kernel_initializer='glorot_normal',
                       recurrent_initializer='orthogonal')
        self.b1 = BatchNormalization()
        self.a1 = Activation('tanh')
        self.d1 = Dropout(0.5)
        self.l2 = Dense(1, activation='linear', kernel_initializer='glorot_normal')

        self.ls =[self.l1, self.b1, self.a1, self.d1, 
                  self.l2]

    def call(self, x):

        for layer in self.ls:
            x = layer(x)

        return x


def normalization(dev_60): # 正規化
    #入力データを正規化する事で、モデル内部のバランスを偏らないようにする
    dev_arr = []
    min = np.min(dev_60)
    max = np.max(dev_60)
    for dev in dev_60:
        dev_arr.append((dev-min)/(max-min))
    return dev_arr

'''
1. データの準備
'''
df_inout = df[59:len(df['Close']) - 1].reset_index(drop=True).loc[:,['Date','Close','Deviation','Increase Rate']]

length_of_sequences = len(df_inout['Close'])
maxlen = 60
test_date = '2010-01-04'
x = [] #入力データ、乖離率(60日分の乖離率)
t = [] #教師データ、上昇度(1日分の上昇度)

for i in range(length_of_sequences - maxlen + 1): #5650 - 60 + 1
    x.append(normalization(df_inout['Deviation'][i:i+maxlen]))
    t.append(df_inout['Increase Rate'][i+maxlen-1]) #教師


x = np.array(x).reshape(-1, maxlen, 1)
t = np.array(t).reshape(-1, 1)

num = len(df_inout) - df_inout[df_inout['Date']== pd.to_datetime(test_date)].index.values[0] + 1 #日付で指定した以降の日が検証期間になる
x_train, x_val, t_train, t_val = \
    train_test_split(x, t, test_size=num, shuffle=False)


'''
2. モデルの構築
'''
model = RNN(60)

'''
3. モデルの学習
'''
criterion = losses.MeanSquaredError()
optimizer = optimizers.Adam(learning_rate=0.001,
                            beta_1=0.9, beta_2=0.999, amsgrad=True)
train_loss = metrics.Mean()
val_loss = metrics.Mean()

def compute_loss(t, y):
    return criterion(t, y)

def train_step(x, t):
    with tf.GradientTape() as tape:
        preds = model(x)
        loss = compute_loss(t, preds)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss(loss)

    return loss

def val_step(x, t):
    preds = model(x)
    loss = compute_loss(t, preds)
    val_loss(loss)
'''
epochs : エポック数、学習の繰り返し回数。学習率のグラフを見ながら調整
batch_size : バッチサイズ、一度の学習の量で小さいほど多く学習してから重み更新をする
'''
epochs = 2
batch_size = 256
n_batches_train = x_train.shape[0] // batch_size + 1
n_batches_val = x_val.shape[0] // batch_size + 1
hist = {'loss': [], 'val_loss': []}
es = EarlyStopping(patience=10, verbose=1)

correct_val = 0
mistake_val = 0
accuracy_rate = []

for epoch in range(epochs):
    x_, t_ = shuffle(x_train, t_train)


    for batch in range(n_batches_train):
        start = batch * batch_size
        end = start + batch_size
        train_step(x_[start:end], t_[start:end])

    for batch in range(n_batches_val):
        start = batch * batch_size
        end = start + batch_size
        val_step(x_val[start:end], t_val[start:end])

    hist['loss'].append(train_loss.result())
    hist['val_loss'].append(val_loss.result())

    print('epoch: {}, loss: {:.3}, val_loss: {:.3f}'.format(
        epoch+1,
        train_loss.result(),
        val_loss.result()
    ))

    y_val = model(x_val)
    for i in range(len(y_val)):
        if y_val[i][0] >= 0.5:
            y_v = 1
        else:
            y_v = 0

        if y_v == t_val[i][0]:
            correct_val = correct_val + 1
        else:
            mistake_val = mistake_val + 1
    accuracy_rate.append(correct_val / (correct_val + mistake_val) * 100) #検証期間の正答率を格納

    if es(val_loss.result()):break

'''
4. モデルの評価
'''
gen = [None] * (maxlen -1)  #ディジタル[0,1]

preds  = [None] * (maxlen*2 -1) #アナログ


print("予測値を計算します")
y = model(x)


correct_val = 0 #評価期間の正解数
mistake_val = 0 #評価期間の不正回数
correct_train = 0 #学習期間の正解数
mistake_train = 0 #学習期間の不正解数

y_val = model(x_val) #評価期間の出力
y_train = model(x_train) #学習期間の出力

#正解回数、不正解回数を計算
for i in range(len(y_val)):
    if y_val[i][0] >= 0.5:
        y_v = 1
    else:
        y_v = 0

    if y_v == t_val[i][0]:
        correct_val = correct_val + 1
    else:
        mistake_val = mistake_val + 1
for i in range(len(y_train)):
    if y_train[i][0] >= 0.5:
        y_t = 1
    else:
        y_t = 0
    
    if y_t == t_train[i][0]:
        correct_train = correct_train + 1
    else:
        mistake_train = mistake_train + 1



#上昇なら1、下降なら0にする
for i in range(len(x)):
    if y[i][0] >= 0.5:
        gen.append(1)
        preds.append(y[i][0])
    else:
        gen.append(0)
        preds.append(y[i][0])
df_inout['Predict'] = gen #予測結果を保存

df_inout.to_csv('df_inout.csv') #dataframe保存


print("学習期間の正答率[%]", correct_train / (correct_train + mistake_train) * 100)
print("検証期間の正答率[%]", correct_val / (correct_val + mistake_val) * 100)



#プロット-----------------------------------------------------------------------------------------
df_inout['Date'] = pd.to_datetime(df_inout['Date'])
df_2001 = df_inout[(df_inout['Date'] >= pd.to_datetime('2001-01-01')) & (df_inout['Date'] <= pd.to_datetime('2001-12-31'))].reset_index(drop=True)
print(df_inout)
#予測上昇度と株価変動
fig = plt.figure()
plt.rc('font', family='serif')
plt.title("日経平均　予測上昇度", fontname="MS Gothic", fontsize=36)
for i in range(1, len(df_2001)):
    if df_2001['Predict'][i] == 1:
        plt.plot(df_2001.loc[(df_2001['Date'] == df_2001['Date'][i-1]) | (df_2001['Date'] == df_2001['Date'][i]), ['Date']], 
                 df_2001.loc[(df_2001['Date'] == df_2001['Date'][i-1]) | (df_2001['Date'] == df_2001['Date'][i]), ['Close']],
                 color='red')
    else:
        plt.plot(df_2001.loc[(df_2001['Date'] == df_2001['Date'][i-1]) | (df_2001['Date'] == df_2001['Date'][i]), ['Date']], 
                 df_2001.loc[(df_2001['Date'] == df_2001['Date'][i-1]) | (df_2001['Date'] == df_2001['Date'][i]), ['Close']],
                 color='blue')


plt.tick_params(labelsize=18) #軸の目盛り

#エポック毎の正解率[%]
fig = plt.figure()
plt.rc('font', family='serif')
plt.title("エポック毎の正解率", fontname="MS Gothic", fontsize=36)
plt.plot([i for i in range(1, len(accuracy_rate)+1)], accuracy_rate, color='black', label="正解率")
plt.legend(prop = {"family" : "MS Gothic","size" : "20"}) #図中のラベル名
plt.tick_params(labelsize=18) #軸の目盛り



plt.show()