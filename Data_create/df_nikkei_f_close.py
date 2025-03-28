import pandas as pd
import numpy as np
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())

'''
"df_5min_forward.csv"の5分足から日足を作り出す。
'''
df = pd.read_csv('df_5min_forward.csv', index_col=0)
df_close = df.iloc[:,[0,1,4]]
df_h_l = df.iloc[:,[0,1,2,3]]

df_day = pd.DataFrame(np.arange(3).reshape(1, 3),columns=['Date', 'Time', '5min Close'],index=['0'])

section_h = []
section_l = []

h_arr = []
l_arr = []

for i in range(1,len(df)):#len(df)
    if df['Date'][i] != df['Date'][i-1]:
        df_day = pd.concat([df_day, df_close[i-1:i]], axis=0)
        h_arr.append(max(section_h))
        l_arr.append(min(section_l))
        section_h = []
        section_l = []

    section_h.append(df_h_l['5min High'][i-1])
    section_l.append(df_h_l['5min Low'][i-1])
df_day = pd.concat([df_day, df_close[len(df)-1:len(df)]], axis=0)
h_arr.append(max(section_h))
l_arr.append(min(section_l))


df_day['Day Close'] = df_day['5min Close']
df_day = df_day.drop(["5min Close"], axis=1, inplace=False)
df_day = df_day.reset_index(drop=True)
df_day = df_day.drop(index=0).reset_index(drop=True)
df_day['Day High'] = h_arr
df_day['Day Low'] = l_arr
print(df_day)
df_day.to_csv('df_day_forward.csv')
