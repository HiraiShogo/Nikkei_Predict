
import pandas as pd
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())

df = pd.read_excel('N225f_2023_Forward.xlsx', sheet_name='日中日足')
df['Date'] = df['日付']
df['Close'] = df['終値']
df = df.drop(["日付","始値","高値","安値","終値","出来高"], axis=1, inplace=False)
df.to_csv('df_nikkei_forward.csv')
df = pd.read_csv('df_nikkei_forward.csv', index_col=0)
df['Date'] = pd.to_datetime(df['Date'])
print(df)


