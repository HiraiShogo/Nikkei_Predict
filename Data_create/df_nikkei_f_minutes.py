
import pandas as pd
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('./nikkei_forward_data')
print(os.getcwd())

file_name = []
df_arr = []
for i in range(1, 24):#24
    print(i)
    if i<=9:
        file_name.append('N225f_200'+str(i)+'.xls')
    elif i<=15:
        file_name.append('N225f_20'+str(i)+'.xls')
    else:
        file_name.append('N225f_20'+str(i)+'.xlsx')
    df_arr.append(pd.read_excel(file_name[i-1], sheet_name='5min'))
print(file_name)
df = pd.concat(df_arr,axis=0).reset_index(drop=True)
df = df[['日付','時間','高値','安値','終値']]
df['Date'] = df['日付']
df['Time'] = df['時間']
df['5min High'] = df['高値']
df['5min Low'] = df['安値']
df['5min Close'] = df['終値']
df = df.drop(["日付","時間","高値","安値","終値"], axis=1, inplace=False)

print(df)


os.chdir('..')
df.to_csv('df_5min_forward.csv')

#df = pd.read_csv('df_5min_forward.csv', index_col=0)
#df['Date'] = pd.to_datetime(df['Date'])




