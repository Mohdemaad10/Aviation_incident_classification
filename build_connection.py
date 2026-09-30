import pandas as pd
import sqlite3

df=pd.read_csv("clean_data.csv")


df.insert(0,"ID",range(1,len(df)+1))

conn=sqlite3.connect("Aviation.db")
cursor=conn.cursor()

df.to_sql("reports",conn,if_exists="replace",index=False)


conn.close()