import pandas as pd
import sqlite3

df=pd.read_csv("clean_data.csv")
pre_df=pd.read_csv("combined_dataset.csv",encoding= "latin1")


df.insert(0,"ID",range(1,len(df)+1))
pre_df.insert(0,"ID",range(1,len(pre_df)+1))

conn=sqlite3.connect("Aviation.db")
cursor=conn.cursor()

df.to_sql("reports",conn,if_exists="replace",index=False)
pre_df.to_sql("full_reports",conn,if_exists="replace",index=False)


conn.close()