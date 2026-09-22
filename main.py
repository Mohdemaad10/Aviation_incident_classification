from fastapi import FastAPI, HTTPException
import joblib
import sqlite3
from pydantic import BaseModel


# Initialize fastapi
app= FastAPI()

# class using pydantic for data validation (tells fastapi which data type to accept)
class PredictionRequest(BaseModel):
    narrative:str

# deserialize the model loaded in logisticregression.ipynb
lr_load=joblib.load("lr_model_pipeline.joblib")

# call prediction
@app.post("/predict")
def prediction(request: PredictionRequest):
     prediction = lr_load.predict([request.narrative])
     return {"prediction": prediction[0]}

# call report
@app.get("/report/{report_id}")
def ReportRequest(report_id: int):
     conn=sqlite3.connect("Aviation.db")
     cursor=conn.cursor()
     cursor.execute('SELECT ID, "Narrative Text", "Event Label" from reports where ID=?',(report_id,))
     row=cursor.fetchone()
     conn.close()
     if row is None:
               raise HTTPException(status_code=404, detail="not found bruh!😮‍💨")
     return {
           "ID": row[0],
           "Narrative Text": row[1],
           "Event Label": row[2]
     }
     

