from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
import os
import joblib
import sqlite3
from pydantic import BaseModel
from dotenv import load_dotenv

# Initialize fastapi
app= FastAPI()

# to read the api_key in .env text file
load_dotenv()

# get the secret key from the environment
secret_api_key=os.getenv("API_KEY")
if secret_api_key is None:
    raise RuntimeError("secret key not provided")

#initialize the what api header fastapi should get
api_key_header=APIKeyHeader(name="X-API-Key", auto_error=False)

#function to verify if the api key given is correct
def verify_api_key(api_key: str = Security(api_key_header)):
      if api_key != secret_api_key:
            raise HTTPException(status_code=401,detail="not valid")
      return api_key

            

# class using pydantic for data validation (tells fastapi which data type to accept)
class PredictionRequest(BaseModel):
    narrative:str

# deserialize the model loaded in logisticregression.ipynb
lr_load=joblib.load("lr_model_pipeline.joblib")

# call prediction
@app.post("/predict")
def prediction(request: PredictionRequest,
               api_key: str = Security(verify_api_key)
               ):
     prediction = lr_load.predict([request.narrative])
     return {"prediction": prediction[0]}

# call report
@app.get("/report/{report_id}")
def ReportRequest(report_id: int,
                  api_key: str = Security(verify_api_key)
                  ):
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
     

