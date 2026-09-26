from fastapi.testclient import TestClient
import os
from dotenv import load_dotenv

#set up the environment variable
load_dotenv()
correct_key=os.getenv("API_KEY")

from main import app

#initialize testclient
client=TestClient(app)

#test 1(if api key wrong or missing)
def test_prediction_missing_key():
    response=client.post(
        "/predict",
        json={"narrative":"altitude decreasing"}
    )
    assert response.status_code==401

#test 1.1(if key is wrong)
def test_prediction_wrong_key():
    response=client.post(
        "/predict",
        headers={"X-API-KEY":"fjdks"},
        json={"narrative":"we lost one engine mid flight and we had to emergency land"}
    )
    assert response.status_code==401

# if entered wrong input
def test_prediction_wrong_input():
    response=client.post(
        "/predict",
        json={"narrative":123},
        headers={"X-API-KEY":correct_key}
    )
    assert response.status_code==422

# testing valid request
def test_valid_request():
    response=client.post(
        "/predict",
        json={"narrative":"we were descending when one of our engines caught fire"},
        headers={"X-API-KEY":correct_key}
    )
    assert response.status_code==200
    #regression testing(if the code changes in future, the test still catches that prediction isnt there)
    assert "prediction" in response.json()

# test if id exceeds maxID
def test_connection_to_db():
    response=client.get(
        "/report/18000",
        headers={"X-API-KEY":correct_key},
    )
    assert response.status_code==404

# test if get request successful
def test_connection2_to_db():
    response=client.get(
        "/report/2453",
        headers={"X-API-KEY":correct_key},
    )
    assert response.status_code==200
    assert response.json()["ID"]==2453

# test if report_id different datatype
def test_connection3_to_db():
    response=client.get(
        "/report/'hello'",
        headers={"X-API-KEY":correct_key},
    )
    assert response.status_code==422

#test if id correct but key incorrect
def test_connection4_to_db():
    response=client.get(
        "/report/15000",
        headers={"X-API-KEY":"dhdjkf"},

    )
    assert response.status_code==401
