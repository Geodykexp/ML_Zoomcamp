from pathlib import Path
from typing import Optional, Union
import pickle
import uvicorn
import requests
import os
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import JSONResponse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from pydantic import BaseModel

app = FastAPI(title="Logistic_Regression_Model")

# Load the trained model from the pickle file
pickle_in = open("pipeline_v1.bin", "rb")
model = pickle.load(pickle_in)

#close the pickle file
pickle_in.close()

@app.post("/score")
def score_client(BaseModel: dict):
 
    # Make predictions using the loaded model
    prediction = model.predict(model)

    # Return the prediction as JSON
    return {"probability": prediction[0]}

@app.get("/")
async def root(request: Request):
url = "YOUR_URL"
client = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}
requests.post(url, json=client).json()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
url = "http://localhost:8000/score"
client = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}

response = requests.post(url, json=client)
probability = response.json()["probability"]
print(f"The probability that this client will get a subscription is: {probability}")