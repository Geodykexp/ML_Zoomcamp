import pickle
from urllib import response
import uvicorn
from fastapi.exceptions import HTTPException    
from fastapi import FastAPI, requests
from pydantic import BaseModel, Field

count = 0

app = FastAPI(title="Logistic_Regression_Model")

# Load the trained model from the pickle file
pickle_in = open("pipeline_v1.bin", "rb")
model = pickle.load(pickle_in)
pickle_in.close()

class ClientData(BaseModel):
    lead_source: str = Field(..., description="organic_search")
    number_of_courses_viewed: int = Field(..., ge=0, description= "Number of courses viewed")
    annual_income: float = Field(..., ge=0, description="Annual income")
@app.get("/")
async def root():
    return{"message": "Welcome to the Logistic Regression Model API. Use the /score endpoint to get predictions."}
#  # return {"requests.post(url, json=client).json()"}

@app.post("/score")
def score_client(payload: ClientData): # type: ignore
    """
    Score a single client and return the positive-class probability.
    """
    sample = [payload.dict()]

    try:
        # If the pipeline is a classifier supporting predict_proba:
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(sample)
            # Assumes binary classification; take probability of the positive class (index 1).
            prob = float(probabilities[0][1])
            return {"probability": prob}
        else:
            # Fallback to predict if predict_proba not available.
            pred = model.predict(sample)
            # Cast to a simple Python type for JSON serialization.
            return {"prediction": pred[0]}
    except Exception as e:
        # Provide a clean error message to the client.
        raise HTTPException(status_code=400, detail=f"Inference failed: {e}")


