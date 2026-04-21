from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict_text

app = FastAPI()

class InputText(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Abuse Detection API :rocket:"}

@app.post("/predict")
def predict(data: InputText):
    return predict_text(data.text)
