# main
from fastapi import FastAPI, UploadFile, FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

# from app.utils import predict_sentiment
# from app.utils_bert import predict_sentiment
# from app.utils_lr import predict_sentiment
from app.utils_b import predict_sentiment, predict_sentiment_batch

app = FastAPI(title="Sentiment Analysis API")


# pydantic-модель запроса
class TextInput(BaseModel):
    text: str


# Эндпоинт предсказания
@app.post("/predict/")
def predict(input_data: TextInput):
    prediction = predict_sentiment(input_data)
    return {"sentiment": prediction}


@app.get("/test/")
def predict():
    return "hello world"


@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...)):
    # Проверка расширения
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    # Чтение CSV в DataFrame
    contents = await file.read()
    output = predict_sentiment_batch(contents)
    return StreamingResponse(output, media_type="text/csv",
                             headers={"Content-Disposition": "attachment;"
                                                             " filename=result.csv"})
