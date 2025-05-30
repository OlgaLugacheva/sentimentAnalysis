# main
import io

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.utils_router import get_model_implementation, get_model_fine_tune
from pydantic import BaseModel
from typing import List
import pandas as pd
app = FastAPI(title="Sentiment Analysis API")


# pydantic-модель запроса
class TextInput(BaseModel):
    text: str


# Эндпоинт предсказания
@app.post("/predict/")
def predict(input_data: TextInput, model_id: str = Query("bert")):
    predict_fn, _ = get_model_implementation(model_id)
    prediction = predict_fn(input_data)
    return {"sentiment": prediction}


@app.get("/test/")
def predict():
    return "hello world"

class FineTuneItem(BaseModel):
    text: str
    label: str

@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...), model_id: str = Query("bert"),
                      model_v: str = Query("bert_model")):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    contents = await file.read()
    _, predict_batch_fn = get_model_implementation(model_id)
    result = predict_batch_fn(contents, model_v)
    csv_data = result["file"]
    accuracy = result["accuracy"]

    # Подготовка ответа
    stream = io.StringIO(csv_data)
    headers = {
        "Content-Disposition": "attachment; filename=result.csv"
    }

    if accuracy is not None:
        headers["X-Accuracy"] = str(accuracy)

    return StreamingResponse(stream, media_type="text/csv", headers=headers)

@app.post("/fine-tune")
def fine_tune_endpoint(data: List[FineTuneItem], model_id: str = Query("bert")):
    df = pd.DataFrame([{"Text": item.text, "Sentiment": item.label} for item in data])
    try:
        fine_tune_batch_fn = get_model_fine_tune(model_id)
        fine_tune_batch_fn(df)
        return {"message": "Модель успешно дообучена"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при дообучении модели: {str(e)}")