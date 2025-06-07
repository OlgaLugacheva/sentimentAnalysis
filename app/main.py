# main
import io

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator
from app.utils_router import get_model_implementation, get_model_fine_tune
from pydantic import BaseModel, Field
from typing import List, Literal
import pandas as pd

app = FastAPI(title="Sentiment Analysis API")


class TextInput(BaseModel):
    text: str = Field(..., min_length=1, description="Нельзя отправлять пустой текст")
    @field_validator("text")
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Текст не должен состоять только из пробелов")
        if not any(char.isalpha() for char in v):
            raise ValueError("Текст должен содержать хотя бы одну букву")
        return v

# Эндпоинт предсказания
@app.post("/predict")
def predict(input_data: TextInput, model_id: str = Query("bert")):
    try:
        predict_fn, _ = get_model_implementation(model_id)
        sentiment = predict_fn(input_data)
        return {"sentiment": sentiment}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Внутренняя ошибка")


class FineTuneItem(BaseModel):
    text: str = Field(..., min_length=1)
    label: Literal["positive", "negative", "neutral"]


# Предсказание по набору данных в виде файла csv
@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...), model_id: str = Query("bert"),
                      model_v: str = Query("bert_model")):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    contents = await file.read()
    try:
        _, predict_batch_fn = get_model_implementation(model_id)
        result = predict_batch_fn(contents, model_v)
        csv_data = result["file"]
        accuracy = result["accuracy"]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Ошибка модели: {str(e)}")
    except Exception:
        raise HTTPException(status_code=500, detail="Ошибка при предсказании")

    # Подготовка ответа
    stream = io.StringIO(csv_data)
    headers = {
        "Content-Disposition": "attachment; filename=result.csv"
    }

    if accuracy is not None:
        headers["X-Accuracy"] = str(accuracy)

    return StreamingResponse(stream, media_type="text/csv", headers=headers)


# Эндпоинт дообучения
@app.post("/fine-tune")
def fine_tune_endpoint(data: List[FineTuneItem], model_id: str = Query("bert")):
    if not data:
        raise HTTPException(status_code=400, detail="Data cannot be empty")
    df = pd.DataFrame([{"Text": item.text, "Sentiment": item.label} for item in data])

    try:
        fine_tune_batch_fn = get_model_fine_tune(model_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Неподдерживаемый тип модели: {str(e)}")
    if fine_tune_batch_fn is None:
        raise HTTPException(status_code=400, detail="Для данной модели недоступно дообучение")
    try:
        fine_tune_batch_fn(df)
        return {"message": "Модель успешно дообучена"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при дообучении модели: {str(e)}")
