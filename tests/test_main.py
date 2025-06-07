import io
import sys
import os
from fastapi.testclient import TestClient
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.main import app

client = TestClient(app)

url = "http://127.0.0.1:8000/"


# ========================
# /predict
# ========================

from unittest.mock import patch, MagicMock

@patch("app.main.get_model_implementation")
def test_predict_valid_text(mock_get_model_impl):
    mock_get_model_impl.return_value = (lambda txt: "positive", None)
    resp = client.post("/predict", json={"text": "I love this!"})
    assert resp.status_code == 200
    assert resp.json() == {"sentiment": "positive"}



@patch("app.main.get_model_implementation")
def test_predict_invalid_model(mock_get_model_impl):
    mock_get_model_impl.side_effect = ValueError("bad model")
    resp = client.post("/predict?model_id=wrong", json={"text": "hi"})
    assert resp.status_code == 400
    assert "bad model" in resp.text

def test_predict_validation_error():
    resp = client.post("/predict", json={"text": ""})
    assert resp.status_code == 422


# ---------------------------------------------------------------------
# /predict-csv
# ---------------------------------------------------------------------

@patch("app.main.get_model_implementation")
def test_predict_csv_ok(mock_get_model_impl):
    batch_fn = MagicMock(return_value={
        "file": "Text,Predict_sentiment\nWow,positive",
        "accuracy": 0.9
    })
    mock_get_model_impl.return_value = (None, batch_fn)

    csv = io.BytesIO(b"Text\nWow")
    resp = client.post(
        "/predict-csv?model_id=bert&model_v=bert_model",
        files={"file": ("test.csv", csv, "text/csv")}
    )

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "text/csv; charset=utf-8"
    assert float(resp.headers["X-Accuracy"]) == 0.9

def test_predict_csv_bad_ext():
    resp = client.post(
        "/predict-csv",
        files={"file": ("bad.txt", io.BytesIO(b"x"), "text/plain")}
    )
    assert resp.status_code == 400
    assert "Only CSV files" in resp.text


# ---------------------------------------------------------------------
# /fine-tune
# # ---------------------------------------------------------------------
#
@patch("app.main.get_model_fine_tune")
def test_fine_tune_ok(mock_get_fine):
    mock_get_fine.return_value = MagicMock()
    payload = [
        {"text": "Great", "label": "positive"},
        {"text": "Awful", "label": "negative"}
    ]
    resp = client.post("/fine-tune?model_id=bert", json=payload)
    assert resp.status_code == 200
    assert "успешно" in resp.json()["message"]

@patch("app.main.get_model_fine_tune")
def test_fine_tune_unsupported(mock_get_fine):
    mock_get_fine.return_value = None
    resp = client.post("/fine-tune?model_id=nope", json=[{"text": "x", "label": "positive"}])
    assert resp.status_code == 400
    assert "недоступно" in resp.text

def test_fine_tune_validation_error():
    resp = client.post("/fine-tune?model_id=bert", json=[{"text": "x", "label": "wrong"}])
    assert resp.status_code == 422

def test_fine_tune_empty():
    resp = client.post("/fine-tune?model_id=bert", json=[])
    assert resp.status_code == 400
    assert "Data cannot be empty" in resp.text


def test_predict_empty_text():
    response = client.post("/predict/?model_id=bert", json={"text": ""})
    assert response.status_code == 422  # validation error


def test_predict_invalid_model_1():
    response = client.post("/predict/?model_id=invalid", json={"text": "Test"})
    assert response.status_code in [400, 500]  # depending on implementation


# ========================
# /predict-csv
# ========================



def test_predict_csv_invalid_extension():
    file = io.BytesIO(b"Not CSV content")
    response = client.post(
        "/predict-csv?model_id=bert&model_v=bert_model",
        files={"file": ("test.txt", file, "text/plain")},
    )
    assert response.status_code == 400
    assert "Only CSV files are supported" in response.text


def test_predict_csv_invalid_model():
    csv_content = "Text\nGood\nBad"
    file = io.BytesIO(csv_content.encode("utf-8"))
    response = client.post(
        "/predict-csv?model_id=unknown&model_v=bert_model",
        files={"file": ("test.csv", file, "text/csv")},
    )
    assert response.status_code in [400, 500]


# ========================
# /fine-tune
# ========================




def test_fine_tune_empty_data():
    response = client.post("/fine-tune?model_id=bert", json=[])
    assert response.status_code == 400
    assert "Data cannot be empty" in response.text


def test_fine_tune_invalid_label():
    data = [{"text": "meh", "label": "unknown"}]
    response = client.post("/fine-tune?model_id=bert", json=data)
    assert response.status_code == 422


def test_fine_tune_unsupported_model():
    data = [{"text": "nice", "label": "positive"}]
    response = client.post("/fine-tune?model_id=invalid", json=data)
    assert response.status_code in [400, 500]
    assert "Неподдерживаемый тип" in response.text


@patch("app.main.get_model_implementation")
def test_predict_valid_text_1(mock_get_model_implementation):
    mock_predict = MagicMock(return_value="positive")
    mock_get_model_implementation.return_value = (mock_predict, None)

    response = client.post("/predict/?model_id=bert", json={"text": "Great!"})
    assert response.status_code == 200
    assert response.json() == {"sentiment": "positive"}

@patch("app.utils_router.get_model_implementation")
def test_predict_invalid_model_2(mock_get_model_implementation):
    mock_get_model_implementation.side_effect = ValueError("Invalid model")
    response = client.post("/predict/?model_id=unknown", json={"text": "Test"})
    assert response.status_code == 400
    assert "Unsupported model_id" in response.text

# ======================
# /predict-csv
# ======================

@patch("app.main.get_model_implementation")
def test_predict_csv_valid_file_200(mock_get_model_implementation):
    mock_predict_batch = MagicMock(return_value={
        "file": "Text,Predict_sentiment\nWow!,positive\nTerrible,negative",
        "accuracy": 0.85
    })
    mock_get_model_implementation.return_value = (None, mock_predict_batch)

    csv_content = "Text\nWow!\nTerrible"
    file = io.BytesIO(csv_content.encode("utf-8"))
    response = client.post(
        "/predict-csv?model_id=bert&model_v=bert_model",
        files={"file": ("test.csv", file, "text/csv")},
    )
    assert response.status_code == 200
    assert "X-Accuracy" in response.headers
    assert float(response.headers["X-Accuracy"]) == 0.85
    assert response.headers["content-type"] == "text/csv; charset=utf-8"

@patch("app.utils_router.get_model_implementation")
def test_predict_csv_model_error(mock_get_model_implementation):
    mock_get_model_implementation.side_effect = Exception("Oops")
    csv_content = "Text\nHello"
    file = io.BytesIO(csv_content.encode("utf-8"))
    response = client.post(
        "/predict-csv?model_id=bert&model_v=bert_model",
        files={"file": ("test.csv", file, "text/csv")},
    )
    assert response.status_code == 500

# ======================
# /fine-tune
# ======================

@patch("app.main.get_model_fine_tune")
def test_fine_tune_valid_data_200(mock_get_model_fine_tune):
    mock_fn = MagicMock()
    mock_get_model_fine_tune.return_value = mock_fn

    payload = [
        {"text": "Amazing experience", "label": "positive"},
        {"text": "It sucked", "label": "negative"},
    ]
    response = client.post("/fine-tune?model_id=bert", json=payload)
    assert response.status_code == 200
    assert "успешно" in response.json()["message"]

@patch("app.main.get_model_fine_tune")
def test_fine_tune_unsupported_model_400(mock_get_model_fine_tune):
    mock_get_model_fine_tune.return_value = None
    payload = [{"text": "Blah", "label": "neutral"}]
    response = client.post("/fine-tune?model_id=bad_model", json=payload)
    assert response.status_code == 400
    assert "недоступно дообучение" in response.text

@patch("app.utils_router.get_model_fine_tune")
def test_fine_tune_crash(mock_get_model_fine_tune):
    mock_get_model_fine_tune.side_effect = Exception("Training crash")
    payload = [{"text": "whatever", "label": "positive"}]
    response = client.post("/fine-tune?model_id=bert", json=payload)
    assert response.status_code == 500
    assert "ошибка" in response.text.lower()