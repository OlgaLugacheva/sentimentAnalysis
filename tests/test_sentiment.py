import io
import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from app.utils_b import TextInput, load_model_components, predict_sentiment, predict_sentiment_batch


# === Тесты для load_model_components ===

@patch("joblib.load")
def test_load_model_components_origin_success(mock_joblib_load):
    mock_vectorizer = MagicMock()
    mock_model = MagicMock()
    mock_encoder = MagicMock()
    mock_joblib_load.side_effect = [mock_vectorizer, mock_model, mock_encoder]

    vectorizer, model, encoder = load_model_components(model_version="_origin")

    assert vectorizer is mock_vectorizer
    assert model is mock_model
    assert encoder is mock_encoder
    assert mock_joblib_load.call_count == 3


@patch("joblib.load", side_effect=FileNotFoundError("File not found"))
def test_load_model_components_raises_http_exception(mock_joblib_load):
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        load_model_components(model_version="_origin")
    assert exc_info.value.status_code == 500
    assert "Ошибка загрузки модели" in exc_info.value.detail


# === Тесты для predict_sentiment ===

@patch("app.utils_b.load_model_components")  # или корректный путь
def test_predict_sentiment_returns_label(mock_load_components):
    mock_vectorizer = MagicMock()
    mock_model = MagicMock()
    mock_encoder = MagicMock()

    mock_sparse_matrix = MagicMock()
    mock_sparse_matrix.toarray.return_value = np.array([[0.1, 0.9]])

    mock_vectorizer.transform.return_value = mock_sparse_matrix
    mock_model.predict_proba.return_value = np.array([[0.1, 0.9]])
    mock_encoder.inverse_transform.return_value = ["positive"]

    mock_load_components.return_value = (mock_vectorizer, mock_model, mock_encoder)

    input_data = TextInput(text="Good product")

    result = predict_sentiment(input_data)

    assert result == "positive"
    mock_vectorizer.transform.assert_called_once()
    mock_model.predict_proba.assert_called_once()
    mock_encoder.inverse_transform.assert_called_once()



# === Тесты для predict_sentiment_batch ===

@patch("app.utils_b.load_model_components")
@patch("app.utils_b.predict_sentiment")
def test_predict_sentiment_batch_with_sentiment_column(mock_predict, mock_load_components):
    mock_predict.side_effect = ["positive", "negative"]
    mock_vectorizer = MagicMock()
    mock_model = MagicMock()
    mock_encoder = MagicMock()
    mock_load_components.return_value = (mock_vectorizer, mock_model, mock_encoder)

    df = pd.DataFrame({
        "Text": ["Nice job", "Terrible service"],
        "Sentiment": ["positive", "negative"]
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)

    result = predict_sentiment_batch(buf.read())

    assert "file" in result
    assert "accuracy" in result
    assert isinstance(result["file"], str)
    assert result["accuracy"] == 1.0  # perfect match


@patch("app.utils_b.load_model_components")
def test_predict_sentiment_batch_missing_text_column(mock_load_components):
    mock_vectorizer = MagicMock()
    mock_model = MagicMock()
    mock_encoder = MagicMock()
    mock_load_components.return_value = (mock_vectorizer, mock_model, mock_encoder)

    df = pd.DataFrame({
        "Message": ["Hello", "World"]
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)

    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        predict_sentiment_batch(buf.read())
    assert exc_info.value.status_code == 400
    assert "CSV must contain 'Text' column" in exc_info.value.detail
