import io
import pandas as pd
import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

from app.utils_bert import TextInput, predict_sentiment, predict_sentiment_batch

from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from src.train_model_bert import train_and_save_bert_model
from src.train_model_bert import fine_tune_model_on_new_data


@patch("src.train_model_bert.classification_report")
@patch("src.train_model_bert.joblib.dump")
@patch("src.train_model_bert.Trainer")
@patch("src.train_model_bert.DataCollatorWithPadding")
@patch("src.train_model_bert.load_tokenizer_and_model")
@patch("src.train_model_bert.tokenize_dataset")
@patch("src.train_model_bert.map_emotions")
@patch("src.data_pre_processing.clean_text")
@patch("src.train_model_bert.pd.read_csv")
def test_train_and_save_bert_model(
        mock_read_csv, mock_clean_text, mock_map_emotions, mock_tokenize_dataset,
        mock_load_tokenizer_and_model, mock_data_collator, mock_trainer_class,
        mock_joblib_dump, mock_classification_report
):
    df = pd.DataFrame({
        "Text": ["sample"] * 10,
        "Sentiment": ["happy", "sad"] * 5
    })
    mock_read_csv.return_value = df
    mock_map_emotions.return_value = df
    mock_tokenize_dataset.return_value = "dataset"

    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_trainer = MagicMock()
    mock_trainer.predict.return_value = MagicMock(
        predictions=np.array([[0.1, 0.9]] * 10),
        label_ids=np.array([1] * 10)
    )

    mock_load_tokenizer_and_model.return_value = (mock_tokenizer, mock_model)
    mock_trainer_class.return_value = mock_trainer
    mock_clean_text.side_effect = lambda x: str(x) + "_clean"
    train_and_save_bert_model()

    assert mock_read_csv.called
    assert mock_model.save_pretrained.called
    assert mock_trainer.train.called
    assert mock_joblib_dump.called
    assert mock_tokenizer.save_pretrained.called
    assert mock_classification_report.called


from unittest.mock import patch, MagicMock
import pandas as pd


@patch("src.train_model_bert.joblib.load")
@patch("src.train_model_bert.Trainer")
@patch("src.train_model_bert.DataCollatorWithPadding")
@patch("src.train_model_bert.load_tokenizer_and_model")
@patch("src.train_model_bert.tokenize_dataset")
@patch("src.train_model_bert.map_emotions")
@patch("src.data_pre_processing.clean_text")
@patch("src.train_model_bert.pd.read_csv")
def test_fine_tune_model_on_new_data(
        mock_read_csv, mock_clean_text, mock_map_emotions, mock_tokenize_dataset,
        mock_load_tokenizer_and_model, mock_data_collator, mock_trainer_class,
        mock_joblib_load
):
    original_data = pd.DataFrame({
        "Text": ["original text"] * 5,
        "Sentiment": ["happy"] * 5
    })

    new_data = pd.DataFrame({
        "Text": ["new text"] * 5,
        "Sentiment": ["happy"] * 5
    })

    mock_read_csv.return_value = original_data
    mock_map_emotions.return_value = original_data
    mock_clean_text.side_effect = lambda x: x + "_clean"
    mock_tokenize_dataset.return_value = "train_dataset"

    mock_label_encoder = MagicMock()
    mock_label_encoder.classes_ = ["happy"]
    mock_label_encoder.transform.return_value = [0] * 10
    mock_joblib_load.return_value = mock_label_encoder

    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_trainer = MagicMock()

    mock_load_tokenizer_and_model.return_value = (mock_tokenizer, mock_model)
    mock_trainer_class.return_value = mock_trainer

    fine_tune_model_on_new_data(new_data)

    assert mock_trainer.train.called
    assert mock_model.save_pretrained.called
    assert mock_tokenizer.save_pretrained.called


@patch("app.utils_bert.load_model_components")
def test_predict_sentiment_returns_label(mock_load_components):
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_encoder = MagicMock()

    # Мокаем tokenizer: должен вернуть dict с тензорами
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]])
    }

    # Мокаем модель
    mock_outputs = MagicMock()
    mock_outputs.logits = torch.tensor([[0.2, 0.8]])  # предполагаем binary классификацию
    mock_model.return_value = mock_outputs
    mock_model.eval.return_value = None
    mock_model.__call__ = lambda **kwargs: mock_outputs

    # Мокаем энкодер
    mock_encoder.inverse_transform.return_value = ["positive"]

    mock_load_components.return_value = (mock_tokenizer, mock_model, mock_encoder)

    input_data = TextInput(text="Great product", model_version="bert_model")

    result = predict_sentiment(input_data)

    assert result == "positive"
    mock_tokenizer.assert_called_once()
    mock_encoder.inverse_transform.assert_called_once()


@patch("app.utils_bert.load_model_components")
@patch("app.utils_bert.load_and_preprocess_data")
def test_predict_sentiment_batch_returns_csv(mock_preprocess, mock_load_components):
    # Setup mocks
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_encoder = MagicMock()

    # Fake tokenizer output
    mock_tokenizer.side_effect = lambda text, **kwargs: {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]])
    }

    mock_outputs = MagicMock()
    mock_outputs.logits = torch.tensor([[0.1, 0.9]])
    mock_model.return_value = mock_outputs
    mock_model.eval.return_value = None

    mock_encoder.inverse_transform.side_effect = lambda x: ["positive"] * len(x)

    mock_load_components.return_value = (mock_tokenizer, mock_model, mock_encoder)

    # Input CSV file
    input_df = pd.DataFrame({
        "Text": ["I love it", "Terrible experience"],
        "Sentiment": ["positive", "negative"]
    })

    # Preprocessed dataframe mock
    preprocessed_df = input_df.copy()
    preprocessed_df["Text_clean"] = preprocessed_df["Text"]

    mock_preprocess.return_value = preprocessed_df

    buffer = io.BytesIO()
    input_df.to_csv(buffer, index=False)
    buffer.seek(0)

    result = predict_sentiment_batch(buffer.read(), model_version="bert_model")

    assert "file" in result
    assert "accuracy" in result
    assert result["accuracy"] == 0.5  # 1 из 2 правильно
    assert "Predict_sentiment" in pd.read_csv(io.StringIO(result["file"])).columns


@patch("app.utils_bert.load_model_components")
def test_predict_sentiment_batch_missing_text_column(mock_load_components):
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_encoder = MagicMock()

    mock_load_components.return_value = (mock_tokenizer, mock_model, mock_encoder)
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
