
from app import utils_bert,  utils_b
from src import train_sent_model,  train_model_bert


def get_model_implementation(model_id: str):
    model_id = model_id.strip().lower()

    if model_id == "bert":
        return utils_bert.predict_sentiment, utils_bert.predict_sentiment_batch
    elif model_id == "b":
        return utils_b.predict_sentiment, utils_b.predict_sentiment_batch
    else:
        raise ValueError(f"Unsupported model_id: {model_id}")

def get_model_fine_tune(model_v: str):
    model_id = model_v.strip().lower()

    if model_id == "bert":
        return train_model_bert.fine_tune_model_on_new_data
    elif model_id == "my":
        return train_sent_model.fine_tune_model_on_new_data
    else:
        raise ValueError(f"Unsupported fine tune model: {model_id}")