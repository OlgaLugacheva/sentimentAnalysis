# utils_router.py

from app import utils_bert,  utils_b


def get_model_implementation(model_id: str, model_v: str):
    model_id = model_id.strip().lower()

    if model_id == "bert":
        return utils_bert.predict_sentiment, utils_bert.predict_sentiment_batch
    elif model_id == "b":
        return utils_b.predict_sentiment, utils_b.predict_sentiment_batch
    else:
        raise ValueError(f"Unsupported model_id: {model_id}")
