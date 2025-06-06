import streamlit as st
import pandas as pd
import requests
from io import BytesIO

st.title("📊 Анализ отзывов ")

API_TEXT_URL = "http://localhost:8000/predict"
API_CSV_URL = "http://localhost:8000/predict-csv"
FINE_TUNE_URL = "http://localhost:8000/fine-tune"

tab1, tab2, tab3 = st.tabs(["Один отзыв", "CSV-файл", "📊 Дашборд CSV"])

# Инициализация флагов и состояний
if "fine_tuned" not in st.session_state:
    st.session_state.fine_tuned = False
if "fine_tuned_model_id" not in st.session_state:
    st.session_state.fine_tuned_model_id = None

# Опции выбора модели и версии
model_options = {
    "BERT (transformers)": "bert",
    "Stacking": "b",
}

model_v_options = {
    "Старая версия": "_origin",
    "Дообученная версия": "_tuned",
}

# TAB 1: Один отзыв
with tab1:
    model_choice = st.selectbox("Выберите модель", list(model_options.keys()), key="model_select_tab1")
    model_id = model_options[model_choice]

    # Показать выбор версии модели, только если она была дообучена
    if st.session_state.fine_tuned and st.session_state.fine_tuned_model_id == model_id:
        model_v_label = st.selectbox("Выберите версию модели", list(model_v_options.keys()), key="version_select_tab1")
        model_v = model_v_options[model_v_label]
    else:
        model_v = model_v_options["Старая версия"]

    text = st.text_area("Введите отзыв", height=150)
    if st.button("Анализировать", key="text_analysis"):
        if not text.strip():
            st.warning("Пожалуйста, введите текст.")
        else:
                response = requests.post(API_TEXT_URL, json={"text": text, "model_id": model_id, "model_v": model_v})
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Тональность: **{result['sentiment']}**")
                else:
                    try:
                        error_detail = response.json().get("detail", "Неизвестная ошибка")
                    except Exception:
                        error_detail = response.text or "Не удалось прочитать ответ сервера"
                    st.error(f"Ошибка API ({response.status_code}): {error_detail}")

# TAB 2: CSV-файл
with tab2:
    model_choice = st.selectbox("Выберите модель", list(model_options.keys()), key="model_select_tab2")
    model_id = model_options[model_choice]

    if st.session_state.fine_tuned and st.session_state.fine_tuned_model_id == model_id:
        model_v_label = st.selectbox("Выберите версию модели", list(model_v_options.keys()), key="version_select_tab2")
        model_v = model_v_options[model_v_label]
    else:
        model_v = model_v_options["Старая версия"]

    uploaded_file = st.file_uploader("Загрузите CSV-файл с колонкой 'Text' (и опционально 'Sentiment')", type=["csv"])

    if uploaded_file is not None:
        if st.button("Анализировать CSV", key="csv_analysis"):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                params = {"model_id": model_id, "model_v": model_v}
                response = requests.post(API_CSV_URL, files=files, params=params)

                if response.status_code == 200:
                    st.session_state.df_result = pd.read_csv(BytesIO(response.content))
                    accuracy = response.headers.get("X-Accuracy")
                    st.success("✅ Анализ завершён.")
                    st.dataframe(st.session_state.df_result)
                    if accuracy:
                        st.info(f"📈 Точность предсказания: **{float(accuracy) * 100:.2f}%**")
                    st.download_button("📥 Скачать результат CSV", response.content,
                                       file_name="result.csv", mime="text/csv")
                else:
                    try:
                        error_detail = response.json().get("detail", "Неизвестная ошибка")
                    except Exception:
                        error_detail = response.text or "Не удалось прочитать ответ сервера"
                    st.error(f"Ошибка API ({response.status_code}): {error_detail}")

            except Exception as e:
                st.error(f"Ошибка при анализе: {e}")

# TAB 3: Дашборд
with tab3:
    df_result = st.session_state.get("df_result")
    if df_result is None:
        st.info("Загрузите и проанализируйте файл во вкладке 'CSV-файл', чтобы увидеть дэшборд.")
    else:
        st.subheader(" Дэшборд по результатам анализа")
        if "Predict_sentiment" not in df_result.columns:
            st.warning("В результатах нет колонки 'Predict_sentiment'. Невозможно построить визуализации.")
        else:
            df_result["Predict_sentiment"] = df_result["Predict_sentiment"].str.lower()
            mask = df_result["Sentiment"].notna()
            true_labels = df_result.loc[mask, "Sentiment"].astype(str)
            predicted_labels = df_result.loc[mask, "Predict_sentiment"].astype(str)

            # Accuracy и несовпадения
            if not true_labels.empty:
                accuracy = (true_labels == predicted_labels).mean()
                mismatched_rows = df_result.loc[
                    mask & (true_labels != predicted_labels), ["Text", "Sentiment", "Predict_sentiment"]
                ]

                st.metric("Accuracy", f"{accuracy * 100:.2f}%")

                if not mismatched_rows.empty:
                    st.subheader("🔍 Несовпадения между истинными и предсказанными значениями")
                    st.dataframe(mismatched_rows)

                    if st.button("🔁 Дообучить модель на этих примерах"):
                        fine_tune_data = mismatched_rows[["Text", "Sentiment"]].rename(columns={
                            "Text": "text",
                            "Sentiment": "label"
                        })

                        try:
                            fine_tune_model_id = "my" if model_id == "b" else "bert"
                            response = requests.post(
                                FINE_TUNE_URL,
                                params={"model_id": fine_tune_model_id},
                                json=fine_tune_data.to_dict(orient="records")
                            )

                            if response.status_code == 200:
                                st.success("✅ Модель успешно дообучена!")
                                st.session_state.fine_tuned = True
                                st.session_state.fine_tuned_model_id = model_id
                            else:
                                try:
                                    error_detail = response.json().get("detail", "Неизвестная ошибка")
                                except Exception:
                                    error_detail = response.text or "Не удалось прочитать ответ сервера"
                                st.error(f"Ошибка дообучения ({response.status_code}): {error_detail}")

                        except Exception as e:
                            st.error(f"Ошибка соединения: {e}")

