import streamlit as st
import requests
import pandas as pd
from io import BytesIO

st.title("📊 Анализ отзывов (через API)")

API_TEXT_URL = "http://localhost:8000/predict"
API_CSV_URL = "http://localhost:8000/predict-csv"

tab1, tab2 = st.tabs(["Один отзыв", "CSV-файл"])

with tab1:
    text = st.text_area("Введите отзыв", height=150)
    if st.button("Анализировать", key="text_analysis"):
        if not text.strip():
            st.warning("Пожалуйста, введите текст.")
        else:
            try:
                response = requests.post(API_TEXT_URL, json={"text": text})
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Тональность: **{result['sentiment']}**")
                else:
                    st.error(f"Ошибка API: {response.status_code}")
            except Exception as e:
                st.error(f"Ошибка соединения с API: {e}")

with tab2:
    uploaded_file = st.file_uploader("Загрузите CSV-файл с колонкой 'Text' (и опционально 'Sentiment')", type=["csv"])
    if uploaded_file is not None:
        if st.button("Анализировать CSV", key="csv_analysis"):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                response = requests.post(API_CSV_URL, files=files)

                if response.status_code == 200:
                    # Читаем CSV-ответ
                    df_result = pd.read_csv(BytesIO(response.content))
                    st.success("✅ Анализ завершён.")
                    st.dataframe(df_result)

                    # Показываем точность, если заголовок X-Accuracy присутствует
                    accuracy = response.headers.get("X-Accuracy")
                    if accuracy:
                        st.info(f"📈 Точность предсказания: **{float(accuracy) * 100:.2f}%**")

                    # Кнопка для скачивания
                    st.download_button("📥 Скачать результат CSV", response.content,
                                       file_name="result.csv", mime="text/csv")
                else:
                    st.error(f"Ошибка API: {response.status_code}")
            except Exception as e:
                st.error(f"Ошибка при обработке файла: {e}")
