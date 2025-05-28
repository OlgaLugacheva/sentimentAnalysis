import streamlit as st
import requests
import pandas as pd
from io import BytesIO

st.title("📊 Анализ отзывов ")

API_TEXT_URL = "http://localhost:8000/predict"
API_CSV_URL = "http://localhost:8000/predict-csv"
FINE_TUNE_URL = "http://localhost:8000/fine-tune"

tab1, tab2, tab3 = st.tabs(["Один отзыв", "CSV-файл", "📊 Дашборд CSV"])

# Инициализация флага дообучения
if "fine_tuned" not in st.session_state:
    st.session_state.fine_tuned = False

# Streamlit UI
model_options = {
    "BERT (transformers)": "bert",
    "Linear Regression": "b",
}

# Опции версий модели
model_v_options = {
    "Старая версия": "bert_model",
    "Дообученная версия": "bert_model_tuned",
}

model_choice = st.selectbox("Выберите модель", list(model_options.keys()))
model_id = model_options[model_choice]

# Отображение версии модели только если выбрана BERT и была дообучена
if model_choice == "BERT (transformers)" and st.session_state.fine_tuned:
    model_v_label = st.selectbox("Выберите версию модели", list(model_v_options.keys()))
    model_v = model_v_options[model_v_label]
else:
    # По умолчанию — старая модель
    model_v = model_v_options["Старая версия"]


def post_json(url, payload, error_msg="Ошибка при соединении с API"):
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Ошибка API: {response.status_code}"
    except Exception as e:
        return None, f"{error_msg}: {e}"

def post_file(url, file_field, file_name, file_content, mime_type="text/csv"):
    try:
        files = {file_field: (file_name, file_content, mime_type)}
        response = requests.post(url, files=files)
        if response.status_code == 200:
            return response, None
        else:
            return None, f"Ошибка API: {response.status_code}"
    except Exception as e:
        return None, f"Ошибка при соединении с API: {e}"

#
# with tab1:
#     text = st.text_area("Введите отзыв", height=150)
#     if st.button("Анализировать", key="text_analysis"):
#         if not text.strip():
#             st.warning("Пожалуйста, введите текст.")
#         else:
#             try:
#                 response = requests.post(API_TEXT_URL, json={"text": text, "model_id": model_id})
#                 if response.status_code == 200:
#                     result = response.json()
#                     st.success(f"Тональность: **{result['sentiment']}**")
#                 else:
#                     st.error(f"Ошибка API: {response.status_code}")
#             except Exception as e:
#                 st.error(f"Ошибка соединения с API: {e}")
#
# import seaborn as sns
# import matplotlib.pyplot as plt
# import plotly.express as px
#
# import streamlit as st
# import requests
# import pandas as pd
# from io import BytesIO
# import seaborn as sns
# import matplotlib.pyplot as plt
# import plotly.express as px
#
# st.title("📊 Анализ отзывов")
#
# API_TEXT_URL = "http://localhost:8000/predict"
# API_CSV_URL = "http://localhost:8000/predict-csv"
# FINE_TUNE_URL = "http://localhost:8000/fine-tune"
#
# tab1, tab2, tab3 = st.tabs(["Один отзыв", "CSV-файл", "📊 Дашборд CSV"])
#
# # Флаги и состояния
# if "fine_tuned" not in st.session_state:
#     st.session_state.fine_tuned = False
# if "df_result" not in st.session_state:
#     st.session_state.df_result = None
#
# # Общие настройки модели
# model_options = {
#     "BERT (transformers)": "bert",
#     "Linear Regression": "b",
# }
# model_v_options = {
#     "Старая версия": "bert_model",
#     "Дообученная версия": "bert_model_tuned",
# }
# model_choice = st.selectbox("Выберите модель", list(model_options.keys()))
# model_id = model_options[model_choice]
# model_v = model_v_options["Старая версия"]
# if model_choice == "BERT (transformers)" and st.session_state.fine_tuned:
#     model_v_label = st.selectbox("Выберите версию модели", list(model_v_options.keys()))
#     model_v = model_v_options[model_v_label]


# ======== TAB 1 ========
with tab1:
    text = st.text_area("Введите отзыв", height=150)
    if st.button("Анализировать", key="text_analysis"):
        if not text.strip():
            st.warning("Пожалуйста, введите текст.")
        else:
            try:
                response = requests.post(API_TEXT_URL, json={"text": text, "model_id": model_id})
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Тональность: **{result['sentiment']}**")
                else:
                    st.error(f"Ошибка API: {response.status_code}")
            except Exception as e:
                st.error(f"Ошибка соединения с API: {e}")

# ======== TAB 2 ========
with tab2:
    uploaded_file = st.file_uploader("Загрузите CSV-файл с колонкой 'Text' (и опционально 'Sentiment')", type=["csv"])
    if uploaded_file is not None:
        if st.button("Анализировать CSV", key="csv_analysis"):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                response = requests.post(f"{API_CSV_URL}?model_id={model_id}&model_v={model_v}", files=files)

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
                    st.error(f"Ошибка API: {response.status_code}")
            except Exception as e:
                st.error(f"Ошибка при анализе: {e}")

# ======== TAB 3 ========
with tab3:
    df_result = st.session_state.get("df_result")
    if df_result is None:
        st.info("Загрузите и проанализируйте файл во вкладке 'CSV-файл', чтобы увидеть дэшборд.")
    else:
        st.subheader("📊 Дэшборд по результатам анализа")
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

                st.metric("📏 Accuracy", f"{accuracy * 100:.2f}%")

                if not mismatched_rows.empty:
                    st.subheader("🔍 Несовпадения между истинными и предсказанными значениями")
                    st.dataframe(mismatched_rows)

                    if st.button("🔁 Дообучить модель на этих примерах"):
                        fine_tune_data = mismatched_rows[["Text", "Sentiment"]].rename(columns={
                            "Text": "text",
                            "Sentiment": "label"
                        })

                        try:
                            response = requests.post(FINE_TUNE_URL, json=fine_tune_data.to_dict(orient="records"))
                            if response.status_code == 200:
                                st.success("✅ Модель успешно дообучена!")
                                st.session_state.fine_tuned = True
                            else:
                                st.error(f"❌ Ошибка дообучения: {response.status_code}")
                        except Exception as e:
                            st.error(f"❌ Ошибка соединения: {e}")


# with tab2:
#     uploaded_file = st.file_uploader("Загрузите CSV-файл с колонкой 'Text' (и опционально 'Sentiment')", type=["csv"])
#     if uploaded_file is not None:
#         if st.button("Анализировать CSV", key="csv_analysis"):
#             try:
#                 files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
#                 response = requests.post(f"{API_CSV_URL}?model_id={model_id}&model_v={model_v}", files=files)
#
#                 if response.status_code == 200:
#                     df_result = pd.read_csv(BytesIO(response.content))
#                     st.success("✅ Анализ завершён.")
#                     st.dataframe(df_result)
#
#                     accuracy = response.headers.get("X-Accuracy")
#                     if accuracy:
#                         st.info(f"📈 Точность предсказания: **{float(accuracy) * 100:.2f}%**")
#
#                     st.download_button("📥 Скачать результат CSV", response.content,
#                                        file_name="result.csv", mime="text/csv")
#
#                     # ===== 🎯 ВИЗУАЛИЗАЦИИ =====
#                     st.markdown("---")
#                     st.subheader("📊 Дэшборд по результатам анализа")
#
#                     if "Predict_sentiment" not in df_result.columns:
#                         st.warning(
#                             "В результирующем файле нет колонки 'Predict_sentiment'. Невозможно построить дэшборды.")
#                     else:
#                         df_result["Predict_sentiment"] = df_result["Predict_sentiment"].str.lower()
#                         ##вынести в метод - класс
#                         mask = df_result["Sentiment"].notna()
#                         true_labels = df_result.loc[mask, "Sentiment"].astype(str)
#                         predicted_labels = df_result.loc[mask, "Predict_sentiment"].astype(str)
#
#                         # Вычисляем долю совпадений
#                         if not true_labels.empty:
#                             accuracy = (true_labels == predicted_labels).mean()
#
#                             # Несовпавшие строки
#                             mismatched_rows = df_result.loc[
#                                 mask & (true_labels != predicted_labels), ["Text", "Sentiment", "Predict_sentiment"]]
#
#                             #
#                             if not mismatched_rows.empty:
#                                 st.subheader(" Несовпадения между истинными и предсказанными значениями:")
#                                 st.dataframe(mismatched_rows)
#                                 if st.button(" Дообучить модель на этих примерах"):
#                                     st.write(" Нажата кнопка дообучения")
#                                     st.write(f" Кол-во несовпадений: {len(mismatched_rows)}")
#                                     fine_tune_data = mismatched_rows[["Text", "Sentiment"]].rename(columns={
#                                         "Text": "text",
#                                         "Sentiment": "label"
#                                     })
#
#                                     # Отправка на бэкенд через API
#                                     response = requests.post("http://localhost:8000/fine-tune",
#                                                              json=fine_tune_data.to_dict(orient="records"))
#
#                                     if response.status_code == 200:
#                                         st.success("✅ Модель успешно дообучена на новых примерах!")
#                                         st.session_state.fine_tuned = True  # <-- ВАЖНО
#
#                                     else:
#                                         st.error(" Ошибка при дообучении модели.")
#
#                         # # Распределение тональностей
#                         # st.markdown("### ➤ Распределение тональностей")
#                         # sentiment_counts = df_result["Predict_sentiment"].value_counts()
#                         # fig_pie = px.pie(
#                         #     names=sentiment_counts.index,
#                         #     values=sentiment_counts.values,
#                         #     title="Распределение по тональностям"
#                         # )
#                         # st.plotly_chart(fig_pie)
#
#                         # Примеры по категориям
#                         # st.markdown("### ➤ Примеры постов")
#                         # for category in ["positive", "negative", "neutral"]:
#                         #     st.markdown(f"**{category.capitalize()}**")
#                         #     examples = df_result[df_result["Predict_sentiment"] == category]["Text"].dropna().head(3)
#                         #     for text in examples:
#                         #         st.write(f"• {text}")
#
#
#
#                         # # Динамика тональностей по дням
#                         # if "Timestamp" in df_result.columns:
#                         #     st.markdown("### ➤ Динамика тональностей по дням")
#                         #     df_result["Timestamp"] = pd.to_datetime(df_result["Timestamp"], errors="coerce")
#                         #     df_result["Date"] = df_result["Timestamp"].dt.date
#                         #     daily = df_result.groupby(["Date", "Predict_sentiment"]).size().reset_index(name="Count")
#                         #     fig_line = px.line(daily, x="Date", y="Count", color="Predict_sentiment", markers=True)
#                         #     st.plotly_chart(fig_line, use_container_width=True)
#
#                         # # Тепловая карта по часам
#                         # if "Hour" in df_result.columns:
#                         #     st.markdown("### ➤ Активность по часам")
#                         #     pivot = df_result.pivot_table(index="Hour", columns="Predict_sentiment", values="Text",
#                         #                                   aggfunc="count").fillna(0)
#                         #     fig, ax = plt.subplots(figsize=(10, 4))
#                         #     sns.heatmap(pivot, annot=True, fmt="g", cmap="YlGnBu", ax=ax)
#                         #     st.pyplot(fig)
#                         #
#                         # # Вовлеченность
#                         # if "Likes" in df_result.columns and "Retweets" in df_result.columns:
#                         #     st.markdown("### ➤ Средняя вовлечённость по тональностям")
#                         #     df_result["Engagement"] = df_result["Likes"].fillna(0) + df_result["Retweets"].fillna(0)
#                         #     engagement_avg = df_result.groupby("Predict_sentiment")["Engagement"].mean()
#                         #     st.bar_chart(engagement_avg)
#                         #
#                         # # География негатива
#                         # if "Country" in df_result.columns:
#                         #     st.markdown("### ➤ Топ стран с негативом")
#                         #     neg_by_country = df_result[df_result["Predict_sentiment"] == "negative"].groupby(
#                         #         "Country").size().sort_values(ascending=False).head(10)
#                         #     st.bar_chart(neg_by_country)
#                         #
#                         #     # Солнечная диаграмма эмоций по странам
#                         #     st.markdown("### ➤ География эмоций")
#                         #     country_stats = df_result.groupby(["Country", "Predict_sentiment"]).size().reset_index(
#                         #         name="Count")
#                         #     fig_geo = px.sunburst(country_stats, path=["Country", "Predict_sentiment"],
#                         #                           values="Count")
#                         #     st.plotly_chart(fig_geo)
#                         #
#                         # # Хэштеги
#                         # if "Hashtags" in df_result.columns:
#                         #     st.markdown("### ➤ Топ хэштегов по реакции")
#                         #     import re
#                         #     from collections import Counter
#                         #
#                         #
#                         #     def extract_hashtags(text):
#                         #         return re.findall(r"#\w+", str(text))
#                         #
#                         #
#                         #     all_hashtags = df_result["Hashtags"].dropna().apply(extract_hashtags).sum()
#                         #     hashtag_counts = Counter(all_hashtags)
#                         #     top_hashtags = pd.DataFrame(hashtag_counts.most_common(10), columns=["Hashtag", "Count"])
#                         #     fig_tags = px.bar(top_hashtags, x="Hashtag", y="Count", title="Топ-10 хэштегов")
#                         #     st.plotly_chart(fig_tags, use_container_width=True)
#
#                 else:
#                     st.error(f"Ошибка API: {response.status_code}")
#             except Exception as e:
#                 st.error(f"Ошибка при обработке файла: {e}")

# with tab3:
#     st.subheader("📊 Визуализация данных")
#
#     if uploaded_file is None:
#         st.info("Сначала загрузите CSV во вкладке 'CSV-файл'.")
#     else:
#         df = pd.read_csv(uploaded_file)
#
#         # Предобработка
#         df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
#         df['hour'] = df['Timestamp'].dt.hour
#         df['date'] = df['Timestamp'].dt.date
#
#         col1, col2 = st.columns(2)
#
#         with col1:
#             st.subheader("Распределение тональностей")
#             st.bar_chart(df['Sentiment'].value_counts())
#
#         with col2:
#             st.subheader("Тональность")
#             if 'Platform' in df.columns:
#                 st.bar_chart(df.groupby(['Platform', 'Sentiment']).size().unstack(fill_value=0))
#             else:
#                 st.warning("Колонка 'airline' не найдена в данных.")
#
#         st.subheader(" Активность по часам")
#         if df['Timestamp'].notnull().any():
#             import altair as alt
#             line_chart = (
#                 alt.Chart(df.dropna(subset=["Timestamp"]))
#                 .mark_line()
#                 .encode(
#                     x="hour:Q",
#                     y="count()",
#                     color="Sentiment"
#                 )
#             )
#             st.altair_chart(line_chart, use_container_width=True)
#
#         if 'negativereason' in df.columns:
#             st.subheader("💢 Причины негатива")
#             st.bar_chart(df['negativereason'].value_counts())
#
#         if 'tweet_coord' in df.columns and df['tweet_coord'].notnull().any():
#             st.subheader("🗺️ География твитов")
#             try:
#                 df['coords'] = df['tweet_coord'].apply(lambda x: eval(x) if pd.notnull(x) else None)
#                 df[['lat', 'lon']] = pd.DataFrame(df['coords'].tolist(), index=df.index)
#                 st.map(df[['lat', 'lon']].dropna())
#             except Exception:
#                 st.warning("Не удалось распарсить координаты.")
#                 # Нормализация колонок
#                 df.columns = df.columns.str.strip()  # убрать пробелы в именах
#                 st.write("Обнаружены колонки:", df.columns.tolist())
#                 # Проверка базовых колонок
#                 if 'Sentiment' in df.columns:
#                     st.subheader("Распределение тональностей")
#                     st.bar_chart(df['Sentiment'].value_counts())
#
#                 if 'Platform' in df.columns:
#                     st.subheader("Тональность по платформам")
#                     st.bar_chart(df.groupby(['Platform', 'Sentiment']).size().unstack(fill_value=0))
#
#                 if 'Country' in df.columns:
#                     st.subheader("🌍 География пользователей")
#                     country_counts = df['Country'].value_counts().reset_index()
#                     country_counts.columns = ['Country', 'Count']
#                     st.dataframe(country_counts)
#
#                 if 'Hashtags' in df.columns:
#                     st.subheader("🔥 Часто используемые хэштеги")
#                     from collections import Counter
#
#                     hashtags = df['Hashtags'].dropna().astype(str).str.lower().str.split()
#                     flat_hashtags = [tag.strip("#,") for sublist in hashtags for tag in sublist if tag]
#                     top_tags = Counter(flat_hashtags).most_common(15)
#                     top_df = pd.DataFrame(top_tags, columns=['Hashtag', 'Count'])
#                     st.bar_chart(top_df.set_index("Hashtag"))
#
#                 if {'Likes', 'Sentiment'}.issubset(df.columns):
#                     st.subheader("❤️ Среднее количество лайков по тональности")
#                     st.bar_chart(df.groupby('Sentiment')['Likes'].mean())
#
#                 if 'Hour' in df.columns and 'Sentiment' in df.columns:
#                     st.subheader(" Активность по часам")
#                     import altair as alt
#
#                     chart = alt.Chart(df).mark_line().encode(
#                         x="Hour:Q",
#                         y="count()",
#                         color="Sentiment"
#                     )
#                     st.altair_chart(chart, use_container_width=True)

