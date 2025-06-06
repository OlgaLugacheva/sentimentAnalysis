import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

from src.train_sent_model import train_models, tfidf_vectorizer_split

class TestModelTraining(unittest.TestCase):

    @patch("src.train_sent_model.TfidfVectorizer")
    def test_tfidf_vectorizer_split(self, mock_vectorizer_class):

        df = pd.DataFrame({
            "Text_clean": ["good product", "bad service", "excellent experience", "poor quality", "great value"] * 10,
            "Sentiment": ["positive", "negative", "positive", "negative", "positive"] * 10
        })

        # Мок vectorizer
        mock_vectorizer = MagicMock()
        mock_vectorizer.fit_transform.return_value = csr_matrix(np.random.rand(50, 100))
        mock_vectorizer.transform.return_value = csr_matrix(np.random.rand(20, 100))
        mock_vectorizer_class.return_value = mock_vectorizer

        label_encoder = MagicMock()
        label_encoder.fit_transform.return_value = np.array([0, 1, 0, 1, 0] * 10)

        # Вызов
        X_train, X_test, y_train, y_test, vectorizer = tfidf_vectorizer_split(df, label_encoder)

        # Проверка
        self.assertEqual(X_train.shape[0], 50)
        self.assertEqual(X_test.shape[0], 20)
        mock_vectorizer.fit_transform.assert_called_once()

    @patch("src.train_sent_model.LogisticRegression")
    @patch("src.train_sent_model.MultinomialNB")
    @patch("src.train_sent_model.RandomForestClassifier")
    @patch("src.train_sent_model.StackingClassifier")  # добавим мок для стекинга
    def test_train_models(self, mock_stacker_cls, mock_rf, mock_nb, mock_lr):
        # Общая заглушка модели
        mock_model = MagicMock()
        mock_model.fit.return_value = None
        mock_model.predict.return_value = np.random.choice([0, 1], size=10)
        mock_model.classes_ = np.array([0, 1])

        # Все базовые модели возвращают один и тот же мок
        mock_lr.return_value = mock_model
        mock_nb.return_value = mock_model
        mock_rf.return_value = mock_model

        # Мок стекинг-модели
        mock_stacker = MagicMock()
        mock_stacker.fit.return_value = None
        mock_stacker.predict.return_value = np.random.choice([0, 1], size=10)
        mock_stacker.classes_ = np.array([0, 1])
        mock_stacker_cls.return_value = mock_stacker

        # Данные
        X_train = csr_matrix(np.random.rand(10, 100))
        X_test = csr_matrix(np.random.rand(10, 100))
        y_train = np.random.choice([0, 1], size=10)
        y_test = np.random.choice([0, 1], size=10)

        # Вызов функции
        train_models(X_train, X_test, y_train, y_test)

        # Проверки, что методы вызывались
        self.assertTrue(mock_model.fit.called, "fit() не был вызван ни разу")
        self.assertTrue(mock_model.predict.called, "predict() не был вызван ни разу")
        self.assertTrue(mock_stacker.fit.called, "StackingClassifier.fit() не был вызван")
        self.assertTrue(mock_stacker.predict.called, "StackingClassifier.predict() не был вызван")

