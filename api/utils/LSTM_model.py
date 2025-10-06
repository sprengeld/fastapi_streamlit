import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from utils.prepare_embedding import load_embedding_from_w2v


# --- Параметры, с которыми модель была создана и обучена ---
HIDDEN_SIZE = 32
SEQ_LEN = 64
EMBEDDING_DIM = 64
# Загрузка модели эмбеддинга
embedding_layer = load_embedding_from_w2v("../api/weights/word2vec.model", "../api/weights/vocab.pkl", embedding_dim=EMBEDDING_DIM)


class BahdanauAttention(nn.Module):
  def __init__(
      self,
      hidden_size: int = HIDDEN_SIZE
  ) -> None:

    super().__init__()
    self.hidden_size = hidden_size
    # Линейный слой для преобразования выходов LSTM в ключи (keys)
    self.W_k = nn.Linear(self.hidden_size, self.hidden_size)
    # Линейный слой для преобразования последнего скрытого состояния в запрос (query)
    self.W_q = nn.Linear(self.hidden_size, self.hidden_size)
    # Линейный слой для вычисления оценок внимания (оценка одного скалярного значения на позицию)
    self.W_v = nn.Linear(self.hidden_size, 1)
    # Функция активации tanh
    self.tanh = nn.Tanh()

  def forward(
      self,
      lstm_outputs: torch.Tensor, # размерность: BATCH_SIZE x SEQ_LEN x HIDDEN_SIZE
      final_hidden: torch.Tensor  # размерность: BATCH_SIZE x HIDDEN_SIZE
      ) -> Tuple[torch.Tensor, torch.Tensor]:

      # Преобразуем выходы LSTM для всех временных шагов в keys
      keys = self.W_k(lstm_outputs) # shape (batch_size, seq_len, hidden_size)

      # Преобразуем последнее скрытое состояние в query и добавляем размерность для seq_len=1 (broadcasting)
      query = self.W_q(final_hidden).unsqueeze(1)  # shape (batch_size, 1, hidden_size)

      # Складываем keys и query с применением broadcasting (для каждого временного шага добавляется query)
      combined = keys + query   # broadcasting сложение по seq_len (batch_size, seq_len, hidden_size)

      # Применяем нелинейность tanh
      combined = self.tanh(combined) # (batch_size, seq_len, hidden_size)

      # Вычисляем оценки внимания с помощью линейного слоя (получается тензор размером (batch_size, seq_len, 1))
      vector = self.W_v(combined) # (batch_size, seq_len, 1)

      # Преобразуем scores в вероятности с помощью softmax по последовательности
      att_weights = F.softmax(vector, dim=1) # (batch_size, seq_len, 1)

      # Используем веса внимания для взвешивания исходных выходов LSTM
      # Перемножаем транспонированные веса (batch_size, 1, seq_len) и lstm_outputs (batch_size, seq_len, hidden_size)
      context = torch.bmm(att_weights.transpose(1, 2),  lstm_outputs)  # (batch_size, 1, hidden_size)
      context = context.squeeze(1)  # (batch_size, hidden_size)

      return context, att_weights

class LSTMWord2VecBahdanauAttention(nn.Module):
  def __init__(self) -> None:
    super().__init__()

    # Используем предобученный эмбеддинг с шага №2 Word2Vec для преобразования входных токенов в плотные векторные представления нужной размерности (EMBEDDING_DIM)
    self.embedding = embedding_layer
    # Создаем LSTM слой
    self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_SIZE, batch_first=True) # batch_first=True означает, что входной тензор имеет форму (batch_size, seq_len, embedding_dim)
    # Используем созданный механизм внимания
    self.attn = BahdanauAttention(HIDDEN_SIZE)
    # Создаем классификатор
    self.clf = nn.Sequential(
        nn.Linear(HIDDEN_SIZE, 128),
        nn.Dropout(),
        nn.Tanh(),
        nn.Linear(128, 1)
    )

  def forward(self, x):
    embeddings = self.embedding(x) # пропускаем вход x (индексы слов) через эмбеддинг
    outputs, (h_n, _) = self.lstm(embeddings) # последнее скрытое состояние (h_n, _),
    att_hidden, att_weights = self.attn(outputs, h_n.squeeze(0)) # получаем агрегированное скрытое состояние и веса внимания
    out = self.clf(att_hidden) # передаем агрегированное состояние в классификатор
    return out, att_weights
