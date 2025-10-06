import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

import string
import re

import numpy as np
import pickle
import torch
from torch import Tensor
import torch.nn as nn
from gensim.models import Word2Vec


# Создание эмбеддинга
def build_embedding_layer(w2v_model: Word2Vec, vocab_to_int: dict, embedding_dim: int, freeze=True):
    vocab_size = len(vocab_to_int) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for word, i in vocab_to_int.items():
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]

    embedding_layer = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=freeze)
    return embedding_layer

def save_vocab(vocab_to_int: dict, path: str):
    with open(path, 'wb') as f:
        pickle.dump(vocab_to_int, f)

def load_vocab(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_embedding_from_w2v(path_to_w2v: str, vocab_to_int_path: str, embedding_dim: int):
    wv = Word2Vec.load(path_to_w2v)
    vocab_to_int = load_vocab(vocab_to_int_path)
    return build_embedding_layer(wv, vocab_to_int, embedding_dim)



# Предобработка текста
def data_preprocessing(text: str) -> str:
    """preprocessing string: lowercase, removing html-tags, punctuation, 
                            stopwords, digits

    Args:
        text (str): input string for preprocessing

    Returns:
        str: preprocessed string
    """    

    text = text.lower()
    text = re.sub('<.*?>', '', text) # html tags
    text = ''.join([c for c in text if c not in string.punctuation])# Remove punctuation
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = ' '.join([word for word in text.split() if not word.isdigit()]) 
    return text

def get_words_by_freq(sorted_words: list[tuple[str, int]], n: int = 10) -> list:
    return list(filter(lambda x: x[1] > n, sorted_words))

def padding(review_int: list, seq_len: int) -> np.array: # type: ignore
    """Make left-sided padding for input list of tokens

    Args:
        review_int (list): input list of tokens
        seq_len (int): max length of sequence, it len(review_int[i]) > seq_len it will be trimmed, else it will be padded by zeros

    Returns:
        np.array: padded sequences
    """    
    features = np.zeros((len(review_int), seq_len), dtype = int)
    for i, review in enumerate(review_int):
        if len(review) <= seq_len:
            zeros = list(np.zeros(seq_len - len(review)))
            new = zeros + review
        else:
            new = review[: seq_len]
        features[i, :] = np.array(new)
            
    return features

def preprocess_single_string(
    input_string: str, 
    seq_len: int, 
    vocab_to_int: dict,
    verbose : bool = False
    ) -> Tensor:
    """Function for all preprocessing steps on a single string

    Args:
        input_string (str): input single string for preprocessing
        seq_len (int): max length of sequence, it len(review_int[i]) > seq_len it will be trimmed, else it will be padded by zeros
        vocab_to_int (dict, optional): word corpus {'word' : int index}. Defaults to vocab_to_int.

    Returns:
        list: preprocessed string
    """    

    preprocessed_string = data_preprocessing(input_string)
    result_list = []
    for word in preprocessed_string.split():
        try: 
            result_list.append(vocab_to_int[word])
        except KeyError as e:
            if verbose:
                print(f'{e}: not in dictionary!')
            pass
    result_padded = padding([result_list], seq_len)[0]

    return Tensor(result_padded)