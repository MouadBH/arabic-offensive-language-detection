import numpy as np
import io

from keras.layers import Embedding, Dense, LSTM, Dropout, Input
#from keras.layers import GRU, MaxPooling1D, Conv1D, Flatten
from keras.preprocessing import text, sequence
from keras.models import Model
from keras.utils import np_utils

from gensim.models.keyedvectors import KeyedVectors

import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def get_embedding_matrix(word_index, embedding_index, vocab_dim):
    print('Building embedding matrix...')
    embedding_matrix = np.zeros((len(word_index) + 1, vocab_dim))
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index.get_vector(word)
        except:
            pass
    print('Embedding matrix built.')        
    return embedding_matrix


def get_model_first(embedding_weights, word_index, vocab_dim, max_length, print_summary=True):
    inp = Input(shape=(max_length,))
    model = Embedding(input_dim=len(word_index)+1,
                      output_dim=vocab_dim,
                      trainable=False,
                      weights=[embedding_weights])(inp)

    model = LSTM(vocab_dim, return_sequences=True)(model)
    model = Dropout(0.2)(model)
    model = LSTM(vocab_dim, return_sequences=False)(model)
    model = Dropout(0.1)(model)
    model = Dense(int(vocab_dim/10), activation='relu')(model)
    model = Dropout(0.1)(model)
    model = Dense(5, activation='softmax')(model)
    model = Model(inputs=inp, outputs=model)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    if print_summary:
        model.summary()
    return model

def train_fit_predict(model, x_train, x_test, y_train, batch_size, epochs):
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs, verbose=1)
    score = model.predict(x_test)
    return history, score, model

def get_init_parameters(path, ext=None):
    if ext == 'vec':
        word_model = KeyedVectors.load_word2vec_format(path).wv
    else:
        word_model = KeyedVectors.load(path).wv
    n_words = len(word_model.vocab)
    vocab_dim = word_model[word_model.index2word[0]].shape[0]
    index_dict = dict()
    for i in range(n_words):
        index_dict[word_model.index2word[i]] = i+1
    return word_model, index_dict, n_words, vocab_dim

def get_max_length(text_data, return_line=False):
    max_length = 0
    long_line = ""
    for line in text_data:
        new = len(line.split())
        if new > max_length:
            max_length = new
            long_line = line
    if return_line:
        return long_line, max_length
    else:
        return max_length
    
def split_datasets(dataframe, test_size, header=True, seed=42):
    x = dataframe.Comment.to_list()
    y = dataframe.is_off.to_list()
    max_length = get_max_length(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)
    print('Dataset splited.')
    return x_train, x_test, y_train, y_test, max_length

def get_train_test(train_raw_text, test_raw_text, n_words, max_length):
    tokenizer = text.Tokenizer(num_words=n_words)
    tokenizer.fit_on_texts(list(train_raw_text))
    word_index = tokenizer.word_index
    train_tokenized = tokenizer.texts_to_sequences(train_raw_text)
    test_tokenized = tokenizer.texts_to_sequences(test_raw_text)
    return sequence.pad_sequences(train_tokenized, maxlen=max_length, padding='post', truncating='post'),\
           sequence.pad_sequences(test_tokenized, maxlen=max_length, padding='post', truncating='post'),\
           word_index

def class_str_2_ind(x_train, x_test, y_train, y_test, classes, n_words, max_length):
    print('Converting data to trainable form...')
    y_encoder = preprocessing.LabelEncoder()
    y_encoder.fit(classes)
    y_train = y_encoder.transform(y_train)
    y_test = y_encoder.transform(y_test)
    train_y_cat = np_utils.to_categorical(y_train, len(classes))
    x_vec_train, x_vec_test, word_index = get_train_test(x_train, x_test, n_words, max_length)
    print('Number of training examples: ' + str(len(x_vec_train)))
    print('Number of testing examples: ' + str(len(x_vec_test)))
    return x_vec_train, x_vec_test, y_train, y_test, train_y_cat, word_index

def get_main_model(word_index, WORD_MODEL, EMBED_SIZE, MAX_TEXT_LENGTH):
    tmp = get_embedding_matrix(word_index, WORD_MODEL, EMBED_SIZE)
    model = get_model(tmp, word_index, EMBED_SIZE, MAX_TEXT_LENGTH, print_summary=True)
    return model

def get_embedding_vectors(vectors, index_dict, n_words, vocab_dim):
    embedding_weights = np.zeros((n_words+1, vocab_dim)) 
    for word, index in index_dict.items():
        embedding_weights[index, :] = vectors[word]
    return embedding_weights

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data


# ==========Deprecated Code===========


def error_on_batch(res, y_test):
    y_pred = [np.argmax(tmp) for tmp in res]
    y_real = [np.argmax(tmp) for tmp in y_test]
    true = [1 for i, j in zip(y_pred, y_real) if i == j]
    err = len(true)/len(y_pred)
    return err

def convert_data(data, vocab, index):
    new = []
    for word in data.split():
        if word in vocab:
            new.append(index[word])
        else:
            new.append(index[''])
    return new