import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing import text, sequence
from gensim.models.keyedvectors import KeyedVectors

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

def get_embedding_matrix(word_index, embedding_index, vocab_dim):
    """
    words not found in embedding index will be all-zeros.
    """
    print('Building embedding matrix...')
    words_not_found = []
    embedding_matrix = np.zeros((len(word_index) + 1, vocab_dim))
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index.get_vector(word)
        except:
            words_not_found.append(word)
            pass
    print('Embedding matrix built.')        
    return embedding_matrix, words_not_found


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
    x_vec_train, x_vec_test, word_index = get_train_test(x_train, x_test, n_words, max_length)
    print('Number of training examples: ' + str(len(x_vec_train)))
    print('Number of testing examples: ' + str(len(x_vec_test)))
    return x_vec_train, x_vec_test, y_train, y_test, word_index

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

# =====================

def train_fit(model, x_train, y_train, batch_size, epochs):
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.2)
    return history, model

def evaluate_model(model, x_train, y_train, x_test, y_test):
    loss, accuracy = model.evaluate(x_train, y_train)
    print("Training Accuracy: {:.4f}".format(accuracy))
    print("Training Loss: {:.4f}".format(loss))
    loss_val, accuracy_val = model.evaluate(x_test, y_test, verbose=True)
    print("Testing Accuracy:  {:.4f}".format(accuracy_val))
    print("Testing Loss:  {:.4f}".format(loss_val))
    
    return accuracy, accuracy_val
    
def get_prediction(model, x_test):
    y_pred = model.predict(x_test, verbose=1)
    y_pred_bool = np.argmax(y_pred, axis=1)
    y_pred = (y_pred > 0.5)
#     print(classification_report(y_test, y_pred))
    return y_pred
    
def get_optimal_epoch(history):
    n = np.argmin(history.history['val_loss'])
    print("Optimal epoch : {}".format(n))
    print("Accuracy on train : {} %".format(np.round(history.history['accuracy'][n]*100, 2)))
    print("Accuracy on test : {} %".format(np.round(history.history['val_accuracy'][n]*100, 2)))
    print("Loss on train : {}".format(np.round(history.history['loss'][n]*100, 2)))
    print("Loss on test : {}".format(np.round(history.history['val_loss'][n]*100, 2)))
    return n
    
def show_train_history(history,train,validation, n):
    plt.figure(figsize=(16, 8))
    plt.plot(range(1, len(history.history[train])+1), history.history[train], label="train")
    plt.plot(range(1, len(history.history[validation])+1), history.history[validation], label="validation")
    plt.plot(n+1,history.history[validation][n],"r*", label="Opt.")
    plt.legend()
    plt.title(str(train) + " Curve")
    plt.ylabel(train)
    plt.xlabel("epochs")
    plt.show()

def save_model(model, folder, name, acc):
    with open('./models/' + folder + '/' + name + '_acc_' + str(acc) + '.json', 'w') as f:
        f.write(model.to_json())
        f.close()

    model.save_weights('./models/' + folder + '/' + name + '_weights_acc_' + str(acc) + '.h5')

    model.save('./models/' + folder + '/' + name + '_acc_' + str(acc) + '.h5')
    
    print("Model Saved Successfully in ./models/" + folder + "/ as " + name + '_acc_' + str(acc))
    
def model_analyse(model, x_test, X_test, y_test):
    pred_test = model.predict(x_test, verbose=True)
    df_blind = pd.DataFrame({'REAL': y_test, 
                             'PRED': pred_test.reshape(pred_test.shape[0],), 
                             'TEXT': X_test})
    df_blind = df_blind.reset_index()[['REAL', 'PRED', 'TEXT']]
    df_blind.PRED = df_blind.PRED.round()
    error_records = df_blind[df_blind.REAL != df_blind.PRED]
    
    print("Number of misclassified reviews: {} out of {}".format(error_records.shape[0], df_blind.shape[0]))
    print("Blind Test Accuracy:  {:.4f}".format(accuracy_score(df_blind.REAL, df_blind.PRED)))
    return df_blind