from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Model, Input
from keras.layers import Bidirectional, concatenate, SpatialDropout1D, GlobalMaxPooling1D
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Conv1D
from keras_contrib.layers import CRF
from keras.utils import to_categorical
from keras_contrib.metrics import crf_marginal_accuracy
from keras_contrib.losses import crf_loss
from keras.preprocessing.sequence import pad_sequences
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

from utils import PullSents, predict_label

class LstmChar(object):

    def __init__(self, data_path):
        """ initialize data details """
        # reading input data and
        data = pd.read_csv(data_path, delimiter='\t')
        self.data = data.fillna(method="ffill")
        self.words = list(set(self.data["WORD"].values))

        self.n_words = len(self.words)
        self.tags = list(set(self.data["TAG"].values))
        self.n_tags = len(self.tags)

        getter = PullSents(self.data)
        self.sentences = getter.sentences

        self.max_length = 75
        self.max_length_char = 10

        # setup an index of words to tags and characters
        self.word_index = {w: i + 2 for i, w in enumerate(self.words)}
        self.word_index["UNK"] = 1
        self.word_index["PAD"] = 0
        self.index_to_word = {i: w for w, i in self.word_index.items()}
        self.tag_index = {t: i + 1 for i, t in enumerate(self.tags)}
        self.tag_index["PAD"] = 0
        self.index_tag = {i: w for w, i in self.tag_index.items()}
        self.chars = set()
        self.n_chars = 0
        self.char_index = {}

    def get_chars(self):
        """ function to get all of the character representations from words """
        self.chars = set([letter for letters in self.words for letter in letters])
        self.n_chars = len(self.chars)
        self.char_index = {char: i + 2 for i, char in enumerate(self.chars)}
        self.char_index["UNK"] = 1
        self.char_index["PAD"] = 0
        all_chars = []
        for sentence in self.sentences:
            sent_seq = []
            for i in range(self.max_length):
                word_seq = []
                for j in range(self.max_length_char):
                    try:
                        word_seq.append(self.char_index.get(sentence[i][0][j]))
                    except:
                        word_seq.append(self.char_index.get("PAD"))
                sent_seq.append(word_seq)
            all_chars.append(np.array(sent_seq))
        return all_chars

    def create_model(self, x_word_train, x_char_train, y_train):
        """ creates a sequential model LSTM model """

        #representation of the input character and word embeddings
        word_input = Input(shape=(self.max_length,))
        word_embedding = Embedding(input_dim=self.n_words + 2, output_dim=20,
        input_length=self.max_length, mask_zero=True)(word_input)
        char_input = Input(shape=(self.max_length, self.max_length_char,))
        char_embedding = TimeDistributed(Embedding(input_dim=self.n_chars + 2,
                                          output_dim=10, input_length=self.max_length_char, mask_zero=True))(char_input)
        # character lstm
        char_lstm = TimeDistributed(LSTM(units=20, return_sequences=False, recurrent_dropout=0.5))(char_embedding)

        #ner lstm
        x = concatenate([word_embedding, char_lstm])
        x = SpatialDropout1D(0.3)(x)
        ner_lstm = Bidirectional(LSTM(units=50, return_sequences=True, recurrent_dropout=0.6))(x)
        output = TimeDistributed(Dense(self.n_tags+1, activation="sigmoid"))(ner_lstm)
        model = Model([word_input, char_input], output)

        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])
        model.summary()
        history = model.fit([x_word_train, np.array(x_char_train).reshape((len(x_char_train), self.max_length, self.max_length_char))],
                           np.array(y_train).reshape(len(y_train), self.max_length, 1),
                           batch_size=32, epochs=30, validation_split=0.1, verbose=1)

        history = pd.DataFrame(history.history)
        return history, model


if __name__== "__main__":
    # load the model and update the data
    lstm_char_model = LstmChar('training_data/data_disease.csv')
    train_words = [[lstm_char_model.word_index[word[0]] for word in sent] for sent in lstm_char_model.sentences]
    train_words_padded = pad_sequences(maxlen=lstm_char_model.max_length, sequences=train_words,
                                       value=lstm_char_model.word_index["PAD"], padding='post', truncating='post')
    x_characters = lstm_char_model.get_chars()
    tags = [[lstm_char_model.tag_index[word[2]] for word in sent] for sent in lstm_char_model.sentences]
    y = pad_sequences(maxlen=lstm_char_model.max_length, sequences=tags, value=lstm_char_model.tag_index["PAD"],
    padding="post", truncating="post")
    x_word_tr, x_word_test, y_train, y_test = train_test_split(train_words_padded, y, test_size=0.2, random_state=2018)
    x_word_tr, x_word_val, y_train, y_val = train_test_split(x_word_tr, y_train, test_size=0.25, random_state=2018)
    
    x_char_tr, x_char_test, y_char_train, y_char_test = train_test_split(x_characters, y, test_size=0.2, random_state=2018)
    x_char_tr, x_char_val, y_char_train, y_char_val   = train_test_split(x_char_tr, y_char_train, test_size=0.25, random_state=2018)
    
    history, model = lstm_char_model.create_model(x_word_tr, x_char_tr, y_train)


   
    # print the results during training
    print(history)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(history["acc"], label="Accuracy")
    plt.ylabel('Accuracy')
    plt.xlabel('Time')
    plt.savefig('CharLSTM_DISEASE.eps')

    # show results of the model
    y_pred = model.predict([x_word_test, np.array(x_char_test).reshape((len(x_char_test), lstm_char_model.max_length, lstm_char_model.max_length_char))])
    predicted_labels = []
    y_tags = []
    total = len(y_pred)
    for i in range(total):
      p = np.argmax(y_pred[i], axis=-1)
      for w, t, pred in zip(x_word_test[i], y_test[i], p):
        if w != 0:
          predicted_labels.append(lstm_char_model.index_tag[pred])
          y_tags.append(lstm_char_model.index_tag[t])
    
    print(classification_report(y_tags, predicted_labels))

    # run tests on validation of the model
    out_file = Path('lstm_char_output_disease.txt')
    out_file.touch()
    y_pred_val = model.predict([x_word_val, np.array(x_char_val).reshape((len(x_char_val), lstm_char_model.max_length, lstm_char_model.max_length_char))])
    predict_val_labels = []
    y_tags_val = []
    words = []
    total_val =len(y_pred_val)
    with out_file.open(mode='w') as f:
      for i in range(total_val):
        p = np.argmax(y_pred_val[i], axis=-1)
        for w, t, pred in zip(x_word_val[i], y_val[i], p):
          if w !=0:
            predict_val_labels.append(lstm_char_model.index_tag[pred])
            y_tags_val.append(lstm_char_model.index_tag[t])
            f.write(f"{lstm_char_model.index_to_word[w]}\t{lstm_char_model.index_tag[pred]}\t{lstm_char_model.index_tag[t]}\n")
    print(classification_report(y_tags_val, predict_val_labels))

