from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Model, Input
from keras.optimizers import Adam
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
from keras.utils import to_categorical
from keras_contrib.metrics import crf_marginal_accuracy
from keras_contrib.losses import crf_loss
from keras.preprocessing.sequence import pad_sequences
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

from utils import PullSents, predict_label


class LstmCRF(object):

    def __init__(self, data_path):
        """ initialize data details """
        # reading input data and
        data = pd.read_csv(data_path, delimiter='\t')
        self.data = data.fillna(method="ffill")
        self.words = list(set(data["WORD"].values))
        self.words.append("</ENDPAD/>")

        self.n_words = len(self.words)
        self.tags = list(set(data["TAG"].values))
        self.n_tags = len(self.tags)

        getter = PullSents(self.data)
        self.sentences = getter.sentences

        self.padded_len = 50

        # setup an index of words to tags
        self.word_index = {w: i + 1 for i, w in enumerate(self.words)}
        self.index_word = {v: k for k, v in self.word_index.items()}
        self.tag_index = {t: i for i, t in enumerate(self.tags)}
        self.index_tag = {i: w for w, i in self.tag_index.items()}

    def update_data(self):
        """
        helper function to update the data for padding
        so that we can make all the inputs the same length
        :return: padded_x, padded_y
        """

        X = [[self.word_index[w[0]] for w in s] for s in self.sentences]
        padded_x = pad_sequences(sequences=X, maxlen=self.padded_len, padding="post", value=self.n_words - 1)
        y = [[self.tag_index[w[2]] for w in s] for s in self.sentences]
        y = pad_sequences(sequences=y, maxlen=self.padded_len, padding="post", value=self.tag_index["O"])
        padded_y = [to_categorical(i, num_classes=self.n_tags) for i in y]
        return padded_x, padded_y

    def create_model(self, x_inputs, y_inputs):
        """
        function to create the model
        :return:
        """
        input = Input(shape=(self.padded_len,))
        embedding_layer = Embedding(input_dim=self.n_words + 1, output_dim=20,
                          input_length=self.padded_len)(input)
        bidirectiona_lstm_layer = Bidirectional(LSTM(units=50, return_sequences=True,
                                   recurrent_dropout=0.2))(embedding_layer)
        dense = TimeDistributed(Dense(50, activation="relu"))(bidirectiona_lstm_layer)
        crf = CRF(self.n_tags)
        out = crf(dense)
        model = Model(input, out)
        epochs = 15
        model.compile(optimizer="adam", loss=crf_loss, metrics=[crf_marginal_accuracy])
        history = model.fit(x_inputs, np.array(y_inputs), batch_size=32, epochs=epochs,
                            validation_split=0.1, verbose=1)
        history_pd = pd.DataFrame(history.history)
        model.save('/content/drive/MyDrive/thesis/lstm_crf_disease_model')
        return history_pd, model


if __name__== "__main__":
    # load the model and update the data
    data = '/content/drive/MyDrive/thesis/data.csv'
    lstm_crf_model = LstmCRF(data)
    updated_X, updated_Y = lstm_crf_model.update_data()
    x_train, x_test, y_train, y_test = train_test_split(updated_X, updated_Y, test_size=0.2, random_state=1)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1)
    history, model = lstm_crf_model.create_model(x_train, y_train)

    # print the results during training
    print(history)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(history["crf_marginal_accuracy"], label="Accuracy")
    plt.ylabel('Accuracy')
    plt.xlabel('Time')
    plt.savefig('BiLSTM_GENE.eps')

    print("Results of the model on test")

    # show results of the model
    test_pred = model.predict(x_test, verbose=1)
    test_labels = predict_label(y_test, lstm_crf_model.index_tag)
    pred_labels = predict_label(test_pred, lstm_crf_model.index_tag)
    print(classification_report(test_labels, pred_labels))
    
    
    print("Results of the model on validation")
    val_pred = model.predict(x_val, verbose=1)
    val_labels = predict_label(y_val, lstm_crf_model.index_tag)
    pred_val_labels = predict_label(val_pred, lstm_crf_model.index_tag)
    print(classification_report(val_labels, pred_val_labels))

    results = zip(x_val, val_labels, pred_val_labels)
    out_path = Path('gene_output.txt')
    out_path.touch(exist_ok=True)
    with out_path.open(mode='w') as f:
      for item in results:
        (value, label, predicted_label) = item
        if label != predicted_label:
          for line in range(len(predicted_label)):
            f.write(f"{lstm_crf_model.index_word.get(value[line])} \t {label[line]} \t{predicted_label[line]}\n")





