""" helper function """
import numpy as np

class PullSents(object):
    """
    function to seperate sentences and split the components for training
    """
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        mapper = lambda sentence: [(word, pos, t) for word, pos, t in zip(sentence["WORD"].values.tolist(),
                                                                sentence["POS"].values.tolist(),
                                                                sentence["TAG"].values.tolist())]
        self.grouped = self.data.groupby("SENT_ID").apply(mapper)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

def predict_label(prediction, index_dict):
    """
    function to predict values of a model
    :param prediction: list of model outputs given an input
    :param index_dict: dictionary mapping index to tag
    :return:
    """
    out = []
    for pred_i in prediction:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(index_dict[p_i].replace("PAD", "O"))
        out.append(out_i)
    return out