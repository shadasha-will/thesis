from typing import List

from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, TokenEmbeddings, TransformerWordEmbeddings, CharacterEmbeddings, ELMoEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.visual.training_curves import Plotter


columns = {0: 'text', 1: 'ner'}
data_folder = '/content/drive/MyDrive/thesis/training_data/'

corpus: Corpus = ColumnCorpus(data_folder, columns, 
                              train_file = 'data_gene_new_valid.tsv',
                              test_file = 'data_gene_new_test.tsv',
                              dev_file = 'data_gene_new_train.tsv')

tag_type = 'ner'
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
embedding_types : List[TokenEmbeddings] = [WordEmbeddings('glove'),
                                           CharacterEmbeddings(),
                                           TransformerWordEmbeddings('dmis-lab/biobert-v1.1')]

embeddings : StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
tagger : SequenceTagger = SequenceTagger(hidden_size=256, embeddings=embeddings,
                                         tag_dictionary=tag_dictionary,
                                         tag_type=tag_type,
                                         use_crf=True)
trainer : ModelTrainer = ModelTrainer(tagger, corpus)
trainer.train('/content/drive/MyDrive/thesis/resources/taggers/ner_gene_biobert', learning_rate=0.1, mini_batch_size=32,
              max_epochs=20, embeddings_storage_mode='none')
