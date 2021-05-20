import pathlib
import pandas
import click


def transform_training_data(training_file):
  """
  this function takes the incoming data set extracts the word and the tag for 
  anago and splits the data into training, testing and validation
  """
  training_data = pandas.read_csv(training_file, delimiter='\t')
  all_sents = len(training_data.SENT_ID.unique())
  half_sents = all_sents / 2
  rest_sents = (half_sents + all_sents) / 2
  train = pathlib.Path('drive/MyDrive/thesis/training_data/data_gene_new_train.tsv')
  test = pathlib.Path('drive/MyDrive/thesis/training_data/data_gene_new_test.tsv')
  valid = pathlib.Path('drive/MyDrive/thesis/training_data/data_disease_new_valid.tsv')
  f = None
  for sent, df_sent in training_data.groupby('SENT_ID'):
    words = df_sent['WORD']
    tags = df_sent['TAG']
    data = zip(words, tags)
   
    if int(df_sent['SENT_ID'].unique()[0]) < half_sents:
      f = train.open("a")
    elif int(df_sent['SENT_ID'].unique()[0]) > half_sents and int(df_sent['SENT_ID'].unique()[0]) < rest_sents:
      f = test.open("a")
    else:
      f = valid.open("a")

    for word, tag in data:
      if len(str(word).split('\t')) > 1:
        word = str(word).split('\t')[0]
        tag = str(word).split('\t')[-1]
        print(f"{word}\t{tag}")
      f.write(f"{word}\t{tag}\n")
    f.write("\n")

transform_training_data('/content/drive/MyDrive/thesis/data_gene.csv')