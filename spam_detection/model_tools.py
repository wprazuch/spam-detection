import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras import preprocessing as t_preprocessing
from tensorflow.keras import Model, Sequential
from tensorflow.keras import layers

import argparse
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def load_data(sms_data_path=r'datasets/sms-spam-collection/spam.csv',
              mail_data_path=r'datasets\mail-spam-collection\spam_ham_dataset.csv'):
    sms_df = pd.read_csv(sms_data_path, engine='python')
    mail_spam_df = pd.read_csv(mail_data_path)

    x_sms, y_sms = sms_df['v2'].values, sms_df['v1'].values
    x_mail, y_mail = mail_spam_df['text'].values, mail_spam_df['label_num'].values

    y_sms[y_sms == 'ham'] = 0
    y_sms[y_sms == 'spam'] = 1

    X, y = np.hstack([x_sms, x_mail]), np.hstack([y_sms, y_mail])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=1)

    return X_train, X_test, y_train.astype(np.float32), y_test.astype(np.float32)


def tokenize_data(data, fit=False):
    if fit == True:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(data)

        with open('tokenizer.pkl', 'wb') as output:
            pickle.dump(tokenizer, output, pickle.HIGHEST_PROTOCOL)

    else:
        try:
            with open('tokenizer.pkl', 'rb') as tokenizer_object:
                tokenizer = pickle.load(tokenizer_object)
        except:
            logging.error("No tokenized object found!")

    data_tokenized = tokenizer.texts_to_sequences(data)

    return data_tokenized


def preprocess_input(data):
    print(data)
    data = tokenize_data(data)
    data = pad_sequences(data, 100)

    return data


def get_num_words():
    try:
        with open('tokenizer.pkl', 'rb') as tokenizer_object:
            tokenizer = pickle.load(tokenizer_object)
    except:
        logging.error("No tokenized object found!")

    return len(tokenizer.index_word.keys())+1


def pad_sequences(data, max_len):
    data = t_preprocessing.sequence.pad_sequences(data, max_len)
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    return data
