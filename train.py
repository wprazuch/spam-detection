from spam_detection.models import simple_lstm_model
from spam_detection.model_tools import load_data, tokenize_data, pad_sequences, get_num_words

import argparse
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Arguments for training an NLP model to detect spam.')
    parser.add_argument('--MAXLEN', default=100, type=int, help='Maximum length of sequence data')
    parser.add_argument('--save_path', type=str, default='models/spam_model',
                        help='Output path for Tensorflow model')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    MAXLEN = args.MAXLEN
    X_train, X_test, y_train, y_test = load_data()

    X_train = tokenize_data(X_train, fit=True)
    X_test = tokenize_data(X_test, fit=False)

    X_train = pad_sequences(X_train, MAXLEN)
    X_test = pad_sequences(X_test, MAXLEN)

    NUM_WORDS = get_num_words()

    model = simple_lstm_model(num_words=NUM_WORDS, max_len=MAXLEN)

    logging.info("Starting to train the model")
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    logging.info("Evaluating the model on test data")
    model.evaluate(X_test, y_test)

    logging.info(f"Saving the model in {args.save_path}")
    model.save(args.save_path)


if __name__ == '__main__':
    main()
