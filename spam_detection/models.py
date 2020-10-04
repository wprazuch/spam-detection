from tensorflow.keras import Model, Sequential
from tensorflow.keras import layers


def simple_lstm_model(num_words=10000, max_len=100):
    model = Sequential()
    model.add(layers.Embedding(num_words, 16, input_length=max_len))
    model.add(layers.LSTM(32))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model
