from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def tf_preprocessing(X, max_sentence_length = 150):
    tk = Tokenizer()
    tk.fit_on_texts(X)
    vocab_size = len(tk.word_index)
    print(f'There are {vocab_size} different words in your corpus')
    X_token = tk.texts_to_sequences(X)
    ### Pad the inputs
    X_pad = pad_sequences(X_token, dtype='float32', padding='post', maxlen = max_sentence_length)
    return X_pad, vocab_size
