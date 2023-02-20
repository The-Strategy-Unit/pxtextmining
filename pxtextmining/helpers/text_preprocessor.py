import re
import emojis
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

def text_preprocessor(text_string):
    """
    Strips punctuation, excess spaces, and metacharacters "r" and "n" from the text. Converts emojis into "__text__"
    (where "text" is the emoji name) and any NAs resulting from text preprocessing into "__notext__".
    :param str text_string: Text string that is passed from
    [sklearn.feature_extraction.text.TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
    :return: text_string : Cleaned text string.
    :rtype: str
    """

    text_string = str(text_string)
    text_string = emojis.decode(text_string)
    pattern = "\:(.*?)\:"  # Decoded emojis are enclosed inside ":", e.g. ":blush:"
    pattern_search = re.search(pattern, text_string)

    # We want to tell the model that words inside ":" are decoded emojis.
    # However, "[^\w]" removes ":". It doesn't remove "_" or "__" though, so we may enclose decoded emojis
    # inside "__" instead.
    if pattern_search is not None:
        emoji_decoded = pattern_search.group(1)
        """if keep_emojis:
            text_string = re.sub(pattern, "__" + emoji_decoded + "__", text_string)
            # Sometimes emojis are consecutive e.g. ❤❤ is encoded into __heart____heart__. Split them.
            text_string = re.sub("____", "__ __", text_string)
        else:
            text_string = re.sub(pattern, "", text_string)"""
        text_string = re.sub(pattern, "__" + emoji_decoded + "__", text_string)
        # Sometimes emojis are consecutive e.g. ❤❤ is encoded into __heart____heart__. Split them.
        text_string = re.sub("____", "__ __", text_string)

    # Remove non-alphanumeric characters
    text_string = re.sub("[^\w]", " ", text_string)

    # Remove excess whitespaces
    text_string = re.sub(" +", " ", text_string)
    text_string = text_string.rstrip() # Removes trailing spaces.
    # text_string = " ".join(text.splitlines())

    if str(text_string) in ("nan", "None", " "):
        text_string = "__notext__"

    return text_string
