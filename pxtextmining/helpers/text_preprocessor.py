import re
import emojis


def text_preprocessor(text_string):
    """
    Strips punctuation, excess spaces, and metacharacters "r" and "n" from the text. Converts emojis into "__text__"
    (where "text" is the emoji name) and any NAs resulting from text preprocessing into "__notext__".

    :param str text_string: Text string that is passed from
        `sklearn.feature_extraction.text.TfidfVectorizer <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html>`_.
    :return: text_string (str): Cleaned text string.
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
