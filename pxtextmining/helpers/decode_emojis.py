import re
import emojis


def decode_emojis(text_string):
    """
    Converts emojis into " __text__ " (where "text" is the emoji name)

    :param str text_string: Text string
    :return: text_string : Text string with decoded emojis
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
        text_string = re.sub(pattern, " __" + emoji_decoded + "__ ", text_string)
    return text_string


# if __name__ == '__main__':
#     test_text = 'testing it out🐻🌻emoji decoder'
#     print(decode_emojis(test_text))
