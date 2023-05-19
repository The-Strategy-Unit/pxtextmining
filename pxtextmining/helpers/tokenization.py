import spacy

# load outside of the function otherwise it will load every time
try:
    en_nlp = spacy.load('en_core_web_lg', disable=["parser", "ner"])
except OSError:
    print('Warning! Have you downloaded the spacy model? Run " python -m spacy download en_core_web_lg " in your terminal')

def spacy_tokenizer(document):
    """Enables use of spacy tokenizer in the sklearn pipeline.

    Args:
        document (str): Text to be tokenized

    Returns:
        (list): List containing tokenized, lemmatized words from input text.
    """
    doc_spacy = en_nlp(document)
    return [token.lemma_ for token in doc_spacy]
