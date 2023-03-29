import spacy

def spacy_tokenizer(document):
    """Enables use of spacy tokenizer in the sklearn pipeline.

    Args:
        document (str): Text to be tokenized

    Returns:
        (list): List containing tokenized, lemmatized words from input text.
    """
    try:
        en_nlp = spacy.load('en_core_web_lg', disable=["parser", "ner"])# Don't put this inside the function- loading it in every CV iteration would tremendously slow down the pipeline.
    except OSError:
        print('Warning! Have you downloaded the spacy model? Run " python -m spacy download en_core_web_lg " in your terminal')
    doc_spacy = en_nlp(document)
    return [token.lemma_ for token in doc_spacy]
