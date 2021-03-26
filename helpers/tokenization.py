from nltk import word_tokenize
# from nltk.stem import WordNetLemmatizer
# import spacy
# nlp = spacy.load("en_core_web_sm")


class LemmaTokenizer:
    """
    NLTK/spaCy-based class for lemmatizing
    https://scikit-learn.org/stable/modules/feature_extraction.html?highlight=stemming
    """

    def __init__(self, lang_model, tknz='wordnet'):
        self.lang_model = lang_model
        self.tknz = tknz

    def __call__(self, doc):
        if self.tknz == 'wordnet':
            # wln = WordNetLemmatizer()
            # return [wln.lemmatize(t) for t in word_tokenize(doc)]
            return [self.lang_model.lemmatize(t) for t in word_tokenize(doc)]
        if self.tknz == 'spacy':
            return [t.lemma_ for t in self.lang_model(doc, disable=["tagger", "parser", "ner"])]