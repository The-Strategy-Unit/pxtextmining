#############################################################################
# NLTK/spaCy-based function for lemmatizing
# ------------------------------------
# https://scikit-learn.org/stable/modules/feature_extraction.html?highlight=stemming
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
nlp = spacy.load("en_core_web_sm")


class LemmaTokenizer:
    def __init__(self, tknz='wordnet'):
        self.tknz = tknz
    def __call__(self, doc):
        if self.tknz == 'wordnet':
            wln = WordNetLemmatizer()
            return [wln.lemmatize(t) for t in word_tokenize(doc)]
        if self.tknz == 'spacy':
            return [t.lemma_ for t in nlp(doc,
                                          disable=["tagger", "parser", "ner"])]