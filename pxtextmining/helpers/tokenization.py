from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy

try:
    nlp = spacy.load("en_core_web_sm") # Don't put this inside the function- loading it in every CV iteration would tremendously slow down the pipeline.
except OSError:
    print('Warning! Have you downloaded the spacy models? Run " python -m spacy download en_core_web_sm " in your terminal')


class LemmaTokenizer:
    """
    Class for custom lemmatization in
    [sklearn.feature_extraction.text.TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
    (see [this](https://scikit-learn.org/stable/modules/feature_extraction.html?highlight=stemming)).
    Uses [spaCy](https://spacy.io/) (``tknz == 'spacy'``) or [NLTK](https://www.nltk.org/) (``tknz == 'wordnet'``).
    """

    def __init__(self, tknz='wordnet'):
        self.tknz = tknz

    def __call__(self, doc):
        if self.tknz == 'wordnet':
            wln = WordNetLemmatizer()
            return [wln.lemmatize(t) for t in word_tokenize(doc)]
        if self.tknz == 'spacy':
            return [t.lemma_ for t in nlp(doc,
                                          disable=["tagger", "parser", "ner"])]
