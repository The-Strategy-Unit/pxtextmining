#############################################################################
# NLTK/spaCy-based function for lemmatizing
# ------------------------------------
# https://scikit-learn.org/stable/modules/feature_extraction.html?highlight=stemming
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

#############################################################################
# Function for automating pipeline for > 1 learners
# ------------------------------------
# https://stackoverflow.com/questions/48507651/multiple-classification-models-in-a-scikit-pipeline-python
class ClfSwitcher(BaseEstimator):
    def __init__(
                 self,
                 estimator=SGDClassifier(max_iter=10000),
    ): self.estimator = estimator

    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        return self

    def predict(self, X, y=None):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def score(self, X, y):
        return self.estimator.score(X, y)

#############################################################################
# Upbalance
# ------------------------------------
def RandomOverSamplerDictionary(y, threshold=200):
    unique, frequency = np.unique(y, return_counts=True)
    rare_classes = pd.DataFrame()
    rare_classes['counts'], rare_classes.index = frequency, unique
    if len(rare_classes[rare_classes.counts < threshold]) == 0:
        rare_classes = rare_classes.to_dict()['counts']
    else:
        rare_classes = rare_classes[rare_classes.counts < threshold]
        # rare_classes.counts = (100, 100, 100, 100, 100, 100, 100)
        rare_classes.counts = threshold
        rare_classes = rare_classes.to_dict()['counts']
    return(rare_classes)

#############################################################################
# Create Class Balance Accuracy scorer
# ------------------------------------
# See p. 40 in https://lib.dr.iastate.edu/cgi/viewcontent.cgi?article=4544&context=etd
def class_balance_accuracy_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    c_i_dot = np.sum(cm, axis=1)
    c_dot_i = np.sum(cm, axis=0)
    cba = []
    for i in range(len(c_dot_i)):
        cba.append(cm[i][i] / max(c_i_dot[i], c_dot_i[i]))
    cba = sum(cba) / (i + 1)
    return cba