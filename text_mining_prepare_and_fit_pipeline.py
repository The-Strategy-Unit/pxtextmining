"""
================================================================================
Classification of patient feedback using algoritmhs that can efficiently
handle sparse matrices
================================================================================

Classify documents by label using a bag-of-words approach.

See example in https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html?highlight=text%20classification%20sparse

"""

##############################################################################
# Load libraries and data
# ------------------------------------
exec(open("./text_mining_import_libraries.py").read())
exec(open("./text_mining_load_and_prepare_data.py").read())

#############################################################################
# Define learners that can handle sparse matrices
# ------------------------------------
learners = [RidgeClassifier(),
            LinearSVC(max_iter=10000),
            SGDClassifier(),
            Perceptron(),
            PassiveAggressiveClassifier(),
            BernoulliNB(),
            ComplementNB(),
            MultinomialNB(),
            KNeighborsClassifier(),
            NearestCentroid(),
            RandomForestClassifier()
            ]

#learners = [LinearSVC(), MultinomialNB()] # Uncomment this for quick & dirty experimentation

#############################################################################
# NLTK-based function for lemmatizing. Will be passed in CountVectorizer()
# ------------------------------------
# https://scikit-learn.org/stable/modules/feature_extraction.html?highlight=stemming
class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

#############################################################################
# Function for automating pipeline for > 1 learners
# ------------------------------------
# https://stackoverflow.com/questions/48507651/multiple-classification-models-in-a-scikit-pipeline-python
class ClfSwitcher(BaseEstimator):
    def __init__(
                 self,
                 estimator=SGDClassifier(),
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
# Prepare pipeline
# ------------------------------------
# Preprocess numerical and text data in different ways.
# Numerical data are min-max normalized because X^2 and Multinomial Naive 
# Bayes can't handle negative data. NOTE: FIND OUT IF IT IS METHODOLOGICALLY 
# SOUND TO DO THIS!
# Pipeline for numeric features
numeric_features = ['comment_polarity']
numeric_transformer = Pipeline(steps=[
    ('minmax', MinMaxScaler())])

# Pipeline for text features
text_features = 'improve' # Needs to be a scalar, otherwise TfidfVectorizer() throws an error
text_transformer = Pipeline(steps=[
    ('tfidf', (TfidfVectorizer(max_df=0.95)))]) # https://kavita-ganesan.com/tfidftransformer-tfidfvectorizer-usage-differences/

# Pass both pipelines/preprocessors to a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('text', text_transformer, text_features)])

# Pipeline with preprocessors, any other operations and a learner
pipe = Pipeline(steps=[('preprocessor', preprocessor),
                       ('kbest', SelectKBest(chi2)),
                      #('rfecv', RFECV(DecisionTreeClassifier(), cv=5)),
                      ('clf', ClfSwitcher())])

# Parameter grid
param_grid_preproc = {
    'clf__estimator': None,
    'preprocessor__text__tfidf__ngram_range': ((1, 1), (2, 2), (1, 3)),
    'preprocessor__text__tfidf__tokenizer': [LemmaTokenizer(), None],
    'preprocessor__text__tfidf__use_idf': [True, False],
    'kbest__k': (5000, 10000),
    #'kbest__k': (np.array([15, 25, 50]) * X_train.shape[0] / 100).astype(int),
    #'rfecv__estimator': [DecisionTreeClassifier()],
    #'rfecv__step': (0.1, 0.25, 0.5) # Has a scoring argument too. Investigate
}

param_grid = []
for i in learners:
    aux = param_grid_preproc.copy()
    aux['clf__estimator'] = [i]
    if i.__class__.__name__ == LinearSVC().__class__.__name__:
        aux['clf__estimator__class_weight'] = [None, 'balanced']
        #aux['clf__estimator__dual'] = [True, False] # https://stackoverflow.com/questions/52670012/convergencewarning-liblinear-failed-to-converge-increase-the-number-of-iterati
    if i.__class__.__name__ == BernoulliNB().__class__.__name__:
        aux['clf__estimator__alpha'] = (0.1, 0.5, 1)
    if i.__class__.__name__ == ComplementNB().__class__.__name__:
        aux['clf__estimator__alpha'] = (0.1, 0.5, 1)
    if i.__class__.__name__ == MultinomialNB().__class__.__name__:
        aux['clf__estimator__alpha'] = (0.1, 0.5, 1)
    if i.__class__.__name__ == SGDClassifier().__class__.__name__:
        aux['clf__estimator__penalty'] = ('l2', 'elasticnet')
    if i.__class__.__name__ == RandomForestClassifier().__class__.__name__:
        aux['clf__estimator__max_features'] = ('sqrt', 0.666)
    param_grid.append(aux)

#############################################################################
# Benchmark classifiers
# ------------------------------------
# Set a scoring measure (other than accuracy) and train several 
# classification models.

# Set Matthews Correlation Coefficient as the scoring measure. Looks promising
# https://towardsdatascience.com/the-best-classification-metric-youve-never-heard-of-the-matthews-correlation-coefficient-3bf50a2f3e9a
# https://towardsdatascience.com/matthews-correlation-coefficient-when-to-use-it-and-when-to-avoid-it-310b3c923f7e
# https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0177678&type=printable
matthews_corrcoef_scorer = make_scorer(matthews_corrcoef)

# Grid search with 5-fold cross-validation
gscv = GridSearchCV(pipe, param_grid, n_jobs=-1, return_train_score=False, 
                    verbose=3, scoring=matthews_corrcoef_scorer)
gscv.fit(X_train, y_train)

#############################################################################
# Save model to disk
# ------------------------------------
# https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
filename = 'finalized_model_4444.sav'
pickle.dump(gscv, open(filename, 'wb'))