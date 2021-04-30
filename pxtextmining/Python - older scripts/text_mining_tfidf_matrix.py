# https://stackoverflow.com/questions/34449127/sklearn-tfidf-transformer-how-to-get-tf-idf-values-of-given-words-in-documen

gscv = joblib.load('finalized_model.sav')

tfidf = gscv.best_estimator_.named_steps['preprocessor'].named_transformers_['text'].named_steps['tfidf']

tfidf_matrix = tfidf.fit_transform(X_test.improve)
feature_names = tfidf.get_feature_names()
tfidf_matrix = pd.DataFrame(tfidf_matrix.toarray(), columns = tfidf.get_feature_names()) #.melt(value_name='tfidf')