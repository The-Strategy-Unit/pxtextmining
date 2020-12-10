# https://medium.com/@aneesha/visualising-top-features-in-linear-svm-with-scikit-learn-and-matplotlib-3454ab18a14d

gscv = joblib.load('finalized_model_4444.sav')

# Extract best estimator and replace ClfSwitcher() with it in the pipeline, otherwise some functions may not work.
# For example, metrics.plot_confusion_matrix() expects to find ('clf', BEST_ESTIMATOR()), but instead it would find
# ('clf', ClfSwitcher(estimator=BEST_ESTIMATOR())), causing it to throw an error.
aux = pd.DataFrame(gscv.best_params_.items())
best_estimator = aux[aux[0] == 'clf__estimator'].reset_index()[1][0]
estimator_position = len(gscv.best_estimator_) - 1
gscv.best_estimator_.steps.pop(estimator_position)
gscv.best_estimator_.steps.append(('clf', best_estimator))
gscv.best_estimator_.fit(X_train, y_train) # Fit whole pipeline with best estimator

feature_names = gscv.best_estimator_.named_steps['preprocessor'].named_transformers_['text'].named_steps['tfidf'].get_feature_names()


import matplotlib.pyplot as plt
def plot_coefficients(classifier, feature_names, top_features=20,
                      which_class='Smoking'):
    # which_class is for one-vs-rest multiclass problems where the coef_ is
    # a table with dimensions n_classes X n_features
    #class_index = np.where(classifier.classes_ == which_class)[0]
    #class_index = np.where(classifier.best_estimator_.named_steps['clf'].estimator.classes_ == which_class)[0]
    class_index = np.where(classifier.best_estimator_.named_steps['clf'].classes_ == which_class)[0]
    #coef = classifier.best_estimator_.named_steps['clf'].coef_[class_index].ravel()
    #coef = classifier.best_estimator_.named_steps['clf'].estimator.coef_[class_index].ravel()
    coef = classifier.best_estimator_.named_steps['clf'].coef_[class_index].ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients,
                                  top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15, 5))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1, 1 + 2 * top_features),
               feature_names[top_coefficients], rotation=60, ha='right')
    plt.show()

plot_coefficients(classifier=gscv, feature_names=feature_names,
                  top_features=20, which_class='Smoking')

