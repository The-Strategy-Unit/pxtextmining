from factories import factory_data_prepros, factory_pipeline


def run_text_classification_pipeline(filename, target, predictor, test_size=0.33,
                                     tknz="spacy", fit_pipeline=True,
                                     metric="class_balance_accuracy",
                                     cv=5, n_iter=100, n_jobs=5, verbose=3
                                     ):

    x_train, y_test, y_train, y_test = factory_data_prepros(filename, target, predictor, test_size)

    result = factory_pipeline(x_train, y_train, tknz=tknz,
                     metric=metric,
                     cv=cv, n_iter=n_iter, n_jobs=n_jobs, verbose=verbose)

    return result