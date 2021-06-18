import pandas as pd
import joblib
from itertools import chain


def factory_predict_unlabelled_text(dataset, predictor, pipe_path,
                                    preds_column=None, column_names='__all__'):
    """
    Predict unlabelled text data using a fitted `Scikit-learn`/`imblearn.pipeline.Pipeline
    <https://imbalanced-learn.org/stable/references/generated/imblearn.pipeline.Pipeline.html#imblearn.pipeline.Pipeline>`_.

    :param dataset: A ``pandas.DataFrame`` (or an object that can be converted into such) with the text data to predict
        classes for.
    :param str predictor: The column name of the text variable.
    :param str pipe_path: A string in the form "path_to_fitted_pipeline/pipeline.sav," where "pipeline" is the name of
        the SAV file with the fitted `Scikit-learn`/`imblearn.pipeline.Pipeline
        <https://imbalanced-learn.org/stable/references/generated/imblearn.pipeline.Pipeline.html#imblearn.pipeline.Pipeline>`_.
    :param str preds_column: The user-specified name of the column that will have the predictions. If `None` (default),
        then the name will be ``text_col_name + '_preds'``.
    :param column_names:  A vector of strings with the names of the columns of the supplied data frame (incl.
        ``text_col_name``) to be added to the returned ``pandas.DataFrame``.  If ``None``, then the only column in the
        returned data frame will be ``preds_column``. Defaults to "__all__".
    :return: A ``pandas.DataFrame`` with the predictions and any other columns supplied in ``column_names``.
    """

    data_unlabelled = pd.DataFrame(dataset)

    # Rename predictor column and replace NAs with empty string.
    data_unlabelled = data_unlabelled.rename(columns={predictor: 'predictor'})
    data_unlabelled['predictor'] = data_unlabelled.predictor.fillna('')

    # Load pipeline and make predictions
    pipe = joblib.load(pipe_path)
    predictions = pipe.predict(data_unlabelled[['predictor']])
    if preds_column is None:
        preds_column = predictor + '_preds'
    data_unlabelled[preds_column] = predictions

    data_unlabelled = data_unlabelled.rename(columns={'predictor': predictor})

    # Set column names of columns to return in final data frame
    if column_names is None:
        column_names = [predictor]
    elif column_names == '__all__':
        column_names == list(data_unlabelled)
    returned_cols = [[preds_column], column_names] # column_names is a list. Put preds_column in a list to create a list
                                                   # of lists to unnest later to get a list of strings.
    returned_cols = list(chain.from_iterable(returned_cols)) # Unnest list of lists.

    return data_unlabelled[returned_cols]
