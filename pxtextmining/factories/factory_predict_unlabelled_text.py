import pandas as pd
import joblib
from itertools import chain


def factory_predict_unlabelled_text(dataset, predictor, pipe_path_or_object,
                                    preds_column=None, column_names='all_cols', theme=None):
    """
    Predict unlabelled text data using a fitted `sklearn.pipeline.Pipeline
    <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_/`imblearn.pipeline.Pipeline
    <https://imbalanced-learn.org/stable/references/generated/imblearn.pipeline.Pipeline.html#imblearn.pipeline.Pipeline>`_.

    **NOTE:** As described later, argument `theme` is for internal use by Nottinghamshire Healthcare NHS Foundation
    Trust or other trusts who use the theme ("Access", "Environment/ facilities" etc.) labels. It can otherwise be
    safely ignored.

    :param dataset: A ``pandas.DataFrame`` (or an object that can be converted into such) with the text data to predict
        classes for.
    :param str predictor: The column name of the text variable.
    :param str, sklearn.model_selection._search.RandomizedSearchCV pipe_path_or_object: A string in the form
        path_to_fitted_pipeline/pipeline.sav," where "pipeline" is the name of the SAV file with the fitted
        ``Scikit-learn``/``imblearn.pipeline.Pipeline`` or a ``sklearn.model_selection._search.RandomizedSearchCV``.
    :param str preds_column: The user-specified name of the column that will have the predictions. If ``None`` (default),
        then the name will be ``predictor + '_preds'``.
    :param column_names:  A ``list``/``tuple`` of strings with the names of the columns of the supplied data frame (incl.
        ``predictor``) to be added to the returned ``pandas.DataFrame``.  If "preds_only", then the only column in
        the returned data frame will be ``preds_column``. Defaults to "all_cols".
    :param str theme: For internal use by Nottinghamshire Healthcare NHS Foundation Trust or other trusts
        that use theme labels ("Access", "Environment/ facilities" etc.). The column name of the theme variable.
        Defaults to `None`. If supplied, the theme variable will be used as a predictor (along with the text predictor)
        in the model that is fitted with criticality as the response variable. The rationale is two-fold. First, to
        help the model improve predictions on criticality when the theme labels are readily available. Second, to force
        the criticality for "Couldn't be improved" to always be "3" in the training and test data, as well as in the
        predictions. This is the only criticality value that "Couldn't be improved" can take, so by forcing it to always
        be "3", we are improving model performance, but are also correcting possible erroneous assignments of values
        other than "3" that are attributed to human error.
    :return: A ``pandas.DataFrame`` with the predictions and any other columns supplied in ``column_names``.
    """

    data_unlabelled = pd.DataFrame(dataset)

    # Rename predictor column to names pipeline knows and replace NAs with empty string.
    if theme is None:
        data_unlabelled = data_unlabelled.rename(columns={predictor: 'predictor'})
    else:
        data_unlabelled = data_unlabelled.rename(columns={predictor: 'predictor', theme: 'theme'})
    data_unlabelled['predictor'] = data_unlabelled.predictor.fillna('')

    # Load pipeline (if not already supplied) and make predictions
    if isinstance(pipe_path_or_object, str):
        pipe = joblib.load(pipe_path_or_object)
    else:
        pipe = pipe_path_or_object
    if theme is None:
        predictions = pipe.predict(data_unlabelled[['predictor']])
    else:
        predictions = pipe.predict(data_unlabelled[['predictor', 'theme']])

    if preds_column is None:
        preds_column = predictor + '_preds'
    data_unlabelled[preds_column] = predictions

    # Rename back to original variable names
    data_unlabelled = data_unlabelled.rename(columns={'predictor': predictor})
    data_unlabelled = data_unlabelled.rename(columns={'theme': theme})

    # Set column names of columns to return in final data frame
    if column_names == 'all_cols':
        column_names = [data_unlabelled]
    elif column_names == 'preds_only':
        column_names = None
    elif type(column_names) is str:
        column_names = [column_names]

    returned_cols = [[preds_column], column_names] # column_names is a list. Put preds_column in a list to create a list
                                                   # of lists to unnest later to get a list of strings.
    returned_cols = [x for x in returned_cols if x is not None]
    returned_cols = list(chain.from_iterable(returned_cols))  # Unnest list of lists.

    return data_unlabelled[returned_cols]
