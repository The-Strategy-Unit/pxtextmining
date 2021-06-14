import pandas as pd
import joblib
from itertools import chain


def factory_predict_unlabelled_text(file_path, dataset, predictor, pipe_path,
                                    preds_column=None, column_names=None):

    # Load data
    if file_path is not None:
        if column_names is None:
            column_names = [predictor]
        data_unlabelled = pd.DataFrame(pd.read_csv(file_path, encoding='utf-8', usecols=column_names))
    else:
        data_unlabelled = pd.DataFrame(dataset)

    # Rename predictor column and replace NAs with empty string.
    data_unlabelled = data_unlabelled.rename(columns={predictor: "predictor"})
    data_unlabelled['predictor'] = data_unlabelled.predictor.fillna('')

    # Load pipeline and make predictions
    pipe = joblib.load(pipe_path)
    predictions = pipe.predict(data_unlabelled[['predictor']])
    if preds_column is None:
        preds_column = predictor + '_preds'
    data_unlabelled[preds_column] = predictions

    returned_cols = [[preds_column], column_names] # column_names is a list. Put preds_column in a list to create a list
                                                   # of lists to unnest later to get a list of strings.
    returned_cols = [x for x in returned_cols if x is not None]
    returned_cols = list(chain.from_iterable(returned_cols)) # Unnest list of lists.

    data_unlabelled = data_unlabelled.rename(columns={"predictor": predictor})

    return data_unlabelled[returned_cols]
