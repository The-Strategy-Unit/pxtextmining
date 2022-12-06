# Using a trained model

The `results` folders (e.g. [`results_label`](https://github.com/CDU-data-science-team/pxtextmining/tree/main/results_label)) always contain a fully trained model in .sav format, as well as performance metrics for the model. The models saved here are used to make predictions on the [experiencesdashboard](https://github.com/CDU-data-science-team/experiencesdashboard) frontend.

To use one of these pretrained models to make predictions, prepare a pandas DataFrame of the text you want to classify. Then, use this dataframe as the dataset argument in [factory_predict_unlabelled_text](../../reference/factories/factory_predict_unlabelled_text). An example of how this can be done is available in [execution_predict](https://github.com/CDU-data-science-team/pxtextmining/blob/main/execution/execution_predict.py).

Be wary of preparing your dataset using `Excel`, as it can cause issues due to text encoding errors. [`LibreOffice`](https://www.libreoffice.org/) is a good alternative.
