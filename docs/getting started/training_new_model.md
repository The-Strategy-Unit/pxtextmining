# Training a new model

To train a new model to categorise patient feedback text, labelled data is required.

Data from phase 1 of the project is available in the folder [`datasets`](https://github.com/CDU-data-science-team/pxtextmining/blob/main/datasets/text_data.csv), or it can also be loaded from an SQL database. If you have your own labelled patient experience feedback, you can use this instead to train your own model.

The [text_classification_pipeline](../../reference/pipelines/text_classification_pipeline) contains the function required to output a fully trained model. Two types of models can be trained using this pipeline, one that can predict the categorical 'label' for the text, or one that can predict the positive or negative 'criticality' score for the text. Examples of the pipeline being used to output each type of model can be seen in [execution_criticality](https://github.com/CDU-data-science-team/pxtextmining/blob/main/execution/execution_criticality.py) and [execution_label](https://github.com/CDU-data-science-team/pxtextmining/blob/main/execution/execution_label.py).

The steps involved in training a model are as follows:

1. The data is loaded and split into training and test sets by the function `factory_data_load_and_split`. This also conducts some basic text preprocessing, such as removing special characters, whitespaces and linebreaks. It produces additional features through the creation of 'text_length' and sentiment scores using [vaderSentiment](https://pypi.org/project/vaderSentiment/) and [textblob](https://pypi.org/project/textblob/). Any invalid lines (e.g. empty strings, NULL values) are removed from the data.

2. The function in `factory_pipeline` creates an sklearn pipeline. This pipeline is comprised of the following steps: first, the preprocessed text input is upsampled to help compensate for the unbalanced dataset. The text is then tokenized and vectorised (turned into numbers that can be processed by the model) using either [spacy](https://spacy.io/) or [wordnet](https://wordnet.princeton.edu/). Feature selection is then conducted to select only the most important features to train the model. A hyperparameter grid is constructed with potential hyperparameter values, depending on the learners/classification models to be tested in the Randomized Search. A Randomized Search is then used to identify the best performing model and its optimal hyperparameters.

3. The fitted pipeline is then evaluated on the test set in `factory_model_performance`. The evaluation metrics used are: ([Accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html), [Class Balance Accuracy](https://lib.dr.iastate.edu/cgi/viewcontent.cgi?article=4544&context=etd) (Mosley, 2013), [Balanced Accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html) (Guyon et al., 2015, Kelleher et al., 2015) and [Matthews Correlation Coefficient](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html) (Baldi et al., 2000, Matthews, 1975)). A visual representation of the performance evaluation is output in the form of a barchart.

4. Writing the results: The fitted pipeline, tuning results, predictions, accuracy
per class, model comparison barchart, training data index, and test data index are output by `factory_write_results`.

The four steps above are all pulled together in [`pxtextmining.pipelines.text_classification_pipeline`](https://github.com/CDU-data-science-team/pxtextmining/tree/main/pxtextmining/pipelines).

Here is a visual display of the process:
![](https://raw.githubusercontent.com/CDU-data-science-team/pxtextmining/main/text_classification_package_structure.png)
