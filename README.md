# pxtextmining: Text Classification of Patient Experience feedback

## Project description
**pxtextmining** is a Python package for classifying and conducting sentiment analysis of patient feedback comments, collected via the [NHS England Friends and Family Test](https://www.england.nhs.uk/fft/) (FFT).

There are two parts to the package. The first, comprising the majority of the codebase, is a machine learning pipeline that trains a model using labelled data. This pipeline outputs a fully trained model which can predict either 'criticality' scores or a thematic 'label' category for some feedback text. Examples of this can be found in the 'execution' folder.

The second part utilises the trained model to make predictions on unlabelled feedback text, outputting predicted labels or criticality scores. An example of how this works using a model trained to predict 'label' is given below:

```
dataset = pd.read_csv('datasets/text_data.csv')
predictions = factory_predict_unlabelled_text(dataset=dataset, predictor="feedback", pipe_path_or_object="results_label/pipeline_label.sav")
```

__We are working openly by open-sourcing the analysis code and data where possible to promote replication, reproducibility and further developments (pull requests are more than welcome!). We are also automating common steps in our workflow by shipping the pipeline as a [Python](https://www.python.org/) package broken down into sub-modules and helper functions to increase usability and documentation.__

## Documentation

Full documentation, including installation instructions, is available on our [documentation page](https://cdu-data-science-team.github.io/pxtextmining/).

## Pipeline to train a new model

The pipeline is built with Python's [`Scikit-learn`](https://scikit-learn.org/stable/index.html) (Pedregosa et al., 2011). The pipeline performs a randomized search ([`RandomizedSearchCV()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV)) to identify the best-performing learner and (hyper)parameter values.

Breakdown of the pipeline process, built by the functions in [pxtextmining.factories](https://github.com/CDU-data-science-team/pxtextmining/tree/main/pxtextmining/factories):

1. The data is loaded and split into training and test sets `factory_data_load_and_split`. This module also conducts some basic text preprocessing, such as removing special characters, whitespaces and linebreaks. It produces additional features through the creation of 'text_length' and sentiment scores using [vaderSentiment](https://pypi.org/project/vaderSentiment/) and [textblob](https://pypi.org/project/textblob/).

2. The function in `factory_pipeline` creates an sklearn pipeline. This pipeline is comprised of the following steps: first, the preprocessed text input is upsampled to help compensate for the unbalanced dataset. The text is then tokenized and vectorised using either [spacy](https://spacy.io/) or [wordnet](https://wordnet.princeton.edu/). Feature selection is then conducted. A hyperparameter grid is constructed with potential hyperparameter values, depending on the learners/classification models to be tested in the Randomized Search. The pipeline is then fitted on the dataset to identify the best model.

3. The fitted pipeline is then evaluated on the test set in `factory_model_performance`. The evaluation metrics used are: ([Accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html), [Class Balance Accuracy](https://lib.dr.iastate.edu/cgi/viewcontent.cgi?article=4544&context=etd) (Mosley, 2013), [Balanced Accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html) (Guyon et al., 2015, Kelleher et al., 2015) and [Matthews Correlation Coefficient](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html) (Baldi et al., 2000, Matthews, 1975)). A visual representation of the performance evaluation is output in the form of a barchart.

4. Writing the results: The fitted pipeline, tuning results, predictions, accuracy
per class, model comparison barchart, training data index, and test data index are output by `factory_write_results`.

The four steps above are all pulled together in [`pxtextmining.pipelines.text_classification_pipeline`](https://github.com/CDU-data-science-team/pxtextmining/tree/main/pxtextmining/pipelines).


## References
Baldi P., Brunak S., Chauvin Y., Andersen C.A.F. & Nielsen H. (2000). Assessing
the accuracy of prediction algorithms for classification: an overview.
_Bioinformatics_  16(5):412--424.

Guyon I., Bennett K. Cawley G., Escalante H.J., Escalera S., Ho T.K., Macià N.,
Ray B., Saeed M., Statnikov A.R, & Viegas E. (2015). [Design of the 2015 ChaLearn AutoML Challenge](https://ieeexplore.ieee.org/document/7280767),
International Joint Conference on Neural Networks (IJCNN).

Kelleher J.D., Mac Namee B. & D’Arcy A.(2015).
[Fundamentals of Machine Learning for Predictive Data Analytics: Algorithms, Worked Examples, and Case Studies](https://mitpress.mit.edu/books/fundamentals-machine-learning-predictive-data-analytics).
MIT Press.

Matthews B.W. (1975). Comparison of the predicted and observed secondary
structure of T4 phage lysozyme. _Biochimica et Biophysica Acta (BBA) - Protein Structure_ 405(2):442--451.

Pedregosa F., Varoquaux G., Gramfort A., Michel V., Thirion B., Grisel O.,
Blondel M., Prettenhofer P., Weiss R., Dubourg V., Vanderplas J., Passos A.,
Cournapeau D., Brucher M., Perrot M. & Duchesnay E. (2011),
[Scikit-learn: Machine Learning in Python](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html).
_Journal of Machine Learning Research_ 12:2825--2830

[^1]: A vritual environment can also be created using Conda, where the commands
for creating and activating it are a little different. See [this](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
