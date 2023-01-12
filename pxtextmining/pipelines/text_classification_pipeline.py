from pxtextmining.factories.factory_data_load_and_split import factory_data_load_and_split
from pxtextmining.factories.factory_pipeline import factory_categorical_pipeline
from pxtextmining.factories.factory_model_performance import factory_model_performance
from pxtextmining.factories.factory_write_results import factory_write_results
import time

def text_classification_pipeline(filename, target, predictor, test_size=0.33,
                                 ordinal=False,
                                 tknz="spacy",
                                 cv=5, n_iter=100, n_jobs=5, verbose=3,
                                 learners=["SGDClassifier"],
                                 save_objects_to_server=True,
                                 save_objects_to_disk=False,
                                 results_folder_name="results",
                                 reduce_criticality=True,
                                 theme=None):

    """
    Function that gathers together all steps in Factories module to train a new model
    pipeline and write the results. Writes 7 files:

    - The fitted pipeline (SAV);
    - All (hyper)parameters tried during fitting and the associated pipeline performance metrics (CSV);
    - The predictions on the test set (CSV);
    - Accuracies per class (CSV);
    - The row indices of the training data (CSV);
    - The row indices of the test data (CSV);
    - A bar plot comparing the mean scores (of the user-supplied metric parameter) from the cross-validation on
      the training set, for the best (hyper)parameter values for each learner (PNG);

    **NOTE:** As described later, arguments `reduce_criticality` and `theme` are for internal use by Nottinghamshire
    Healthcare NHS Foundation Trust or other trusts who use the theme ("Access", "Environment/ facilities" etc.) and
    criticality labels. They can otherwise be safely ignored.

    :param str filename: Dataset name (CSV), including the data type suffix. If None, data is read from the database.
    :param str target: Name of the response variable.
    :param str predictor: Name of the predictor variable.
    :param float test_size: Proportion of data that will form the test dataset.
    :param bool ordinal: Whether to fit an ordinal classification model. The ordinal model is the implementation of
        `Frank and Hall (2001) <https://www.cs.waikato.ac.nz/~eibe/pubs/ordinal_tech_report.pdf>`_ that can use any standard classification model.
    :param str tknz: Tokenizer to use ("spacy" or "wordnet").
    :param int cv: Number of cross-validation folds.
    :param int n_iter: Number of parameter settings that are sampled
        (see `sklearn.model_selection.RandomizedSearchCV
        <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html>`_).
    :param int n_jobs: Number of jobs to run in parallel (see ``sklearn.model_selection.RandomizedSearchCV``).
    :param int verbose: Controls the verbosity (see ``sklearn.model_selection.RandomizedSearchCV``).
    :param list[str] learners: A list of ``Scikit-learn`` names of the learners to tune. Must be one or more of
        "SGDClassifier", "RidgeClassifier", "Perceptron", "PassiveAggressiveClassifier", "BernoulliNB", "ComplementNB",
        "MultinomialNB", "KNeighborsClassifier", "NearestCentroid", "RandomForestClassifier".
    :param bool save_objects_to_server: Whether to save the results to the server. **NOTE:** The feature that writes
        results to the database is for internal use only. It will be removed when a proper API is developed for this
        function.
    :param bool save_objects_to_disk:  Whether to save the results to disk. See ``results_folder_name``.
    :param str results_folder_name: Name of folder in which to save the results. It will create a new folder or
        overwrite an existing one that has the same name.
    :param bool reduce_criticality: For internal use by Nottinghamshire Healthcare NHS Foundation Trust or other trusts
        that hold data on criticality. If `True`, then all records with a criticality of "-5" (respectively, "5") are
        assigned a criticality of "-4" (respectively, "4"). This is to avoid situations where the pipeline breaks due to
        a lack of sufficient data for "-5" and/or "5". Defaults to `False`.
    :param str theme: For internal use by Nottinghamshire Healthcare NHS Foundation Trust or other trusts
        that use theme labels ("Access", "Environment/ facilities" etc.). The column name of the theme variable.
        Defaults to `None`. If supplied, the theme variable will be used as a predictor (along with the text predictor)
        in the model that is fitted with criticality as the response variable. The rationale is two-fold. First, to
        help the model improve predictions on criticality when the theme labels are readily available. Second, to force
        the criticality for "Couldn't be improved" to always be "3" in the training and test data, as well as in the
        predictions. This is the only criticality value that "Couldn't be improved" can take, so by forcing it to always
        be "3", we are improving model performance, but are also correcting possible erroneous assignments of values
        other than "3" that are attributed to human error.
    :return: A ``tuple`` of length 7 containing the following objects, in order:
        - The fitted ``Scikit-learn``/``imblearn`` pipeline;
        - A ``pandas.DataFrame`` with all (hyper)parameter values and models tried during fitting;
        - A ``pandas.DataFrame`` with the predictions on the test set;
        - A ``pandas.DataFrame`` with accuracies per class;
        - A bar plot comparing the mean scores (of the user-supplied metric parameter) from the cross-validation on
          the training set, for the best (hyper)parameter values for each learner.
        - The row indices of the training data;
        - The row indices of the test data
    :rtype: tuple
    """

    start_time = time.time()

    x_train, x_test, y_train, y_test, index_training_data, index_test_data = \
        factory_data_load_and_split(filename, target, predictor, test_size, reduce_criticality, theme)

    pipe = factory_categorical_pipeline(x_train, y_train, tknz, ordinal, cv, n_iter, n_jobs, verbose, learners, theme)

    pipe, tuning_results, pred, accuracy_per_class, p_compare_models_bar, model_summary = \
        factory_model_performance(pipe, x_train, y_train, x_test, y_test)

    pred, index_training_data, index_test_data = factory_write_results(pipe,
                                                                        tuning_results,
                                                                        pred,
                                                                        accuracy_per_class,
                                                                        p_compare_models_bar,
                                                                        target,
                                                                        index_training_data,
                                                                        index_test_data,
                                                                        model_summary,
                                                                        save_objects_to_server,
                                                                        save_objects_to_disk,
                                                                        results_folder_name)

    print("--- Time for pipeline completion: %s seconds ---" % (time.time() - start_time))

    return pipe, tuning_results, pred, accuracy_per_class, p_compare_models_bar, index_training_data, index_test_data
