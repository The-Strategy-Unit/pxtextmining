import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, matthews_corrcoef, accuracy_score
from pxtextmining.helpers.metrics import class_balance_accuracy_score
from sklearn.dummy import DummyClassifier


def get_metrics(x_train, x_test, y_train, y_test, model=None):
    """Function to produce performance metrics for a specific machine learning model.

    :param pd.DataFrame x_train: Training data (predictor).
    :param pd.Series y_train: Training data (target).
    :param pd.DataFrame x_test: Test data (predictor).
    :param pd.Series y_test: Test data (target).
    :param str model: Trained classifier. Defaults to 'dummy' which instantiates dummy classifier for baseline metrics.

    :return: A tuple containing the following objects, in order:
            A python dict containing the performance metrics 'accuracy', 'balanced accuracy', 'class balance accuracy',
            and 'matthews correlation coefficient';
            A pd.Series containing the predicted values for x_test, produced by the model.
    :rtype: tuple
    """
    if model == None:
        model = DummyClassifier(strategy = 'stratified')
        model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    metrics = {}
    metrics['accuracy'] = round (accuracy_score(y_test, y_pred),2)
    metrics['balanced accuracy'] = round (balanced_accuracy_score(y_test, y_pred),2)
    metrics['class balance accuracy'] = round (class_balance_accuracy_score(y_test, y_pred),2)
    metrics['matthews correlation'] = round(matthews_corrcoef(y_test, y_pred),2)
    return metrics, y_pred

def get_accuracy_per_class(y_test, pred):
    """Function to produce accuracy per class for the predicted categories, compared against real values.

    :param pd.Series y_test: Test data (real target values).
    :param pd.Series pred: Predicted target values.

    :return: The computed accuracy per class metrics for the model.
    :rtype: pd.DataFrame
    """
    cm = confusion_matrix(y_test, pred)
    accuracy_per_class = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    accuracy_per_class = pd.DataFrame(accuracy_per_class.diagonal())
    accuracy_per_class.columns = ["accuracy"]
    unique, frequency = np.unique(y_test, return_counts=True)
    accuracy_per_class["class"], accuracy_per_class["counts"] = unique, frequency
    accuracy_per_class = accuracy_per_class[["class", "counts", "accuracy"]]
    return accuracy_per_class

def factory_model_performance(pipe, x_train, y_train, x_test, y_test,
                              metric="class_balance_accuracy_score"):

    """
    Evaluate the performance of a fitted pipeline.

    :param sklearn.pipeline.Pipeline pipe: Fitted [sklearn.pipeline.Pipeline]
    (https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
    :param pd.DataFrame x_train: Training data (predictor).
    :param pd.Series y_train: Training data (target).
    :param pd.DataFrame x_test: Test data (predictor).
    :param pd.Series y_test: Test data (target).
    :param str metric: Performance metrics ("accuracy_score", "balanced_accuracy_score",
        "matthews_corrcoef", "class_balance_accuracy_score").
    :return: A tuple containing the following objects, in order:
            The fitted Scikit-learn/imblearn pipeline;
            A pandas.DataFrame with all (hyper)parameter values and models tried during fitting;
            A pandas.DataFrame with the predictions on the test set;
            A pandas.DataFrame with accuracies per class;
            A bar plot comparing the mean scores (of the user-supplied metric parameter)
                from the cross-validation on the training set, for the best
                (hyper)parameter values for each learner;
    :rtype: tuple
    """

    refit = metric.replace("_", " ").replace(" score", "").title()

    aux = pd.DataFrame(pipe.best_params_.items())
    best_estimator = aux[aux[0] == "clf__estimator"].reset_index()[1][0]
    estimator_position = len(pipe.best_estimator_) - 1
    pipe.best_estimator_.steps.pop(estimator_position)
    pipe.best_estimator_.steps.append(("clf", best_estimator))
    pipe.best_estimator_.fit(x_train, y_train)


    perf_metrics, pred = get_metrics(x_train, x_test, y_train, y_test, model=pipe.best_estimator_)
    baseline_metrics, baseline_preds = get_metrics(x_train, x_test, y_train, y_test, model = None)
    accuracy_per_class = get_accuracy_per_class(y_test, pred)
    best_params = {k:v for (k,v) in pipe.best_params_.items()}

    model_summary = { 'Dummy model performance metrics': baseline_metrics,
                      'Trained model performance metrics': perf_metrics,
                      'Best estimator': pipe.best_estimator_.named_steps["clf"],
                      'Best parameters': best_params
    }
    # print("The best estimator is %s" % (pipe.best_estimator_.named_steps["clf"]))
    # print("The best parameters are:")
    # for param, value in pipe.best_params_.items():
    #     print("{}: {}".format(param, value))
    # print("The best score from the cross-validation for \n the supplied scorer (" +
    #       refit + ") is %s"
    #       % (round(pipe.best_score_, 2)))

    tuning_results = pd.DataFrame(pipe.cv_results_)
    tuned_learners = []
    for i in tuning_results["param_clf__estimator"]:
        tuned_learners.append(i.__class__.__name__)
    tuning_results["learner"] = tuned_learners
    y_axis = "mean_test_" + refit
    tuning_results = tuning_results.sort_values(y_axis, ascending=False)
    tuning_results.columns = tuning_results.columns.str.replace('alltrans__process__', '') # When using ordinal with theme='label', names are too long.

    # Convert non-numeric to strings. This is to ensure that writing to MySQL won't throw an error.
    # (There MUST be a better way of fixing this!)
    for i in tuning_results.columns:
        if (
                (not isinstance(tuning_results[i][0], float)) and
                (not isinstance(tuning_results[i][0], int)) and
                (not isinstance(tuning_results[i][0], str))
        ):
            tuning_results[i] = tuning_results[i].apply(str)

    print("Plotting performance of the best of each estimator...")

    # Find the best tunings for each model. #
    # Note that SGDClassifier fits a logistic regression when loss is "log", but a Linear SVM when loss is "hinge".
    # Looking at column "learner" in "tuning results", one cannot tell which of the two models SGD is.
    # Let's make that clear.
    if 'param_clf__estimator__loss' in tuning_results.columns: # Need statement as models other than SGD don't have loss.
        learners = []
        for i, j in zip(tuning_results["learner"], tuning_results["param_clf__estimator__loss"]):
            if j == "log":
                learners.append("Logistic")
            elif j == "hinge":
                learners.append("Linear SVM")
            else:
                learners.append(i)
        tuning_results["learner"] = learners

    # Now, let's find the best tunings for each of the fitted models
    aux = tuning_results.filter(regex="mean_test|learner").groupby(["learner"]).max().reset_index()
    aux = aux.sort_values([y_axis], ascending=False)
    aux = aux.melt("learner")
    aux["variable"] = aux["variable"].str.replace("mean_test_", "")
    aux["learner"] = aux["learner"].str.replace("Classifier", "")

    p_compare_models_bar = sns.barplot(x="learner", y="value", hue="variable",
                                       data=aux)
    p_compare_models_bar.figure.set_size_inches(15, 13)
    p_compare_models_bar.set_xticklabels(p_compare_models_bar.get_xticklabels(),
                                         rotation=90)
    plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    p_compare_models_bar.set(xlabel=None, ylabel=None,
                             title="Learner performance ordered by " + refit)

    print("Fitting optimal pipeline on whole dataset...")
    pipe.best_estimator_.fit(pd.concat([x_train, x_test]), np.concatenate([y_train, y_test]))

    return pipe, tuning_results, pred, accuracy_per_class, p_compare_models_bar, model_summary
