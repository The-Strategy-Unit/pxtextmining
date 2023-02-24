import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, matthews_corrcoef, accuracy_score
from pxtextmining.helpers.metrics import class_balance_accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn import metrics
from sklearn.multioutput import MultiOutputClassifier
from tensorflow.keras import Sequential, Model
from pxtextmining.factories.factory_predict_unlabelled_text import turn_probs_into_binary, fix_no_labels


def get_multilabel_metrics(x_test, y_test, labels, model = None, training_time = None, x_train = None, y_train = None):
    """Function to produce performance metrics for a multilabel machine learning model.

    :param pd.DataFrame x_test: Test data (predictor).
    :param pd.Series y_test: Test data (target).
    :param pd.DataFrame x_train: Training data (predictor). Defaults to None, only needed if model = None
    :param pd.Series y_train: Training data (target). Defaults to None, only needed if model = None
    :param str model: Trained classifier. Defaults to 'dummy' which instantiates dummy classifier for baseline metrics.

    :return: None
    :rtype: None
    """

    metrics_string = '\n *****************'
    model_metrics = {}
    if model == None:
        model = DummyClassifier(strategy = 'uniform')
        if isinstance(x_train, pd.Series):
            model.fit(x_train, y_train)
        else:
            raise ValueError('For dummy model, x_train and y_train must be provided')
    # TF Keras models output probabilities with model.predict, whilst sklearn models output binary outcomes
    # Get them both to output the same (binary outcomes) and take max prob as label if no labels predicted at all
    if isinstance(model, (Sequential, Model)):
        y_probs = model.predict(x_test)
        binary_preds = turn_probs_into_binary(y_probs)
        y_pred = fix_no_labels(binary_preds, y_probs, model = 'tf')
    else:
        binary_preds = model.predict(x_test)
        y_probs = np.array(model.predict_proba(x_test))
        y_pred = fix_no_labels(binary_preds, y_probs, model = 'sklearn')
    c_report_str = metrics.classification_report(y_test, y_pred,
                                            target_names = labels, zero_division=0)
    model_metrics['exact_accuracy'] = metrics.accuracy_score(y_test, y_pred)
    model_metrics['hamming_loss'] = metrics.hamming_loss(y_test, y_pred)
    model_metrics['macro_jaccard_score'] = metrics.jaccard_score(y_test, y_pred, average = 'macro')
    if isinstance(model, (Sequential, Model)):
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        model_summary = "\n".join(stringlist)
        metrics_string += f'\n{model_summary}\n'
    else:
        metrics_string += f'\n{model}\n'
    metrics_string += f'\n\nTraining time: {training_time}\n'
    for k,v in model_metrics.items():
        metrics_string += f'\n{k}: {v}'
    metrics_string += '\n\n Classification report:\n'
    metrics_string += c_report_str
    # per_class_jaccard = zip(labels,metrics.jaccard_score(y_test, y_pred, average = None, zero_division = 0))
    # metrics_string += '\nper class Jaccard score:'
    # for k,v in per_class_jaccard:
    #     print(f'{k}: {v}')
    return metrics_string


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

def factory_model_performance(pipe, x_train, y_train, x_test, y_test):

    """
    Evaluate the performance of a fitted pipeline.

    :param sklearn.pipeline.Pipeline pipe: Fitted [sklearn.pipeline.Pipeline]
    (https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
    :param pd.DataFrame x_train: Training data (predictor).
    :param pd.Series y_train: Training data (target).
    :param pd.DataFrame x_test: Test data (predictor).
    :param pd.Series y_test: Test data (target).
    :return: A tuple containing the following objects, in order:
            The fitted Scikit-learn/imblearn pipeline;
            A pandas.DataFrame with all (hyper)parameter values and models tried during fitting;
            A pandas.DataFrame with the predictions on the test set;
            A pandas.DataFrame with accuracies per class;
            A bar plot comparing the mean performance metric scores from the cross-validation on
            the training set, for the best (hyper)parameter values for each learner;
            A dict containing performance metrics and model metadata.
    :rtype: tuple
    """

    aux = pd.DataFrame(pipe.best_params_.items())
    best_estimator = aux[aux[0] == "clf__estimator"].reset_index()[1][0]
    estimator_position = len(pipe.best_estimator_) - 1
    pipe.best_estimator_.steps.pop(estimator_position)
    pipe.best_estimator_.steps.append(("clf", best_estimator))
    pipe.best_estimator_.fit(x_train, y_train)


    perf_metrics, pred = get_metrics(x_train, x_test, y_train, y_test, model=pipe.best_estimator_)
    baseline_metrics, _ = get_metrics(x_train, x_test, y_train, y_test, model = None)
    accuracy_per_class = get_accuracy_per_class(y_test, pred)
    best_params = {k:v for (k,v) in pipe.best_params_.items()}

    model_summary = { 'Dummy model performance metrics': baseline_metrics,
                      'Trained model performance metrics': perf_metrics,
                      'Best estimator': pipe.best_estimator_.named_steps["clf"],
                      'Best parameters': best_params
    }

    for k,v in model_summary.items():
        print(f'{k}: {v}')

    tuning_results = pd.DataFrame(pipe.cv_results_)
    tuned_learners = []
    for i in tuning_results["param_clf__estimator"]:
        tuned_learners.append(i.__class__.__name__)
    tuning_results["learner"] = tuned_learners
    y_axis = "mean_test_Class Balance Accuracy"
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
                             title="Learner performance ordered by Class Balance Accuracy")

    print("Fitting optimal pipeline on whole dataset...")
    pipe.best_estimator_.fit(pd.concat([x_train, x_test]), np.concatenate([y_train, y_test]))

    return pipe, tuning_results, pred, accuracy_per_class, p_compare_models_bar, model_summary
