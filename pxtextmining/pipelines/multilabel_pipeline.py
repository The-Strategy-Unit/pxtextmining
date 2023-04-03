import random

from sklearn.model_selection import train_test_split

from pxtextmining.factories.factory_data_load_and_split import (
    bert_data_to_dataset, load_multilabel_data,
    process_and_split_multilabel_data)
from pxtextmining.factories.factory_model_performance import \
    get_multilabel_metrics
from pxtextmining.factories.factory_pipeline import (
    calculating_class_weights, create_bert_model,
    create_bert_model_additional_features, create_tf_model,
    search_sklearn_pipelines, train_bert_model, train_tf_model)
from pxtextmining.factories.factory_write_results import \
    write_multilabel_models_and_metrics
from pxtextmining.helpers.text_preprocessor import tf_preprocessing
from pxtextmining.params import major_cats

def run_sklearn_pipeline(additional_features = False, target= major_cats, models_to_try = ["mnb", "knn", "svm", "rfc"], path = 'test_multilabel'):
    """Runs all the functions required to load multilabel data, preprocess it, and split it into training and test sets.
    Creates sklearn pipelines and hyperparameters to search, using specified estimators.
    For each estimator type selected, performs a randomized search across the hyperparameters to identify the parameters providing the best
    results on the holdout data within the randomized search.
    Evaluates the performance of the refitted estimator with the best hyperparameters on the test set, and saves the model
    and the performance metrics to a specified folder.

    Args:
        additional_features (bool, optional): Whether or not additional features (question type and text length) are used. Defaults to False.
        target (list, optional): The target labels, which should be columns in the dataset DataFrame. Defaults to major_cats.
        models_to_try (list, optional): List of the estimators to try. Defaults to ["mnb", "knn", "svm", "rfc"]. Permitted values are "mnb" (Multinomial Naive Bayes), "knn" (K Nearest Neighbours), "svm" (Support Vector Classifier), or "rfc" (Random Forest Classifier).
        path (str, optional): Path where the models are to be saved. If path does not exist, it will be created. Defaults to 'test_multilabel'.
    """
    random_state = random.randint(1,999)
    # This line on loading dataframe has to be manually edited depending on the target and the filename, currently
    df = load_multilabel_data(filename = 'datasets/hidden/multilabeldata_2.csv', target = 'major_categories')
    X_train, X_test, Y_train, Y_test = process_and_split_multilabel_data(df, target = target,
                                                                         additional_features =additional_features, random_state = random_state)
    models, training_times = search_sklearn_pipelines(X_train, Y_train, models_to_try = models_to_try, additional_features = additional_features)
    model_metrics = []
    for i in range(len(models)):
        m = models[i]
        t = training_times[i]
        model_metrics.append(get_multilabel_metrics(X_test, Y_test, random_state = random_state,
                                                    labels = target, model_type = 'sklearn', model = m, training_time = t))
    write_multilabel_models_and_metrics(models,model_metrics,path=path)

def run_tf_pipeline(target= major_cats, path = 'test_multilabel/tf'):
    """Runs all the functions required to load multilabel data, preprocess it, and split it into training and test sets.
    Creates tf.keras LSTM model and trains it on the train set.
    Evaluates the performance of trained model with the best hyperparameters on the test set, and saves the model
    and the performance metrics to a specified folder.
    Cannot currently take additional features, is only designed for text data alone.
    This model architecture performs very poorly and may be taken out of the model.

    Args:
        target (list, optional): The target labels, which should be columns in the dataset DataFrame. Defaults to major_cats.
        path (str, optional): Path where the models are to be saved. If path does not exist, it will be created. Defaults to 'test_multilabel'.
    """
    random_state = random.randint(1,999)
    df = load_multilabel_data(filename = 'datasets/multilabeldata_2.csv', target = 'major_categories')
    X_train, X_test, Y_train, Y_test = process_and_split_multilabel_data(df, target = target, random_state = random_state)
    X_train_pad, vocab_size = tf_preprocessing(X_train)
    X_test_pad, _ = tf_preprocessing(X_test)
    class_weights_dict = calculating_class_weights(Y_train)
    model = create_tf_model(vocab_size)
    model_trained, training_time = train_tf_model(X_train_pad, Y_train, model, class_weights_dict = class_weights_dict)
    model_metrics = get_multilabel_metrics(X_test_pad, Y_test, random_state = random_state, labels = major_cats,
                                           model_type = 'tf',
                                           model = model_trained, training_time = training_time)
    write_multilabel_models_and_metrics([model_trained],[model_metrics],path=path)

def run_bert_pipeline(additional_features = False, path = 'test_multilabel/bert'):
    """Runs all the functions required to load multilabel data, preprocess it, and split it into training, test and validation sets.
    Creates tf.keras Transformer model with additional layers specific to the classification task, and trains it on the train set.
    Evaluates the performance of trained model with the best hyperparameters on the test set, and saves the model
    and the performance metrics to a specified folder.

    Args:
        additional_features (bool, optional): Whether or not additional features (question type and text length) are used. Defaults to False.
        path (str, optional): Path where the models are to be saved. If path does not exist, it will be created. Defaults to 'test_multilabel'.
    """
    random_state = random.randint(1,999)
    print(f'random_state is: {random_state}')
    # This line on loading dataframe has to be manually edited depending on the target and the filename, currently
    df = load_multilabel_data(filename = 'datasets/hidden/multilabeldata_2.csv', target = 'major_categories')
    X_train_val, X_test, Y_train_val, Y_test = process_and_split_multilabel_data(df, target = major_cats, preprocess_text = False,
                                                                                 additional_features = additional_features, random_state = random_state)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.2, random_state = random_state)
    train_dataset = bert_data_to_dataset(X_train, Y_train, additional_features = additional_features)
    val_dataset = bert_data_to_dataset(X_val, Y_val, additional_features = additional_features)
    test_dataset = bert_data_to_dataset(X_test, Y = None, additional_features = additional_features)
    class_weights_dict = calculating_class_weights(Y_train_val)
    if additional_features == True:
        model = create_bert_model_additional_features(Y_train)
    else:
        model = create_bert_model(Y_train)
    model_trained, training_time = train_bert_model(train_dataset, val_dataset, model,
                                                    class_weights_dict = class_weights_dict, epochs = 25)
    model_metrics = get_multilabel_metrics(test_dataset, Y_test, random_state = random_state, labels = major_cats,
                                           model_type = 'bert',
                                           model = model_trained, training_time = training_time,
                                           additional_features = additional_features, already_encoded = True)
    write_multilabel_models_and_metrics([model_trained],[model_metrics],path=path)

if __name__ == '__main__':
    run_sklearn_pipeline(additional_features = True)
