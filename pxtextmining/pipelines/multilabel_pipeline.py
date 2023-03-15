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


def run_sklearn_pipeline():
    random_state = random.randint(1,999)
    df = load_multilabel_data(filename = 'datasets/hidden/multilabeldata_2.csv', target = 'major_categories')
    major_cats = ['Access to medical care & support',
    'Activities',
    'Additional',
    'Category TBC',
    'Communication & involvement',
    'Environment & equipment',
    'Food & diet',
    'General',
    'Medication',
    'Mental Health specifics',
    'Patient journey & service coordination',
    'Service location, travel & transport',
    'Staff']
    X_train, X_test, Y_train, Y_test = process_and_split_multilabel_data(df, target = major_cats,
                                                                         additional_features =True, random_state = random_state)
    models, training_times = search_sklearn_pipelines(X_train, Y_train, models_to_try = ['knn', 'svm', 'rfc'], additional_features = True)
    model_metrics = []
    for i in range(len(models)):
        m = models[i]
        t = training_times[i]
        model_metrics.append(get_multilabel_metrics(X_test, Y_test, random_state = random_state,
                                                    labels = major_cats, model_type = 'sklearn', model = m, training_time = t))
    # model_metrics.append(get_multilabel_metrics(X_test, Y_test, labels = major_cats, x_train = X_train, y_train = Y_train, model = None))
    write_multilabel_models_and_metrics(models,model_metrics,path='test_multilabel/additional_features')

def run_tf_pipeline():
    random_state = random.randint(1,999)
    df = load_multilabel_data(filename = 'datasets/multilabeldata_2.csv', target = 'major_categories')
    major_cats = ['Access to medical care & support',
    'Activities',
    'Additional',
    'Category TBC',
    'Communication & involvement',
    'Environment & equipment',
    'Food & diet',
    'General',
    'Medication',
    'Mental Health specifics',
    'Patient journey & service coordination',
    'Service location, travel & transport',
    'Staff']
    X_train, X_test, Y_train, Y_test = process_and_split_multilabel_data(df, target = major_cats, random_state = random_state)
    X_train_pad, vocab_size = tf_preprocessing(X_train)
    X_test_pad, _ = tf_preprocessing(X_test)
    class_weights_dict = calculating_class_weights(Y_train)
    model = create_tf_model(vocab_size)
    model_trained, training_time = train_tf_model(X_train_pad, Y_train, model, class_weights_dict = class_weights_dict)
    model_metrics = get_multilabel_metrics(X_test_pad, Y_test, random_state = random_state, labels = major_cats,
                                           model_type = 'tf',
                                           model = model_trained, training_time = training_time)
    write_multilabel_models_and_metrics([model_trained],[model_metrics],path='test_multilabel/tf')

def run_bert_pipeline(additional_features = False):
    random_state = random.randint(1,999)
    print(f'random_state is: {random_state}')
    df = load_multilabel_data(filename = 'datasets/hidden/multilabeldata_2.csv', target = 'major_categories')
    major_cats = ['Access to medical care & support',
    'Activities',
    'Additional',
    'Category TBC',
    'Communication & involvement',
    'Environment & equipment',
    'Food & diet',
    'General',
    'Medication',
    'Mental Health specifics',
    'Patient journey & service coordination',
    'Service location, travel & transport',
    'Staff']
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
    model_trained.save('test_multilabel/bert_additional_features')
    model_metrics = get_multilabel_metrics(test_dataset, Y_test, random_state = random_state, labels = major_cats,
                                           model_type = 'bert',
                                           model = model_trained, training_time = training_time,
                                           additional_features = additional_features, already_encoded = True)
    write_multilabel_models_and_metrics([model_trained],[model_metrics],path='test_multilabel/bert_additional_features')

if __name__ == '__main__':
    run_bert_pipeline(additional_features = True)
