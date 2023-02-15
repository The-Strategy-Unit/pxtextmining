from pxtextmining.factories.factory_data_load_and_split import load_multilabel_data, process_and_split_multilabel_data
from pxtextmining.factories.factory_model_performance import get_multilabel_metrics
from pxtextmining.factories.factory_pipeline import train_sklearn_multilabel_models, search_sklearn_pipelines
from pxtextmining.factories.factory_write_results import write_multilabel_models_and_metrics

# Should I put all of this into an 'pipeline' object??


df = load_multilabel_data(filename = 'datasets/phase_2_test.csv', target = 'major_categories')
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
X_train, X_test, Y_train, Y_test = process_and_split_multilabel_data(df, target = major_cats)

# models, training_times = search_sklearn_pipelines(X_train, Y_train, models_to_try = ['mnb', 'knn', 'svm', 'rfc'])
model_metrics = []
for i in range(len(models)):
    m = models[i]
    t = training_times[i]
    model_metrics.append(get_multilabel_metrics(X_test, Y_test, labels = major_cats, model = m, training_time = t))
model_metrics.append(get_multilabel_metrics(X_test, Y_test, labels = major_cats, x_train = X_train, y_train = Y_train, model = None))
write_multilabel_models_and_metrics(models,model_metrics,path='test_multilabel')
