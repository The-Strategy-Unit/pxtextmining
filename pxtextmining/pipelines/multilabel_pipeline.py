from pxtextmining.factories.factory_data_load_and_split import load_multilabel_data, process_and_split_multilabel_data
from pxtextmining.factories.factory_model_performance import get_multilabel_metrics
from pxtextmining.factories.factory_pipeline import train_sklearn_multilabel_models


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
X_train, X_test, Y_train, Y_test = process_and_split_multilabel_data(df, target = major_cats, vectorise = True)
models = train_sklearn_multilabel_models(X_train, Y_train)
for m in models:
    get_multilabel_metrics(X_test, Y_test, labels = major_cats, model = m)
