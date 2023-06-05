import datetime
import time

import numpy as np
from scipy import stats
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import cross_validate
from tensorflow.keras import Sequential, layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.layers import Dense, Dropout, Input, concatenate
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from transformers import DistilBertConfig, TFDistilBertForSequenceClassification
import xgboost as xgb

from pxtextmining.helpers.tokenization import spacy_tokenizer
from pxtextmining.params import model_name

model_name = model_name


def create_sklearn_pipeline_sentiment(
    model_type, num_classes, tokenizer=None, additional_features=False
):
    """Creates sklearn pipeline and hyperparameter grid for searching, for a multiclass target.

    Args:
        model_type (str): Allows for selection of different estimators. Permitted values are "svm" (Support Vector Classifier), or "xgb" (XGBoost).
        num_classes (int): Number of target classes.
        tokenizer (str, optional): Allows for selection of "spacy" tokenizer. Defaults to None, which is the default sklearn tokenizer
        additional_features (bool, optional): Whether or not additional features (question type, text length) are to be included in the features fed into the model. Defaults to True.

    Returns:
        (tuple): Tuple containing the `sklearn.pipeline.Pipeline` with the selected estimator, and a `dict` containing the hyperparameters to be tuned.

    """
    if additional_features == True:
        cat_transformer = OneHotEncoder(handle_unknown="ignore")
        vectorizer = create_sklearn_vectorizer(tokenizer=None)
        preproc = make_column_transformer(
            (cat_transformer, ["FFT_q_standardised"]),
            (vectorizer, "FFT answer"),
        )
        params = {
            "columntransformer__tfidfvectorizer__ngram_range": ((1, 1), (1, 2), (2, 2)),
            "columntransformer__tfidfvectorizer__max_df": [
                0.85,
                0.86,
                0.87,
                0.88,
                0.89,
                0.9,
                0.91,
                0.92,
                0.93,
                0.94,
                0.95,
                0.96,
                0.97,
                0.98,
                0.99,
            ],
            "columntransformer__tfidfvectorizer__min_df": stats.uniform(0, 0.1),
        }
    else:
        preproc = create_sklearn_vectorizer(tokenizer=tokenizer)
        params = {
            "tfidfvectorizer__ngram_range": ((1, 1), (1, 2), (2, 2)),
            "tfidfvectorizer__max_df": [
                0.85,
                0.86,
                0.87,
                0.88,
                0.89,
                0.9,
                0.91,
                0.92,
                0.93,
                0.94,
                0.95,
                0.96,
                0.97,
                0.98,
                0.99,
            ],
            "tfidfvectorizer__min_df": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    if model_type == "svm":
        pipe = make_pipeline(
            preproc,
            SVC(
                probability=True,
                class_weight="balanced",
                max_iter=1000,
                cache_size=1000,
            ),
        )
        params["svc__C"] = stats.uniform(0.1, 20)
        params["svc__kernel"] = [
            "linear",
            "rbf",
            "sigmoid",
        ]
    if model_type == "xgb":
        pipe = make_pipeline(
            preproc,
            xgb.XGBClassifier(
                num_class=num_classes, objective="multi:softmax", n_estimators=500
            ),
        )
        params["xgbclassifier__max_depth"] = [4, 5, 6, 7, 8]
        params["xgbclassifier__min_child_weight"] = [0.5, 1, 2, 5]
        params["xgbclassifier__gamma"] = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    return pipe, params


def create_bert_model(Y_train, model_name=model_name, max_length=150, multilabel=True):
    """Creates Transformer based model trained on text data, with last layer added on
    for multilabel classification task. Number of neurons in last layer depends on number of labels in Y target.

    Args:
        Y_train (pd.DataFrame): DataFrame containing one-hot encoded targets
        model_name (str, optional): Type of pretrained transformer model to load. Defaults to `model_name` set in `pxtextmining.params`
        max_length (int, optional): Maximum length of text to be passed through transformer model. Defaults to 150.
        multilabel (Bool, optional): Whether the target is multilabel or not. If set to False, target is multiclass. Defaults to True.

    Returns:
        (tensorflow.keras.models.Model): Compiled Tensforflow Keras model with pretrained transformer layers and last layer suited for multilabel classification task
    """
    config = DistilBertConfig.from_pretrained(model_name)
    transformer_model = TFDistilBertForSequenceClassification.from_pretrained(
        model_name, output_hidden_states=False
    )
    bert = transformer_model.layers[0]
    input_ids = Input(shape=(max_length,), name="input_ids", dtype="int32")
    inputs = {"input_ids": input_ids}
    bert_model = bert(inputs)[0][:, 0, :]
    dropout = Dropout(config.dropout, name="pooled_output")
    pooled_output = dropout(bert_model, training=False)
    if multilabel == True:
        output = Dense(
            units=Y_train.shape[1],
            kernel_initializer=TruncatedNormal(stddev=config.initializer_range),
            activation="sigmoid",
            name="output",
        )(pooled_output)
        loss = BinaryCrossentropy()
    else:
        output = Dense(
            units=Y_train.shape[1],
            kernel_initializer=TruncatedNormal(stddev=config.initializer_range),
            activation="softmax",
            name="output",
        )(pooled_output)
        loss = CategoricalCrossentropy()
    model = Model(inputs=inputs, outputs=output, name="DistilBERT")
    # compile model
    optimizer = Adam(5e-5)
    metrics = ["CategoricalAccuracy"]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def create_bert_model_additional_features(
    Y_train, model_name=model_name, max_length=150, multilabel=True
):
    """Creates Transformer based model trained on text data, concatenated with an additional Dense layer taking additional inputs, with last layer added on
    for multilabel classification task. Number of neurons in last layer depends on number of labels in Y target.

    Args:
        Y_train (pd.DataFrame): DataFrame containing one-hot encoded targets
        model_name (str, optional): Type of pretrained transformer model to load. Defaults to `model_name` set in `pxtextmining.params`
        max_length (int, optional): Maximum length of text to be passed through transformer model. Defaults to 150.
        multilabel (Bool, optional): Whether the target is multilabel or not. If set to False, target is multiclass. Defaults to True.

    Returns:
        (tensorflow.keras.models.Model): Compiled Tensforflow Keras model with pretrained transformer layers, three question_type inputs passed through a Dense layer, and last layer suited for multilabel classification task

    """
    config = DistilBertConfig.from_pretrained(model_name)
    transformer_model = TFDistilBertForSequenceClassification.from_pretrained(
        model_name, output_hidden_states=False
    )
    bert = transformer_model.layers[0]
    input_ids = Input(shape=(max_length,), name="input_ids", dtype="int32")
    input_text = {"input_ids": input_ids}
    bert_model = bert(input_text)[0][:, 0, :]
    dropout = Dropout(config.dropout, name="pooled_output")
    bert_output = dropout(bert_model, training=False)
    # Get onehotencoded categories in (3 categories)
    input_cat = Input(shape=(3,), name="input_cat")
    cat_dense = Dense(units=10, activation="relu")
    cat_dense = cat_dense(input_cat)
    # concatenate both together
    concat_layer = concatenate([bert_output, cat_dense])
    if multilabel == True:
        output = Dense(
            units=Y_train.shape[1],
            kernel_initializer=TruncatedNormal(stddev=config.initializer_range),
            activation="sigmoid",
            name="output",
        )(concat_layer)
        loss = BinaryCrossentropy()
    else:
        output = Dense(
            units=Y_train.shape[1],
            kernel_initializer=TruncatedNormal(stddev=config.initializer_range),
            activation="softmax",
            name="output",
        )(concat_layer)
        loss = CategoricalCrossentropy()
    model = Model(inputs=[input_ids, input_cat], outputs=output)
    optimizer = Adam(5e-5)
    metrics = ["CategoricalAccuracy"]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def train_bert_model(
    train_dataset, val_dataset, model, class_weights_dict=None, epochs=30
):
    """Trains compiled transformer model with early stopping.

    Args:
        train_dataset (tf.data.Dataset): Train dataset, tokenized with huggingface tokenizer, in tf.data.Dataset format
        val_dataset (tf.data.Dataset): Validation dataset, tokenized with huggingface tokenizer, in tf.data.Dataset format
        model (tf.keras.models.Model): Compiled transformer model with additional layers for specific task.
        class_weights_dict (dict, optional): Dict containing class weights for each target class. Defaults to None.
        epochs (int, optional): Number of epochs to train model for. Defaults to 30.

    Returns:
        (tuple): Tuple containing trained model and the training time as a str.
    """
    es = EarlyStopping(patience=2, restore_best_weights=True)
    start_time = time.time()
    model.fit(
        train_dataset.shuffle(1000).batch(16),
        epochs=epochs,
        batch_size=16,
        class_weight=class_weights_dict,
        validation_data=val_dataset.batch(16),
        callbacks=[es],
    )
    total_time = round(time.time() - start_time, 0)
    training_time = str(datetime.timedelta(seconds=total_time))
    return model, training_time


def calculating_class_weights(y_true):
    """Function for calculating class weights for target classes, to be used when fitting a model.

    Args:
        y_true (pd.DataFrame): Dataset containing onehot encoded multilabel targets

    Returns:
        (dict): Dict containing calculated class weights for each target label.
    """
    y_np = np.array(y_true)
    number_dim = np.shape(y_np)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = compute_class_weight("balanced", classes=[0.0, 1.0], y=y_np[:, i])
    class_weights_dict = {}
    for i in range(len(weights)):
        class_weights_dict[i] = weights[i][-1]
    return class_weights_dict


def create_tf_model(vocab_size=None, embedding_size=100):
    """Creates LSTM multilabel classification model using tensorflow keras, with a layer of outputs matching what is currently the number of major_categories.

    Args:
        vocab_size (int, optional): Number of different words in vocabulary, as derived from tokenization process. Defaults to None.
        embedding_size (int, optional): Size of embedding dimension to be output by embedding layer. Defaults to 100.

    Returns:
        (tf.keras.models.Model): Compiled tensorflow keras LSTM model.
    """
    model = Sequential()
    model.add(
        layers.Embedding(
            input_dim=vocab_size + 1, output_dim=embedding_size, mask_zero=True
        )
    )
    model.add(layers.LSTM(50))
    model.add(layers.Dense(20, activation="relu"))
    model.add(layers.Dense(13, activation="sigmoid"))
    model.compile(
        loss="binary_crossentropy",
        optimizer="rmsprop",
        metrics=["CategoricalAccuracy", "Precision", "Recall"],
    )
    return model


def train_tf_model(X_train, Y_train, model, class_weights_dict=None):
    """Trains tensorflow keras model. Some overlap with train_bert_model, could probably be merged.

    Args:
        X_train (pd.DataFrame): DataFrame containing tokenized text features.
        Y_train (pd.DataFrame): DataFrame containing one-hot encoded multilabel targets.
        model (tf.keras.models.Model): Compiled tensorflow keras model.
        class_weights_dict (dict, optional): Dict containing class weights for target classes. Defaults to None.

    Returns:
        (tuple): Tuple containing trained model and the training time as a str.
    """
    es = EarlyStopping(patience=3, restore_best_weights=True)
    start_time = time.time()
    model.fit(
        X_train,
        Y_train,
        epochs=200,
        batch_size=32,
        verbose=1,
        validation_split=0.2,
        callbacks=[es],
        class_weight=class_weights_dict,
    )
    seconds_taken = round(time.time() - start_time, 0)
    training_time = str(datetime.timedelta(seconds=seconds_taken))
    return model, training_time


def create_sklearn_vectorizer(tokenizer=None):
    """Creates vectorizer for use with sklearn models, either using sklearn tokenizer or the spacy tokenizer

    Args:
        tokenizer (str, optional): Enables selection of spacy tokenizer. Defaults to None, which is sklearn default tokenizer.

    Returns:
        (sklearn.feature_extraction.text.TfidfVectorizer): sklearn TfidfVectorizer with either spacy or sklearn tokenizer
    """
    if tokenizer == "spacy":
        vectorizer = TfidfVectorizer(tokenizer=spacy_tokenizer)
    else:
        vectorizer = TfidfVectorizer()
    return vectorizer


def create_sklearn_pipeline(model_type, tokenizer=None, additional_features=True):
    """Creates sklearn pipeline and hyperparameter grid for searching, depending on model_type selected.

    Args:
        model_type (str): Allows for selection of different estimators. Permitted values are "mnb" (Multinomial Naive Bayes), "knn" (K Nearest Neighbours), "svm" (Support Vector Classifier), or "rfc" (Random Forest Classifier).
        tokenizer (str, optional): Allows for selection of "spacy" tokenizer. Defaults to None, which is the default sklearn tokenizer
        additional_features (bool, optional): Whether or not additional features (question type, text length) are to be included in the features fed into the model. Defaults to True.

    Returns:
        (tuple): Tuple containing the `sklearn.pipeline.Pipeline` with the selected estimator, and a `dict` containing the hyperparameters to be tuned.
    """
    if additional_features == True:
        cat_transformer = OneHotEncoder(handle_unknown="ignore")
        vectorizer = create_sklearn_vectorizer(tokenizer=None)
        # num_transformer = RobustScaler()
        preproc = make_column_transformer(
            (cat_transformer, ["FFT_q_standardised"]),
            (vectorizer, "FFT answer"),
            # (num_transformer, ["text_length"]),
        )
        params = {
            "columntransformer__tfidfvectorizer__ngram_range": ((1, 1), (1, 2), (2, 2)),
            "columntransformer__tfidfvectorizer__max_df": [
                0.9,
                0.95,
                0.99,
            ],
            "columntransformer__tfidfvectorizer__min_df": [0,0.01,0.02],
        }
    else:
        preproc = create_sklearn_vectorizer(tokenizer=tokenizer)
        params = {
            "tfidfvectorizer__ngram_range": ((1, 1), (1, 2), (2, 2)),
            "tfidfvectorizer__max_df": stats.uniform(0.8, 1),
            "tfidfvectorizer__min_df": stats.uniform(0.01, 0.1),
        }
    if model_type == "mnb":
        pipe = make_pipeline(preproc, MultiOutputClassifier(MultinomialNB()))
        params["multioutputclassifier__estimator__alpha"] = stats.uniform(0.1, 1)
    if model_type == "knn":
        pipe = make_pipeline(preproc, KNeighborsClassifier())
        params["kneighborsclassifier__n_neighbors"] = stats.randint(1, 100)
    if model_type == "svm":
        pipe = make_pipeline(
            preproc,
            MultiOutputClassifier(
                SVC(
                    probability=True,
                    class_weight="balanced",
                    max_iter=1000,
                    cache_size=1000,
                ),
                n_jobs=-1,
            ),
        )
        params["multioutputclassifier__estimator__C"] = [10,15,20]
        params["multioutputclassifier__estimator__gamma"] = np.logspace(-9, 3, 13)
        # params["multioutputclassifier__estimator__kernel"] = [
        #     "linear",
        #     "rbf",
        #     "sigmoid",
        # ]
    if model_type == "rfc":
        pipe = make_pipeline(preproc, RandomForestClassifier(n_jobs=-1))
        params["randomforestclassifier__max_depth"] = stats.randint(5, 50)
        params["randomforestclassifier__min_samples_split"] = stats.randint(2, 5)
        params["randomforestclassifier__class_weight"] = [
            "balanced",
            "balanced_subsample",
            None,
        ]
        params["randomforestclassifier__min_samples_leaf"] = stats.randint(1, 10)
        params["randomforestclassifier__max_features"] = ["sqrt", "log2", None, 0.3]
    if model_type == "xgb":
        pipe = make_pipeline(preproc, xgb.XGBClassifier(tree_method="hist", n_estimators=200))
        params["xgbclassifier__max_depth"] = [4, 5, 6, 7, 8]
        params["xgbclassifier__min_child_weight"] = [0.5, 1, 2, 5]
        params["xgbclassifier__gamma"] = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    return pipe, params


def search_sklearn_pipelines(
    X_train, Y_train, models_to_try, target=None, additional_features=True
):
    """Iterates through selected estimators, instantiating the relevant sklearn pipelines and searching for the optimum hyperparameters.

    Args:
        X_train (pd.DataFrame): DataFrame containing the features to be fed into the estimator
        Y_train (pd.DataFrame): DataFrame containing the targets
        models_to_try (list): List containing the estimator types to be tried. Permitted values are "mnb" (Multinomial Naive Bayes), "knn" (K Nearest Neighbours), "svm" (Support Vector Classifier), or "rfc" (Random Forest Classifier).
        additional_features (bool, optional): Whether or not additional features (question type, text length) are to be included in the features fed into the model. Defaults to True.

    Raises:
        ValueError: If model_type includes value other than permitted values "mnb", "knn", "svm", or "rfc"

    Returns:
        (tuple): Tuple containing: a `list` containing the refitted pipelines with the best hyperparameters identified in the search, and a `list` containing the training times for each of the pipelines.
    """
    models = []
    training_times = []
    for model_type in models_to_try:
        if model_type not in ["mnb", "knn", "svm", "rfc", "xgb"]:
            raise ValueError(
                "Please choose valid model_type. Options are mnb, knn, svm, xgb or rfc"
            )
        else:
            if target == "sentiment":
                num_classes = len(np.unique(Y_train))
                pipe, params = create_sklearn_pipeline_sentiment(
                    model_type,
                    num_classes=num_classes,
                    tokenizer=None,
                    additional_features=additional_features,
                )
            else:
                pipe, params = create_sklearn_pipeline(
                    model_type, additional_features=additional_features
                )
            start_time = time.time()
            search = RandomizedSearchCV(
                pipe,
                params,
                scoring="f1_macro",
                n_iter=100,
                cv=4,
                n_jobs=-2,
                refit=True,
            )
            search.fit(X_train, Y_train)
            models.append(search.best_estimator_)
            training_time = round(time.time() - start_time, 0)
            training_times.append(str(datetime.timedelta(seconds=training_time)))
    return models, training_times


def create_and_train_svc_model(X_train, Y_train):
    """Creates pipeline with a Support Vector Classifier using specific hyperparameters identified through previous gridsearching.

    Args:
        X_train (pd.DataFrame): DataFrame containing the features to be fed into the estimator
        Y_train (pd.DataFrame): DataFrame containing the targets

    Returns:
        (tuple): Tuple containing: a fitted `pipeline` with a MultiOutputClassifier utilising a Support Vector Classifier estimator, and a `str` of the training time taken for the fitting of the pipeline.
    """
    cat_transformer = OneHotEncoder(handle_unknown="ignore")
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=0, ngram_range=(1, 2))
    preproc = make_column_transformer(
        (cat_transformer, ["FFT_q_standardised"]),
        (vectorizer, "FFT answer"),
    )
    pipe = make_pipeline(
        preproc,
        MultiOutputClassifier(
            SVC(
                C=15,
                probability=True,
                class_weight="balanced",
                max_iter=1000,
                cache_size=1000,
            ),
        ),
    )
    start_time = time.time()
    pipe.fit(X_train, Y_train)
    training_time = round(time.time() - start_time, 0)
    training_time = str(datetime.timedelta(seconds=training_time))
    return pipe, training_time
