import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow as tf

def multi_label_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_pred = tf.math.round(y_pred)
    exact_matches = tf.math.reduce_all(y_pred == y_true, axis=1)
    exact_matches = tf.cast(exact_matches, tf.float32)
    return tf.math.reduce_mean(exact_matches)


def class_balance_accuracy_score(y_true, y_pred):
    """
    Function for Class Balance Accuracy scorer
    (p. 40 in [Mosley 2013](https://lib.dr.iastate.edu/cgi/viewcontent.cgi?article=4544&context=etd)).

    :param array y_true: True classes, shape = [n_samples].
    :param array y_pred: Predicted classes, shape = [n_samples].
    :return: The Class Balance Accuracy score.
    :rtype: float
    """

    cm = confusion_matrix(y_true, y_pred)
    c_i_dot = np.sum(cm, axis=1)
    c_dot_i = np.sum(cm, axis=0)
    cba = []
    for i in range(len(c_dot_i)):
        cba.append(cm[i][i] / max(c_i_dot[i], c_dot_i[i]))
    cba = sum(cba) / (i + 1)
    return cba
