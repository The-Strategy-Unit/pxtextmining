import numpy as np
from sklearn.metrics import confusion_matrix


def class_balance_accuracy_score(y_true, y_pred):
    """
    Function for Class Balance Accuracy scorer
    (p. 40 in `Mosley 2013 <https://lib.dr.iastate.edu/cgi/viewcontent.cgi?article=4544&context=etd>`_).

    :param array y_true: True classes, shape = [n_samples].
    :param array y_pred: Predicted classes, shape = [n_samples].
    :return: cba (`float`): The Class Balance Accuracy score.
    """

    cm = confusion_matrix(y_true, y_pred)
    c_i_dot = np.sum(cm, axis=1)
    c_dot_i = np.sum(cm, axis=0)
    cba = []
    for i in range(len(c_dot_i)):
        cba.append(cm[i][i] / max(c_i_dot[i], c_dot_i[i]))
    cba = sum(cba) / (i + 1)
    return cba
