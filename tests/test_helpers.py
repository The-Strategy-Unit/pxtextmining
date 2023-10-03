import numpy as np

from pxtextmining.helpers.text_preprocessor import tf_preprocessing


def test_text_preprocessor(grab_test_X_additional_feats):
    data = grab_test_X_additional_feats["FFT answer"]
    X_pad, vocab_size = tf_preprocessing(data)
    assert isinstance(X_pad, np.ndarray) is True
    assert len(X_pad) == data.shape[0]
    assert isinstance(vocab_size, int) is True
