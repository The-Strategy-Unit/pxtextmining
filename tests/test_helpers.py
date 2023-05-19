from pxtextmining.helpers.text_preprocessor import tf_preprocessing
from pxtextmining.helpers.tokenization import spacy_tokenizer
import numpy as np

def test_text_preprocessor(grab_test_X_additional_feats):
    data = grab_test_X_additional_feats['FFT answer']
    X_pad, vocab_size = tf_preprocessing(data)
    assert type(X_pad) == np.ndarray
    assert len(X_pad) == data.shape[0]
    assert type(vocab_size) == int

def test_spacy_tokenizer():
    document = 'This is some incredibly interesting text'
    processed = spacy_tokenizer(document)
    assert type(processed) == list
    assert len(processed) == len(document.split())
