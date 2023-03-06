from pxtextmining.factories import factory_data_load_and_split
import pandas as pd



def test_clean_empty_features():
    df_with_empty_lines = pd.DataFrame({'text': ['Some text', '', ' ', 'More text']})
    clean_df = factory_data_load_and_split.clean_empty_features(df_with_empty_lines)
    assert clean_df.shape == (2,1)
