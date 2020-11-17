#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 08:39:26 2020

@author: 
"""

##############################################################################
# Load and prepare data
# ------------------------------------
print('Loading dataset...')
filename = "text_data_4444.csv"
text_data = pd.read_csv(filename)
text_data = text_data.rename(columns={'super': 'target'})
type(text_data)

# Strip \r and \n from the text
print('Stripping whitespaces and line brakes from text...')
for text, index in zip(text_data['improve'], text_data.index):
    text_data['improve'][index] = " ".join(text.splitlines())

#############################################################################
# Calculate polarity and subjectivity of feedback and add to data
# ------------------------------------
# Polarity and subjectivity will hopefully add an extra piece of
# useful information in the feature set for the models to learn from.
"""text_polarity = []
text_subjectivity = []
for i in range(len(text_data)):
    tb = TextBlob(text_data['improve'][i])
    text_polarity.append(tb.sentiment.polarity)
    text_subjectivity.append(tb.sentiment.subjectivity)

text_data['comment_polarity'] = text_polarity
text_data['comment_subjectivity'] = text_subjectivity

# It may however not work, because TextBlob seems to fail to capture the 
# positive polarity of comments for label "Couldn't be improved".
text_data[text_data['target'] == "Couldn't be improved"]['comment_polarity'].hist()

# Force polarity to always be 1 for "Couldn't be improved"
text_data.loc[text_data['target'] == "Couldn't be improved", 'comment_polarity'] = 1
text_data[text_data['target'] == "Couldn't be improved"]['comment_polarity'].hist()

# NOTE: Problem with forcing polarity to always be 1 for Can't be improved is
# that we simply cannot do that for incoming, unclassified comments. Since 
# TextBlob isn't doing a very good job at detecting positive polarity for 
# this class, it wouldn't be a reliable feature.
"""
#############################################################################
# Apply Spacy's Named Entity Recognition (NER)
# ------------------------------------
# Create a function that detects NERs and run it on all text records.
# Then replace each word that has a NER with its NER type.
# For identification purposes, place NER types between '_',
# e.g. 'CARDINAL' will become '_CARDINAL_'.

"""nlp = spacy.load("en_core_web_sm")

def ner_detector(text):
    ner = []
    doc = nlp(text)
    for ent in doc.ents:
        ner.append((ent.text, ent.label_))
    return ner

if os.path.isfile('text_data_4444_ner.csv'):
    filename = "text_data_4444_ner.csv"
    text_data = pd.read_csv(filename)
else:
    ner = text_data['improve'].map(ner_detector)
    text = text_data['improve'].copy()
    
    for index, value in zip(text_data.index, ner):
        if ner[index]:
            original_string = value[0][0]
            replacer_string = '_' + value[0][1] + '_'
            text[index] = text[index].replace(original_string, replacer_string)
        text_data['improve'] = text"""

#############################################################################
# Split a training set and a test set
# ------------------------------------
#X = text_data['improve']  # This way it's a series. Don't do text_data.drop(['target'], axis=1) as TfidfVectorizer() doesn't like it
#X = text_data[['improve', 'comment_polarity', 'comment_subjectivity']]
print('Preparing training and test sets...')
X = pd.DataFrame(text_data['improve']) # Safest bet is to always have X as a DataFrame. That way, the column selector in the preprocessor doesn't complain
y = text_data['target'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33,
                                                    stratify=y,
                                                    shuffle=True,
                                                    random_state=42
                                                    )