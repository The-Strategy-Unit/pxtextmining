#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 08:39:26 2020

@author: 
"""

##############################################################################
# Load and prepare data
# ------------------------------------
filename = "text_data_4444.csv"
text_data = pd.read_csv(filename)
text_data = text_data.rename(columns={'super': 'target'})
type(text_data)

#############################################################################
# Calculate polarity and subjectivity of feedback and add to data
# ------------------------------------
# Polarity and subjectivity will hopefully add an extra piece of
# useful information in the feature set for the models to learn from.
text_polarity = []
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

#############################################################################
# Split a training set and a test set
# ------------------------------------
#X = text_data['improve']  # This way it's a series. Don't do text_data.drop(['target'], axis=1) as TfidfVectorizer() doesn't like it
X = text_data[['improve', 'comment_polarity', 'comment_subjectivity']]
y = text_data['target'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33,
                                                    stratify=y,
                                                    shuffle=True,
                                                    random_state=42
                                                    )