# Using a trained model

The `current_best_multilabel` folder should contain a fully trained `sklearn` model in .sav format, as well as performance metrics for the model.

The Transformer-based `tensorflow.keras` model is over 1GB and cannot be shared via GitHub. However, it will be made available via the API, which is forthcoming in a future release of this package.

This page breaks down the steps in the function `pxtextmining.pipelines.factory_predict_unlabelled_text.predict_multilabel_sklearn`, which can make predictions using the `sklearn` model available via GitHub. This is a high-level explanation of the processes; for more detailed technical information please see the relevant code reference page.

```python

# Step 1: Conduct preprocessing on text:
# Temove trailing whitespaces, NULL values, NaNs, and punctuation. Converts to lowercase.
text_no_whitespace = text.replace(r"^\s*$", np.nan, regex=True)
text_no_nans = text_no_whitespace.dropna()
text_cleaned = text_no_nans.astype(str).apply(remove_punc_and_nums)
processed_text = text_cleaned.astype(str).apply(clean_empty_features)

# Step 2: Make predictions with the trained model
binary_preds = model.predict(processed_text)

# Step 3: Get predicted probabilities for each label
pred_probs = np.array(model.predict_proba(processed_text))

# Step 4: Some samples do not have any predicted labels.
# For these, take the label with the highest predicted probability.
predictions = fix_no_labels(binary_preds, pred_probs, model_type="sklearn")

# Step 5: Convert predictions to a dataframe.
preds_df = pd.DataFrame(predictions, index=processed_text.index, columns=labels)
preds_df["labels"] = preds_df.apply(get_labels, args=(labels,), axis=1)

```
