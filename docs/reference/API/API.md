# pxtextmining API

To facilitate the use of the models trained in this project, an API has been created using the FastAPI library. Users will be able to send their patient experience feedback comments to the model via the API, and will receive the predicted labels for those comments.

At this stage in the project (March 2023) it is only available locally as a proof of concept, and has not yet been deployed externally. By May 2023 we are hoping to have a publicly available version of the API, although we will likely only share the URL with partner trusts.

## How to make an API call

1. Prepare the data in JSON format. In Python, this is a `list` containing as many `dict`s as there are comments to be predicted. Each `dict` has two compulsory keys, `comment_id` and `comment_text`. The values of the dict should be in `str` format. For example:

```python
[
  { 'comment_id': '1', # The comment_id values in each dict must be unique.
    'comment_text': 'This is the first comment. Nurse was great.'} ,
  { 'comment_id': '2',
    'comment_text': 'This is the second comment. The ward was freezing.'} ,
  { 'comment_id': '3',
    'comment_text': '' } # This comment is an empty string.
]
```

2. Send the JSON containing the text data to the `predict_multilabel` endpoint.

3. After waiting for the data to be processed and passed through the machine learning model, receive predicted labels at the same endpoint, in the example format below.

```python
[
  { 'comment_id': '1',
    'comment_text': 'This is the first comment. Nurse was great.',
    'labels': ['Staff']} ,
  { 'comment_id': '2',
    'comment_text': 'This is the second comment. The ward was freezing.',
    'labels': ['General', 'Environment & equipment']} ,
  { 'comment_id': '3',
    'comment_text': '',
    'labels': ['Labelling not possible'] }
]
```
