# pxtextmining: Text Classification of Patient Experience feedback

This Docker container contains the pxtextmining machine learning models trained as part of the [Patient Experience Qualitative Data Categorisation project](https://cdu-data-science-team.github.io/PatientExperience-QDC/).

To use this Docker container to predict your unlabelled text:

1. Set up your folders. You will need to set up a folder containing two other folders, data_in and data_out, as below.
```
docker_data/
├─ data_in/
├─ data_out/

```

2. Prepare your data. Save the data you wish to pass through the machine learning models as json, in the data_in folder. The data should be in the following format:

In Python, a `list` containing as many `dict`s as there are comments to be predicted. Each `dict` has three compulsory keys:

  * `comment_id`: Unique ID associated with the comment, in `str` format. Each Comment ID per API call must be unique.
  * `comment_text`: Text to be classified, in `str` format.
  * `question_type`: The type of question asked to elicit the comment text. Questions are different from trust to trust, but they all fall into one of three categories:
       * `what_good`: Any variation on the question "What was good about the service?", or "What did we do well?"
       * `could_improve`: Any variation on the question "Please tell us about anything that we could have done better", or "How could we improve?"
       * `nonspecific`: Any other type of nonspecific question, e.g. "Please can you tell us why you gave your answer?", or "What were you satisfied and/or dissatisfied with?".

```python
# In Python

text_data = [
              { 'comment_id': '1', # The comment_id values in each dict must be unique.
                'comment_text': 'This is the first comment. Nurse was great.',
                'question_type': 'what_good' },
              { 'comment_id': '2',
                'comment_text': 'This is the second comment. The ward was freezing.',
                'question_type': 'could_improve' },
              { 'comment_id': '3',
                'comment_text': '',  # This comment is an empty string.
                'question_type': 'nonspecific' }
            ]

```

```R
# In R

library(jsonlite)

comment_id <- c("1", "2", "3")
comment_text <- c(
  "This is the first comment. Nurse was great.",
  "This is the second comment. The ward was freezing.",
  ""
)
question_type <- c("what_good", "could_improve", "nonspecific")
df <- data.frame(comment_id, comment_text, question_type)
text_data <- toJSON(df)
```

3. Save the JSON data in the data_in folder, as follows:

```python
# In Python

json_data = json.dumps(text_data)
with open("data_in/file_01.json", "w") as outfile:
    outfile.write(json_data)
```

```R
# In R

json_data <- toJSON(text_data, pretty = TRUE)
write(json_data, file = "data_in/file_01.json")
```

4. Your file structure should now look like this:

```
docker_data/
├─ data_in/
│  ├─ file_01.json
├─ data_out/
```

5. Mount the docker_data folder as the `data` volume for the Docker container and run the container. Pass the filename for the input JSON as the first argument. The following arguments are also available:
   - `--local-storage` or `-l` flag for local storage (does not delete the files in data_in after completing predictions)
   - `--target` or `-t` to select the machine learning models used. Options are `m` for multilabel, `s` for `sentiment`, or `ms` for both. Defaults to `ms` if nothing is selected.

A sample command would be:
`docker run --rm -it -v /docker_data:/data ghcr.io/cdu-data-science-team/pxtextmining:latest file_01.json -l `

6. The predictions will be outputted as a json file in the data_out folder, with the same filename. After running successfully, the final folder structure should be:

```
docker_data/
├─ data_in/
│  ├─ file_01.json
├─ data_out/
   ├─ file_01.json
```
