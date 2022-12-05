blablabla

The text dataset is loaded either as CSV from folder [`datasets`](https://github.com/CDU-data-science-team/pxtextmining/tree/main/datasets)
or is loaded directly from the database. (Loading from/writing to the database
is for internal use only.) Because `Excel` can cause all
sorts of issues with text encodings, it may be best to use
[`LibreOffice`](https://www.libreoffice.org/).
The `results` folders (e.g. [`results_label`](https://github.com/CDU-data-science-team/pxtextmining/tree/main/results_label)) always contain a SAV
of the fitted model and a PNG of the learner comparison bar plot. Results tables
are written as CSV files in a "results_" folder. All results files and folders
have a  "_target_variable_name" suffix, for example "tuning_results_label.csv"
if the  dependent variable is `label`.

Here is a visual display of the process:

![](https://raw.githubusercontent.com/CDU-data-science-team/pxtextmining/main/text_classification_package_structure.png)
