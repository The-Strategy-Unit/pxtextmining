# pxtextmining: Text Classification of Patient Experience feedback

## Project description
Nottinghamshire Healthcare NHS Foundation Trust hold  patient feedback that is 
currently manually labelled by our "coders" (i.e. the staff who read the 
feedback and decide what it is about). As we hold thousands of patient feedback 
records, we (the [Data Science team](https://cdu-data-science-team.github.io/team-blog/about.html)) are running 
this project to aid the coders with a text classification pipeline that will 
semi-automate the labelling process. We are also working in partnership with 
other NHS trusts who hold patient feedback text. Read more [here](https://involve.nottshc.nhs.uk/blog/new-nhs-england-funded-project-in-our-team-developing-text-mining-algorithms-for-patient-feedback-data/) and [here](https://cdu-data-science-team.github.io/team-blog/posts/2020-12-14-classification-of-patient-feedback/).

__We are working openly by open-sourcing the analysis code and data where possible to promote replication, reproducibility and further developments (pull requests are more than welcome!). We are also automating common steps in our workflow by shipping the pipeline as a [Python](https://www.python.org/) package broken down into sub-modules and helper functions to increase usability and documentation.__

## Documentation
1. [Installation](#installation);
2. [Execution](#execution);
3. Pipeline [description](#pipeline);
4. Function/class [documentation](https://cdu-data-science-team.github.io/pxtextmining/index.html);

## Installation

We will show how to install `pxtextmining` from both 
[PyPI](https://pypi.org/project/pxtextmining/) or the [GitHub](https://github.com/CDU-data-science-team/pxtextmining) repo.

**Before doing so, it is best to create a Python Virtual Environment[^1] 
in which 
to install `pxtextmining` and its dependencies.** Let's call the virtual 
environment `text_venv`:

1. Open a terminal, navigate to the folder where you want to put the virtual 
   environment and run:
   - `python3 -m venv text_venv` (Linux & MacOS);
   - `python -m venv text_venv` (Windows);
1. Activate the virtual environment. In the folder containing folder `text_venv`
   run:
   - `source text_venv/bin/activate` (Linux & MacOS);
   - `text_venv\Scripts\activate` (Windows);

### Install from PyPI

1. Install `pxtextmining` and its PyPI dependencies:
   - `pip3 install pxtextmining==0.3.4`  (Linux & MacOS);
   - `pip install pxtextmining==0.3.4` (Windows);
1. We also need to install a couple of 
   [`spaCy`](https://github.com/explosion/spacy-models) models. 
   
   These are obtained from URL links and thus need to be installed separately:
   - **Linux & MacOS**
     ```
     pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz
     pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-2.3.1/en_core_web_lg-2.3.1.tar.gz
     ```
   - **Windows**
     ```
     pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz
     pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-2.3.1/en_core_web_lg-2.3.1.tar.gz
     ```
   
   Note that the second model is pretty large, so the installation may take a 
   while.

All steps in one go:

1. **Linux & MacOS**
   ```
   python3 -m venv text_venv
   source text_venv/bin/activate
   pip3 install pxtextmining==0.3.4
   pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz
   pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-2.3.1/en_core_web_lg-2.3.1.tar.gz
   ```
1. **Windows**
   ```
   python -m venv text_venv
   text_venv\Scripts\activate
   pip install pxtextmining==0.3.4
   pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz
   pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-2.3.1/en_core_web_lg-2.3.1.tar.gz
   ```

### Install from GitHub

1. To begin with, download the repo.
1. Install wheel: 
   - `pip3 install wheel`  (Linux & MacOS);
   - `pip install wheel` (Windows);
1. Install all the dependencies of `pxtextmining`. Inside the repo's folder, 
   run: 
   - `pip3 install -r requirements.txt` (Linux & MacOS);
   - `pip install -r requirements.txt` (Windows);
   
   This will also install the `spaCy` models, so no additional commands are 
   required as when installing from [PyPI](#install-from-pypi).  Note that the 
   second model is pretty large, so the installation may take a while.
1. Install `pxtextmining` as a Python package. Inside the repo's folder, 
   run: 
   - `python3 setup.py install` (Linux & MacOS);
   - `python setup.py install` (Windows);

All steps in one go:

1. **Linux & MacOS**
   ```
   python3 -m venv text_venv
   source text_venv/bin/activate
   pip3 install wheel
   pip3 install -r requirements.txt
   python3 setup.py install
   ```
1. **Windows**
   ```
   python -m venv text_venv
   text_venv\Scripts\activate
   pip install wheel
   pip install -r requirements.txt
   python setup.py install
   ```

## Execution

Our example scripts are saved in folder [execution](https://github.com/CDU-data-science-team/pxtextmining/tree/main/execution). 
The execution scripts are nothing more than a call of function 
`pxtextmining.pipelines.text_classification_pipeline` with user-specified
arguments. The two example scripts, `execution_label.py` and
`execution_criticality.py` run the pipeline for each of the two target variables
in [datasets](https://github.com/CDU-data-science-team/pxtextmining/tree/main/datasets). 
Note that `execution_criticality.py` runs ordinal classification 
(`ordinal=True`).

Users can create their own execution script(s). Run the script in a Python 
IDE (Integrated Development Environment) or on the terminal (do not forget to 
activate the virtual environment first) with:

   - `python3 execution/<script_name.py>` (Linux & MacOS).
   - `python execution/<script_name.py>` (Windows);

For example:

   - `python3 execution/execution_label.py` (Linux & MacOS).
   - `python execution/execution_label.py` (Windows);

The results will be saved in a "results" folder such as [results_label](https://github.com/CDU-data-science-team/pxtextmining/tree/main/results_label).

## Pipeline

The pipeline is built with Python's 
[`Scikit-learn`](https://scikit-learn.org/stable/index.html) (Pedregosa et al., 2011). 
During fitting, both the "Bag-of-Words" approach and a word embedding-based 
approach are tried out. The pipeline performs a random grid search ([`RandomizedSearchCV()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV)) to identify the best-performing learner 
and (hyper)parameter values. The process also involves a few pre- and post-fitting steps:

1. Data load and split into training and test sets ([`factory_data_load_and_split.py`](https://cdu-data-science-team.github.io/pxtextmining/pxtextmining.factories.html#module-pxtextmining.factories.factory_data_load_and_split)).

2. Text pre-processing (e.g. remove special characters, whitespaces and line breaks) and tokenization, token lemmatization, calculation of Term Frequency-Inverse Document Frequencies (TF-IDFs), up-balancing of rare classes, feature selection, pipeline training and learner benchmarking ([`factory_pipeline.py`](https://cdu-data-science-team.github.io/pxtextmining/pxtextmining.factories.html#module-pxtextmining.factories.factory_pipeline)).

3. Evaluation of pipeline performance on test set, production of evaluation 
metrics (Accuracy score, 
[Class Balance Accuracy](https://lib.dr.iastate.edu/cgi/viewcontent.cgi?article=4544&context=etd) (Mosley, 2013), 
[Balanced Accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html) (Guyon et al., 2015, Kelleher et al., 2015) or 
[Matthews Correlation Coefficient](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html) (Baldi et al., 2000, Matthews, 1975)) and plots, and fitting of best performer 
on whole dataset 
([`factory_model_performance.py`](https://cdu-data-science-team.github.io/pxtextmining/pxtextmining.factories.html#module-pxtextmining.factories.factory_model_performance)).

4. Writing the results: fitted pipeline, tuning results, predictions, accuracy 
per class, model comparison bar plot, training data index, and test data index ([`factory_write_results.py`](https://cdu-data-science-team.github.io/pxtextmining/pxtextmining.factories.html#module-pxtextmining.factories.factory_write_results)).

5. Predicting unlabelled text ([`factory_predict_unlabelled_text.py`](https://cdu-data-science-team.github.io/pxtextmining/pxtextmining.factories.html#module-pxtextmining.factories.factory_predict_unlabelled_text)).

There are a few helper functions and classes available in the [helpers](https://cdu-data-science-team.github.io/pxtextmining/pxtextmining.helpers.html#pxtextmining-helpers-package) 
folder that the aforementioned factories make use of.

The factories are brought together in a single function [`text_classification_pipeline.py`](https://cdu-data-science-team.github.io/pxtextmining/pxtextmining.pipelines.html#module-pxtextmining.pipelines.text_classification_pipeline) that runs the whole process. This function can be run in a user-made 
script such as 
[`execution/execution_label.py`](https://github.com/CDU-data-science-team/pxtextmining/blob/main/execution/execution_label.py). 
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

## References
Baldi P., Brunak S., Chauvin Y., Andersen C.A.F. & Nielsen H. (2000). Assessing 
the accuracy of prediction algorithms for classification: an overview. 
_Bioinformatics_  16(5):412--424.

Guyon I., Bennett K. Cawley G., Escalante H.J., Escalera S., Ho T.K., Macià N., 
Ray B., Saeed M., Statnikov A.R, & Viegas E. (2015). [Design of the 2015 ChaLearn AutoML Challenge](https://ieeexplore.ieee.org/document/7280767), 
International Joint Conference on Neural Networks (IJCNN).

Kelleher J.D., Mac Namee B. & D’Arcy A.(2015). 
[Fundamentals of Machine Learning for Predictive Data Analytics: Algorithms, Worked Examples, and Case Studies](https://mitpress.mit.edu/books/fundamentals-machine-learning-predictive-data-analytics). 
MIT Press.

Matthews B.W. (1975). Comparison of the predicted and observed secondary 
structure of T4 phage lysozyme. _Biochimica et Biophysica Acta (BBA) - Protein Structure_ 405(2):442--451.

Pedregosa F., Varoquaux G., Gramfort A., Michel V., Thirion B., Grisel O., 
Blondel M., Prettenhofer P., Weiss R., Dubourg V., Vanderplas J., Passos A., 
Cournapeau D., Brucher M., Perrot M. & Duchesnay E. (2011), 
[Scikit-learn: Machine Learning in Python](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html). 
_Journal of Machine Learning Research_ 12:2825--2830

[^1]: A vritual environment can also be created using Conda, where the commands 
for creating and activating it are a little different. See [this](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
