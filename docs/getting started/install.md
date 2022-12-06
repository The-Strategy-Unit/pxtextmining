## Installation

You can install `pxtextmining` from either [PyPI](https://pypi.org/project/pxtextmining/) or [GitHub](https://github.com/CDU-data-science-team/pxtextmining).

### Optional: create a Python Virtual Environment in which to install `pxtextmining` and its dependencies.
This will allow you to keep the package versions required by the package separate from any others that you may have installed. Let's call the virtual environment `text_venv`:

1. Open a terminal, navigate to the folder where you want to put the virtual
   environment and run:
      - `python3 -m venv text_venv` (Linux & MacOS);
      - `python -m venv text_venv` (Windows);
2. Activate the virtual environment. In the folder containing folder `text_venv`
   run:
      - `source text_venv/bin/activate` (Linux & MacOS);
      - `text_venv\Scripts\activate` (Windows);

### Option 1: Install from PyPI

1. Install `pxtextmining` and its PyPI dependencies:
      - `pip3 install pxtextmining`  (Linux & MacOS);
      - `pip install pxtextmining` (Windows);
2. We also need to install a couple of
   [`spaCy`](https://github.com/explosion/spacy-models) models.
      - `python -m spacy download en_core_web_sm`
      - `python -m spacy download en_core_web_lg`

   Note that the second model is pretty large, so the installation may take a
   while.

### Option 2: Install from GitHub

1. To begin with, clone the repository from github.

2. Install wheel, this helps with sorting the different packages and their dependencies:
      - `pip3 install wheel`  (Linux & MacOS);
      - `pip install wheel` (Windows);
3. Install all the dependencies of `pxtextmining`. Inside the repo's folder, run:
      - `pip3 install -r requirements.txt` (Linux & MacOS);
      - `pip install -r requirements.txt` (Windows);

4. Install `pxtextmining` as a Python package. Inside the repo's folder, run:
      - `python3 setup.py install` (Linux & MacOS);
      - `python setup.py install` (Windows);
