## Installation

You can install `pxtextmining` from either [PyPI](https://pypi.org/project/pxtextmining/) or [GitHub](https://github.com/CDU-data-science-team/pxtextmining).

### Optional: create a Python Virtual Environment in which to install `pxtextmining` and its dependencies.
This will allow you to keep the package versions required by the package separate from any others that you may have installed. Let's call the virtual environment `text_venv`:

*Manually creating virtual environment*
1. Open a terminal, navigate to the folder where you want to put the virtual
   environment and run:
      - `python3 -m venv text_venv` (Linux & MacOS);
      - `python -m venv text_venv` (Windows);
2. Activate the virtual environment. In the folder containing folder `text_venv`
   run:
      - `source text_venv/bin/activate` (Linux & MacOS);
      - `text_venv\Scripts\activate` (Windows);

*Using pyenv to create the virtual environment*
If you have pyenv installed:

1. Run `pyenv virtualenv text_venv` to create the new virtual environment.
2. Activate it with `pyenv activate text_venv`

### Option 1: Install from PyPI
This option allows you to use the functions coded in pxtextmining.

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
This option is recommended as it gives you access to the full datasets and already trained models.

1. To begin with, clone the repository from github.

2. Navigate to the repository folder on your computer. run `pip install .`

3. We also need to install a couple of
   [`spaCy`](https://github.com/explosion/spacy-models) models.
      - `python -m spacy download en_core_web_sm`
      - `python -m spacy download en_core_web_lg`

   Note that the second model is pretty large, so the installation may take a
   while.
