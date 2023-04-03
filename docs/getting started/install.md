## Installation

You can install `pxtextmining` from either [PyPI](https://pypi.org/project/pxtextmining/) or [GitHub](https://github.com/CDU-data-science-team/pxtextmining).

The recommended method is to clone the repository from GitHub, as this will also include the models and datasets.

### Option 1: Install from PyPI
This option allows you to use the functions coded in pxtextmining.

1. Install `pxtextmining` and its PyPI dependencies:
      - `pip install pxtextmining`
2. We also need to install the [`spaCy`](https://github.com/explosion/spacy-models) model used in [`pxtextmining.helpers.tokenization`](../reference/pxtextmining/helpers/tokenization.md)
   Note that the second model is pretty large, so the installation may take a while.
      - `python -m spacy download en_core_web_lg`


### Option 2 (RECOMMENDED): Install from GitHub
This option is recommended as it gives you access to the full datasets and already trained models.

1. To begin with, [clone the repository from github](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository).

2. It is also recommended to [create a new virtual environment](https://docs.python.org/3/library/venv.html), using your chosen method of managing Python environments.

3. The package uses `poetry` for dependency management. First, run `pip install poetry`.

4. Then, run `poetry install --with dev`.

5. We also need to install the [`spaCy`](https://github.com/explosion/spacy-models) model used in [`pxtextmining.helpers.tokenization`](../reference/pxtextmining/helpers/tokenization.md)
   Note that the second model is pretty large, so the installation may take a while.
      - `python -m spacy download en_core_web_lg`
