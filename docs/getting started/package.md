# Package structure

## pxtextmining

The `pxtextmining` package is constructed using the following elements:

- **`pxtextmining.factories`**
This module contains vast majority of the code in the package. There are five different stages, each corresponding to a different submodule.

      - `factory_data_load_and_split`: Loading of multilabel data, preprocessing, and splitting into train/test/validation sets as appropriate.

      - `factory_pipeline`: Construction and training of different models/estimators using the `sklearn`, `tensorflow.keras` and `transformers` libraries.

      - `factory_model_performance`: Evaluation of a trained model, to produce performance metrics. The decision-making process behind the peformance metrics chosen can be seen on the [project documentation website](https://cdu-data-science-team.github.io/PatientExperience-QDC/pxtextmining/performance_metrics.html). The current best performance metrics can be found in the `current_best_multilabel` folder in the main repository.

      - `factory_predict_unlabelled_text`: More info here

- **`pxtextmining.helpers`**
This module contains some helper functions which are used in `pxtextmining.factories`. Some of this is legacy code, so this may just be moved into the `factories` submodule in future versions of the package.

- **`pxtextmining.pipelines`**
All of the processes in `pxtextmining.factories` are pulled together in `multilabel_pipeline`, to create the complete end-to-end process of data processing, model creation, training, evaluation, and saving.

There is also a `pxtextmining.params` file which is used to standardise specific variables that are used across the entire package. The aim of this is to reduce repetition across the package, for example when trying different targets or model types.

## API

Separate from the `pxtextmining` package is the API, which can be found in the folder `api`. It is constructed using FastAPI and Uvicorn. The aim of the API is to
