# Training a new model

To train a new model to categorise patient feedback text, labelled data is required.

Data from phase 1 of the project is available in the folder [`datasets`](https://github.com/CDU-data-science-team/pxtextmining/blob/main/datasets/text_data.csv), or it can also be loaded from an SQL database. If you have your own labelled patient experience feedback, you can use this instead.

The [text_classification_pipeline](../../reference/pipelines/text_classification_pipeline) contains the function required to output a fully trained model. Two types of models can be trained using this pipeline, one that can predict the categorical 'label' for the text, or one that can predict the positive or negative 'criticality' score for the text. Examples of the pipeline being used to output each type of model can be seen in [execution_criticality](https://github.com/CDU-data-science-team/pxtextmining/blob/main/execution/execution_criticality.py) and [execution_label](https://github.com/CDU-data-science-team/pxtextmining/blob/main/execution/execution_label.py).

Here is a visual display of the process:
![](https://raw.githubusercontent.com/CDU-data-science-team/pxtextmining/main/text_classification_package_structure.png)
