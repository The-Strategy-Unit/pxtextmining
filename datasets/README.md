Please note that the Care Opinion data is being shared under the [CC BY-NC-SA 4.0 licence](https://creativecommons.org/licenses/by-nc-sa/4.0/) and is generated from the [Care Opinion API](https://www.careopinion.org.uk/info/api-v2).


Two out of the six participating trusts have agreed to make their data available publicly.

An explanation of the dataset columns for phase 2 is available below.



Comment ID: ID for the specific comment.

Trust: NHS Trust where comment originated.

Respondent ID: ID for the specific respondent. Not linked to any personal identifiable information.

Date: Date the comment was provided.

Service type 1: Department relating to the comment.

Service type 2: Subdepartment relating to the comment.

FFT categorical answer: Quantitative score attached to the comment. 1 is "very good", 5 is "very poor".

FFT question: The specific question asked by the NHS trust to elicit the qualitative text response.

FFT answer: The qualitative text response provided by the respondent to the FFT question.

Person identifiable info?: Whether or not the FFT answer contains any person identifiable info, as flagged by the labeller.

Comment sentiment: The sentiment score applied to the FFT answer by the labeller. 1 is "very positive", 5 is "very negative". Mixed comments have been labelled as "3", neutral.

All other columns are the qualitative framework labels, in one hot encoded format. The version of the framework being used is reflected in the filename. Full details of the framework are available on the [project documentation website](https://the-strategy-unit.github.io/PatientExperience-QDC/framework/framework3.html).
