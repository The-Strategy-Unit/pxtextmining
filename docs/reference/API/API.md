# pxtextmining APIs

We have created two different APIs for labelling patient experience feedback. Both APIs are free to use and completely open source. For help and support with using them, please contact (Chris Beeley)[chris.beeley1@nhs.net].

The "Quick API" is faster and simpler, as it uses an sklearn model which is quicker to make predictions. The performance of predictions from this API can be seen on our project documentation website. It is less accurate than the slow API. This API is a more 'traditional' style of API.

The "Slow API" utilises sklearn models as well as the slower but more powerful transformer-based Distilbert model. Due to the demanding hardware requirements of this model, we have set up a slower and slightly more complex API which combines (ensembles) together these models but has higher performance overall.

## Security

The data is submitted via a secure HTTPS connection. All data is encrypted in transit with HTTPS, using the SSL/TLS protocol for encryption and authentication. The data is stored in blob storage on a UK-based Azure container instance for the duration of the model predictions, and is then immediately deleted. Ad hoc support is provided where possible, no uptime or other guarantees exist.
