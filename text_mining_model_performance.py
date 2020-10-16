#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 08:24:05 2020

@author: 
"""

gscv = joblib.load('finalized_model_4444.sav')

estimator_position = len(gscv.best_estimator_) - 1
best_estimator = gscv.best_estimator_[estimator_position]
gscv.best_estimator_.steps.pop(estimator_position)
gscv.best_estimator_.steps.append(('clf', best_estimator))

print('The best estimator is %s' % (gscv.best_estimator_[estimator_position]))
print('The best parameters are:')
for param, value in gscv.best_params_.items():
    print('{}: {}'.format(param, value))
print('The best score for scorer "%s" is %s' 
      % (gscv.scorer_, round(gscv.best_score_, 2)))

gscv.best_estimator_.fit(X_train, y_train)
pred = gscv.best_estimator_.predict(X_test)

print('Model accuracy is %s percent' 
      % (int(gscv.best_estimator_.score(X_test, y_test) * 100)))

cm = metrics.confusion_matrix(y_test, pred)
print('Confusion matrix:\n %s' % DataFrame(cm))
metrics.plot_confusion_matrix(gscv.best_estimator_, X_test, y_test)
#ConfusionMatrixDisplay(cm).plot()


tuning_results = DataFrame(gscv.cv_results_)
print(tuning_results.columns)
tuned_learners = []
for i in tuning_results['param_clf__estimator']:
    tuned_learners.append(i.__class__.__name__)
tuning_results['learner'] = tuned_learners

print('Plotting performance of the different models')
p_compare_models = sns.boxplot(x="learner", y="mean_test_score",
            data=tuning_results)
p_compare_models.set_xticklabels(g.get_xticklabels(), rotation=90)
p_compare_models.set(xlabel=None, ylabel=str(gscv.scorer_),
                     title='Mean test score for each (hyper)parameter combination')
