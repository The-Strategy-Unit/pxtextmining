#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 08:24:05 2020

@author: 
"""

gscv = joblib.load('finalized_model_4444.sav')

"""# Extract best estimator and replace ClfSwitcher() with it in the pipeline
aux = pd.DataFrame(gscv.best_params_.items())
best_estimator = aux[aux[0] == 'clf__estimator'].reset_index()[1][0]
estimator_position = len(gscv.best_estimator_) - 1
gscv.best_estimator_.steps.pop(estimator_position)
gscv.best_estimator_.steps.append(('clf', best_estimator))"""

# Print various results and metrics
print('The best estimator is %s' % (gscv.best_estimator_.named_steps['clf'].estimator))
print('The best parameters are:')
for param, value in gscv.best_params_.items():
    print('{}: {}'.format(param, value))
print('The best score from the cross-validation for \n the supplied scorer (' +
      refit +  ') is %s' 
      % (round(gscv.best_score_, 2)))

# Predict on test dataset
pred = gscv.best_estimator_.predict(X_test)
y_pred_and_x_test = pd.DataFrame(pred, columns=['pred'])
y_pred_and_x_test['improve'] = X_test['improve'].values
y_pred_and_x_test.to_csv('y_pred_and_x_test.csv', index=False)

# Evaluate on test dataset
print('Model accuracy on the test set is %s percent' 
      % (int(gscv.best_estimator_.score(X_test, y_test) * 100)))
print('Balanced accuracy on the test set is %s percent' 
      % (int(balanced_accuracy_score(y_test, pred) * 100)))
print('Class balance accuracy on the test set is %s percent' 
      % (int(class_balance_accuracy_score(y_test, pred) * 100)))
print('Matthews correlation on the test set is %s ' 
      % (round(matthews_corrcoef(y_test, pred), 2)))

cm = metrics.confusion_matrix(y_test, pred)
print('Confusion matrix:\n %s' % pd.DataFrame(cm))
metrics.plot_confusion_matrix(gscv.best_estimator_, X_test, y_test)
#ConfusionMatrixDisplay(cm).plot()

# Accuracy per class
accuracy_per_class = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 
accuracy_per_class = pd.DataFrame(accuracy_per_class.diagonal())
accuracy_per_class.columns = ['accuracy']
unique, frequency = np.unique(y_test, return_counts = True)
accuracy_per_class['class'], accuracy_per_class['counts'] = unique, frequency
accuracy_per_class = accuracy_per_class[['class', 'counts', 'accuracy']]
accuracy_per_class.to_csv('accuracy_per_class.csv', index=False)

# Plot all tuning results for all learners to be able to compare performances
tuning_results = pd.DataFrame(gscv.cv_results_)
print(tuning_results.columns)
tuned_learners = []
for i in tuning_results['param_clf__estimator']:
    tuned_learners.append(i.__class__.__name__)
tuning_results['learner'] = tuned_learners

# Save as CSV
y_axis = 'mean_test_' + refit
aux = tuning_results.sort_values(y_axis, ascending=False)
#aux = aux.filter(regex='mean|learner')
aux.to_csv('tuning_results_' + refit.lower().replace(' ', '_') + '.csv')

#############################################################################
# Plot results
# ------------------------------------
# Boxplots of average (over all CV folds) learner performance for main scoring 
# measure (par 'refit' in grid search) for each (hyper)parameter combination.
# Par plots of average (over all CV folds) learner performance for different 
# scoring measures for the best (hyper)parameter combination for each learner.
print('Plotting performance of the different models')
# Boxplots
"""p_compare_models_box = sns.boxplot(x="learner", y=y_axis,
            data=tuning_results)
p_compare_models_box.set_xticklabels(p_compare_models_box.get_xticklabels(), 
                                 rotation=90)
p_compare_models_box.set(xlabel=None, ylabel=refit,
                     title='Mean test score for each (hyper)parameter combination')"""

# Bar plots
aux = tuning_results.filter(regex='mean_test|learner').groupby(['learner']).max().reset_index()
aux = aux.sort_values([y_axis], ascending=False)
print(aux)
aux = aux.melt('learner')
aux['variable'] = aux['variable'].str.replace('mean_test_', '')
aux['learner'] = aux['learner'].str.replace('Classifier', '')

p_compare_models_bar = sns.barplot(x='learner', y='value', hue='variable', 
                               data=aux)
p_compare_models_bar.figure.set_size_inches(15, 13)
p_compare_models_bar.set_xticklabels(p_compare_models_bar.get_xticklabels(), 
                                 rotation=90)
plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
p_compare_models_bar.set(xlabel=None, ylabel=None,
                     title='Learner performance ordered by ' + refit)
filename = "p_compare_models_bar_" + refit.lower().replace(' ', '_') + ".png"
p_compare_models_bar.figure.savefig(filename)
