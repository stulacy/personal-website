+++
date = 2016-02-04
draft = false
tags = ["predictive modelling"]
title = "How to correctly use cross-validation in predictive modelling"
math = false
+++

While writing the [previous post](http://stuartlacy.co.uk/2016/01/26/appreciating-the-distinction-between-explanatory-and-predictive-modelling/) on the two 'cultures' of statistical modeling for prediction and inference, I realised that I was glossing over an extremely important area of predictive modeling, and judging by frequent [StackExchange](http://stats.stackexchange.com/) posts, one that is often misunderstood. As you will have summarised from the title, I'm talking about _cross-validation_.

If done correctly, cross-validation (CV) will provide a thorough assessment of a predictive model providing you with: unbiased, publishable results; a means of selecting the final model instance to use for your application; and an accurate estimate of the model's performance on future data. However, if done badly it can be prove worse than not using it at all. For instance, you may have determined your model to correctly mark 95% of all received spam emails, however, in practice it actually only predicts 90%, leading to complaints from your customers.

## What is cross-validation?

Cross-validation (CV) is a resampling technique that enables you to reuse your data to build multiple models; this comes in handy in two main situations which are explored in further detail in the following sections. 

Its main use in predictive modeling derives from the desire to obtain an estimate of how well the model will perform to future data, this is in contrast to exploratory modeling whereby the objective is to build the model which best explains the variance in the entire sample data, from which to infer estimates for the population parameters. Previously, evaluation predictive models was achieved by segmenting the available data into two groups, the training and test sets, whereby models were fitted to the training observations and assessed on the test set. Their accuracy (or other evaluation criterion) on the test set represents the expected performance of the model to future data, as will be encountered in an applied system.

However, the main problem with this is that there is a large amount of variance in subsetting the data into two groups, unless it is a particularly large sample (I'd use 10,000 samples as a rule of thumb for a 'large' data set). The easier to predict patterns could have been randomly allocated to the test set, thereby providing an optimistic estimate of predictive power, or vice versa. Furthermore, if you have multiple competing models from which you need to select a final implementation and decide to choose the model with the highest test set score, this maximum score could again be due to chance as model accuracy is a random variable.

A better way of comparing models - particularly when the data set is not too large - is to _resample_ the data to allow multiple estimates of predictive performance. This is what CV does, and despite its relative simplicity, using it in practice can still cause confusion; I've seen firsthand examples of academic papers published in the Machine Learning community where CV was incorrectly employed, resulting in overly optimistic evaluations of the specific algorithm that was being introduced. This post will attempt to explain the rationale behind using CV, along with providing detailed instructions so that you can successfully employ it in your own work with confidence, and notice when others are incorrectly using it.

There are several different implementations of CV, all revolving around the same core theme. I'll discuss _k-fold CV_ in this post as it's probably the most commonly used version. The algorithm works as follows:

  1. Split the data into _k_ equally sized groups (known as folds), common choices for k are 10 or 5.
  2. Iterate through k times the following:
    a. Fit a model to the data found in every fold **except** fold k.
    b. Assess this model on the 'held-out' fold, k

## What is CV used for?

Before you go ahead and try to blindly apply the above algorithm, you need to know when to use CV. In general, it is used in two common situations in the predictive modeling pipeline:

  1. Model **selection**
  2. Model **evaluation**

## Model selection

Say you want to compare a linear logistic regression model to a Support Vector Machine (SVM) to decide which to use for your application. As mentioned above, comparing the scores on the data the models are fit to doesn't provide any information about the predictive powers of the two competing models and thereby encourages overfitting, while the use of a single holdout test set is itself subject to chance variation (known as selection bias). By instead following the k-fold CV procedure outlined above, you will obtain k scores (whether they be accuracy, AUC, RMSE, kappa, etc...) for each of the competing models. Each of these scores provides a measure of how well that model fared on unseen data, and so by taking an average of the k measures (and providing confidence intervals) you obtain an overall perspective of which model is better at predicting unseen patterns.

Selecting which model to use typically involves picking the model with the maximum (or minimum in certain contexts) **cross-validated score**, i.e. the mean of the k scores. However, this could also result in choosing a model which happened to perform the best due to chance alone, and so an alternative approach is to select the simplest model (i.e. with the least parameters) which is still within one standard error of the maximum score. This method allows you to choose a high performing model, which might be less overfit to these particular k validation folds.

It is also important to choose a sensible value of k here, as low values will still be subject to large amounts of variance. I typically use a method called _repeated k-fold cross-validation_, whereby the entire k-fold process is repeated a set number of times. I generally use 5 repeats of 10-folds, resulting in 50 scores to select my final model from.

Note that model selection isn't just used for picking between models built under differing learning algorithms: it can also be used to select hyper-parameter values (such as the choice of SVM kernel function and cost), or between different combinations of learning algorithm and hyper-parameters. Note that the more models you include in your selection pool, the more likely you are to find a high performing model on the test folds by chance, it's important to use subjective knowledge to include a range of appropriate learning algorithms for each application, along with a large enough number of CV iterations. Experience will help in this regard.

The final point to make about model selection is how you build your final model instance to be used in practice. Say you were comparing logistic regression models with linear SVMs with 5 different cost values, and discovered that out of 10 folds, the linear SVM with c=0.01 had the highest mean accuracy and is therefore selected to be the final model. The only problem is that you have 10 instances of this setup from each iteration of the CV loop. 

### How do you choose which of the k instances to use as your final model to predict future patterns?

The answer is simply fit a linear SVM with c=0.01 onto the entire data set; the cross-validation process merely selects the model building configuration, you need to actually build the final model instance afterwards. It is important in this stage to therefore view model selection as a process of selecting the model **configuration** to be used, not the actual instance of a model itself.

## Model evaluation

Now that you have built your model that you will use in your application to predict future data patterns, it would be useful have an idea of how well it will perform. You may think that you could use the cross-validated score obtained during the model selection phase; however, this measure is **biased** and unrepresentative of true generalisation performance. The bias stems from the fact that you choose that particular model configuration (its learning algorithm and/or hyper-parameter values) based on its high cross-validation score, it is probable that its accuracy on a new unseen data set would therefore regress to the mean.

Likewise with the model selection process, the generalisation measure shouldn't be tied to a specific model instance, but rather your whole workflow of model selecting and building. What I mean by this is that you need to obtain a score of how well a model selected by the above process would perform on a completely unseen data set, and preferably multiple repeats of this process to reduce selection bias. Yep, you've guessed it, we're going to need cross-validation!

At each iteration of the CV loop we need to select and build a model on the k-1 training folds, and then assess its performance on the kth test fold. For simple learning algorithms with no hyper-parameters to tune, this simply involves fitting a model instance to the training folds. However, it gets a little more involved with more complex systems.

If we wish to evaluate our SVM building process, which involves tuning over five values of cost, then at each iteration of the evaluation cross-validation loop we need to run a **secondary cross-validation process** over the k-1 training folds, performing the exact same role as that described in the model selection section above. This will obtain an optimal cost value for fold k, you then fit a model instance with that particular hyper-parameter value on the k-1 training folds and assess on the test fold. Save the test fold scores (of the outer CV loop, you're not obliged to save the inner CV results unless you're particularly interested in the parameter tuning), and calculate the mean (with confidence intervals) to use as your overall unbiased predicted accuracy for new data.

It's important to note that on each of the outer CV iterations, your model being evaluated may be built using different values of the hyper-parameters, i.e. on fold 1 c=3 was selected on the inner CV loops as optimal, while on fold 2 it was c=5 etc... This is desired behaviour, the cross-validated score represents your entire model building process, not that of a specific model instance. It means that if you were to run your model selection process, comprising selecting from multiple learning algorithms and hyper-parameter values with CV, you would expect to see an accuracy of X on future unseen data, regardless of what specific model configuration was selected for the final model instance.

A quick example to reinforce this distinction between the outer evaluation loop and the inner selection CV then. Say I'm running k-fold CV to evaluate my model with j-fold CV used to tune hyper-parameters on a data set with 1,000 observations, and k=10, j=5:

  - Split the data into 10 folds of 100 observations each
  - For each outer fold 1..k do:
    - Combine the data from folds != k into a training set of 900
    - Split the training data into 5 folds of 180 patterns
    - For each inner fold 1..j do:
      - Form the training data from all folds != j, this will contain 720 patterns
      - For each possible hyper-parameter value:
          - Fit a model with this value on the 720 inner training patterns
          - Evaluate this model on the remaining jth test fold
    - Calculate the mean inner cross-validated score for each hyper-parameter value
    - Select the hyper-parameter value with the maximum cross-validated score (or minimum complexity model 1 SE away, see above)
    - Fit a model instance to the k-1 training folds (900 patterns) using this selected model configuration
    - Assess this model's goodness on the k test fold
  - Calculate the mean cross-validated score across the k test folds and use as your estimate of model performance on new data

## Summary

It's very important to understand the role of cross-validation and how to employ it correctly to ensure that you aren't biasing your evaluation measures, and can adequately tune your model building process. It's not a complex process, however when you're running the nested CV setup for accurate model evaluation it can be confusing keeping track of which loop is which. During the nested CV run for model evaluation, I try to mentally separate the three stages run at each fold of the outer loop:

  - **Selecting** an optimal hyper-parameter value or learning algorithm selection with a secondary CV loop on the k-1 outer training folds
  - **Building** a model instance with the tuned model configuration on the k-1 training folds
  - **Evaluating** this model instance on the kth test fold

A typical workflow for me would involve running the full nested CV loop first to see what typical performance I'd expect for a given application, before building my final model with a separate single CV loop, thereby not introducing any bias into my estimated performance. Often I'm interested to see on the whole how well different learning algorithms do on a data set, and so will run the nested CV separately for each algorithm, tuning hyper-parameters in the inner loop. This allows me to compare the unbiased performance of each learning algorithm, however, if I were to select a single learning algorithm from these results it would introduce selection bias. I typically use at least 5 repeats of 10-fold CV for model evaluation, often more to reduce my confidence intervals. While for model selection I'm generally content with 10-fold CV.

