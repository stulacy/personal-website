+++
date = 2016-01-26
draft = false
tags = ["data analysis", "machine learning", "statistics"]
title = "Appreciating the distinction between explanatory and predictive modelling"
math = false
+++

## "Two Cultures"

One aspect of statistical modeling which can be taken for granted by those with a bit of experience, but may not be immediately obvious to newcomers, is the difference between modeling for **explanation** and modeling for **prediction**. When you're a newbie to modeling you may think that this only has an effect on how you interpret your results and what conclusions you're aiming to make, but it has a far bigger impact than that, from influencing the way you form the models, to the types of learning algorithms you use, and even how you evaluate their performance. There is even a difference in terms of the backgrounds of practitioners using these two methods, as statisticians tend to be more concerned with **inference** while applied modelers are typically less interested in the underlying relationships in the data, in favour of being able to predict future events with greatest accuracy. Since many textbooks on modeling are written by statisticians, this is an important aspect which tends to get glossed over at times.

The aim of this post is to provide an overview of the main differences between these two approaches, so that people approaching predictive modeling from a non-statistical background can understand the context in which many modelers work (and thus textbooks are written), allowing them to make the most effective design decisions. I'll describe how the two cultures of predictive and explanatory modeling differ with regards to several different aspects of the model building process.

The idea for this blog came from my own experience learning to model for inference purposes after coming from a predictive background, and reading Leo Breiman's excellent paper [Statistical Modeling: The Two Cultures](https://projecteuclid.org/euclid.ss/1009213726), which rather excellently summarised the confusion I felt.

### Aim

While this should be rather obvious, the motivation for modeling the data is rather different for explanation and predictive analyses. Occasionally, however, people start running their data analysis before asking themselves the important question of "What is it I'm trying to determine?". When doing any form of predictive modeling, the aim is to generate as much predictive power as possible, typically with little regard to how this is achieved (although this is not necessarily always the case). There may be unequal cost distributions or other such considerations but the focus will be on predicting unseen data as the model will likely be used in an applied situation. 

Explanatory modeling, on the other hand, is more concerned with understanding relationships in the data to establish the effects of various predictors on the response. The main application is for statistical inference, the means by which population behaviours are estimated from sample data. This will generally require a more specific research question to be derived before proceeding with any analysis, such as "Does the number of people going to university in a country have an effect on the GDP? And if so, what is the form of this relationship?" Ideally, this research objective should be stated in the form of a null hypothesis before any of the sample data has been examined, with the data taking the form of either trial data (which facilitates simpler conclusions), or observational data (whereby evidence of an association does not guarantee a cause and effect relationship).

### Model choice

Explanatory analysis requires simpler, more interpretable models - typically in the form of a choice from the generalised linear models (GLM) family. These facilitate a clear relationship between the output and the predictor, "A male earns x% more than a woman with all other variables the same". The use of models for statistical inference allows for the investigation of a relationship between two variables, while controlling for confounding effects (such as race and income, while accounting for education level). Linear models also enable hypothesis tests for the significance of covariates. However, they require more manual tweaking as if any of the assumptions aren't met (distribution of the response, independent and identically distributed variables, non-constant residual variance), then the model loses power. This requires a simple model with a large amount of manual input and diagnostics to determine the model which best explains the data. The conclusion may be at the end of the analysis that the current model only explains a certain amount of the variation of the random response variable, and there are other underlying factors which haven't been accounted for.

Predictive modeling on the other hand, is unconcerned with how the model is formed, as long as it can be proven to generalise well to unseen data. This means that while linear models will commonly be employed at first to get a feel for the data, more powerful non-linear techniques will be evaluated to tweak out any extra predictive power, where a 1 percentage point increase in accuracy can result in hundreds more incident conditions being successfully diagnosed each year. These methods, such as SVMs, Random Forests, or boosted trees, can be hard to reasonably interpret beyond a baseline value of which explanatory variables are most important to the prediction.

### Evaluation criteria

Even the criteria used to assess whether a model is 'good' or not depends on the scenario in which it is being employed. Explanatory models are assessed by a multitude of ways, to ensure that they best represent the data. A common measure of fit is the R^2 measure of variance explained by the model, which is related to how much the data deviates from the regression line. Furthermore, you would typically run a suite of diagnostics to ensure that your assumptions are met, for instance that the residuals have a constant variance, are normally distributed, and there is no evidence of a non-linear relationship; regression for inference is primarily concerned with ensuring that your data fits the data best. 

For predictive applications, on the other hand, models tend to be evaluated with a single criterion measuring their performance at predicting some unseen data. The use of complex non-linear models means that it's unlikely for the residuals to be non-linearly distributed for instance, as the model itself can form non-linear relationships with several orders of interactions. While an inference problem trying to model a binary outcome (for instance if an unborn child will be male or female) will use the standard diagnostic tools to assess model fit, a predictive model would instead rate its success by its accuracy (or AUC).

### Data resampling
As previously mentioned, predictive models are judged by their ability to predict future events. Obviously when building a model it isn't possible to have direct access to future events without a time machine, and so a means of simulating this is required. The standard approach is to divide the data into two subsets, one for fitting the model, with the other used as an unseen test set, on which the model will predict the response variable for all the observations and have its performance assessed. This process is typically repeated several times to account for variance in the subsetting of the full data set, in a process called _cross-validation_.

Inference on the other hand is concerned with making statements about the population from which the data has been collected, based on the sample available. Therefore, all available data is used in building the model to provide the most accurate representation of the underlying relationships evident in the population. Statisticians typically involved with inference analysis are sometimes guilty of building models on the entire data set and stating that the model has predictive power of x%, despite never having tested this on an unseen test set. 

## Summary

There are numerous other differences, such as the scale of the problems tackled in each domain - predictive modeling can be used on data sets with thousands of variables with modern deep learning dimensionality reduction techniques while inference models aim to reduce the number of insignificant factors in the equation - to the differing focus on the data itself (inference) or the algorithm (predictive). The aim of this post has been to summarise the main contrasts between what one would expect to be almost identical fields and to stress the importance of knowing what the purpose of your analysis is before you start.