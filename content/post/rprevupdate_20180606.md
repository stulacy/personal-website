+++
date = 2018-06-07
draft = false
tags = ["R", "epidemiology", "software"]
title = "rprev 1.0.0 released with lots of new features"
math = false
+++

I'm very happy to announce the first 'official' release of version 1.0.0 of [rprev](https://cran.r-project.org/web/packages/rprev/index.html), the R package for estimating disease prevalence by simulation.
This is useful for epidemiologists who have registry data and want to know disease prevalence from time periods longer than is covered by the registry.
I first released it [almost exactly two years ago](https://stuartlacy.co.uk/2016/06/08/an-r-package-for-estimating-disease-prevalence-by-simulation-rprev/) but had always intended to update it with the features in this release.


This is a major update as it adds a lot of functionality to the package and makes non-backwards compatible changes in the process. 
However, the new features and cleaner parameterisation make it worthwhile!

The previous version wasn't very flexible, as it forced disease incidence to be modelled as a homogeneous Poisson process, while survival used a Weibull distribution.
These models are simple enough to work in many situations, however, the lack of flexibility always frustrated me.

The new version takes a different approach by allowing the survival and incidence models to be passed in by the user, provided they have made available methods for the governing behaviours of these two models: generating an incident population and estimating survival probability of individuals at set times. 
To save users constantly writing trivial models, `rprev` comes with sensible defaults: a homogeneous Poisson process for incidence as before, with the default survival model now a standard parametric model with a choice of distribution.  

This object-oriented implementation allows for a number of existing survival models to be used with relative ease, as they only need correctly parameterised methods to be provided. 
`rprev` has provided interfaces for `survival::survreg` and `flexsurv::flexsurvreg`, opening access to a large variety of survival models.
For example, `flexsurvreg` can be used to create custom models, or use the large range of existing ones, including Royston-Parmar spline models.

The mixture and non-mixture cure models from `flexsurvcure` can be also be used and are very appropriate for long-term survival estimation. 
An additional cure model implementation is provided by `fixed_cure`, which reverts individual survival probabilities back to population mortality rates after a set period of time since diagnosis. 
UK mortality rates are provided with the package, but custom population life tables can be used instead.

I've tried to give a brief overview of the new features without making this post too long, but please see the [user guide vignette](https://cran.r-project.org/web/packages/rprev/vignettes/user_guide.html) for a full set of examples, and don't hesitate to get in touch with any feedback.
