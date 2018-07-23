+++
date = 2018-07-22
draft = false
tags = ["multistateutils", "R", "multi-state modelling"]
title = "New release of multistateutils"
math = false
+++

Released new version of [multistateutils](https://stuartlacy.co.uk/2018/06/17/multistateutils-functions-for-using-multi-state-models-in-r/)

[On CRAN](https://cran.r-project.org/web/packages/multistateutils/index.html)

Two new additions:

## msprep2

replacement for msprep that handles data stored in long format. 
this is more natural form of storing data particularly when taking from DB

show example

Also allows for reversible markov chains
See example below

## Discrete event simulation

Application of simulation engine to estimating statistics for a given population rather than an individual,like the length of stay and transition probabilities functions do.
Currently this function just runs the simulation for a given cohort, you'll need to estimate any statitics from the output, which is state entry times
Useful for health modelling where want to estimate figures such as number of patients who will receive a certain treatment in a given year interval
Can run it 2 ways, either until everyone is dead or with a set time limit

