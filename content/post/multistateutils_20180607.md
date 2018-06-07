+++
date = 2018-06-06
draft = true
tags = ["R", "software"]
title = "multistateutils: functions for using multi-state models in R"
math = false
+++

I've [previously](https://stuartlacy.co.uk/2018/04/20/rdes-discrete-event-simulation-in-r-for-estimating-transition-probabilities-from-a-multi-state-model/) mentioned a package for estimating transition probabilities from parametric multi-state models that I'd put onto GitHub.
I've now added a few additional features to make it into a general package for resources for multi-state modelling and it's [now on CRAN](), while the development is still [on GitHub](https://github.com/stulacy/multistateutils). 

The new features I've added are functions to estimate Length of Stay, which similarly the transition probability estimation is similar to the implementation in `flexsurv` but can obtain multiple estimates from one simulation, making it more efficient in cases where multiple esitmates are required.

The other new feature is a visualisation of transition probabilities using dynamic predictions, as shown below. Standard plots of transition probabilities involving estimating probabilities from a fixed state and time, i.e. starting state at $t=0$. Dynamic predictions vary the starting state and time to produce a more thorough overview of the predicted pathway through the state transition model. In the diagram below, transition probabilties are updated every 2 years.

Further features will include the ability to run full cohort discrete event simulation and summarising values of interest from it, such as for estimating healthcare costs in health economic evaluation. For the time being, check it out [on CRAN](TODO) and let me know if you have any feedback.

![](/img/multistateutils_20180607/state_pathway.png)
