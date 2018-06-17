+++
date = 2018-06-17
draft = false
tags = ["R", "software"]
title = "multistateutils: functions for using multi-state models in R"
math = false
+++

[A month ago](https://stuartlacy.co.uk/2018/04/20/rdes-discrete-event-simulation-in-r-for-estimating-transition-probabilities-from-a-multi-state-model/) I mentioned that I'd been using a discrete event simulation for estimating transition probabilities from parametric multi-state models.
I've now turned this code into a general package containing resources for multi-state modelling, called `multistateutils` (I know, I'm very imaginative) which may be of interest to other people working with multi-state models in R. The current release is [available on CRAN](https://cran.r-project.org/web/packages/multistateutils/index.html), while the development is still [on GitHub](https://github.com/stulacy/multistateutils). 

One new feature I've added is the ability to estimate length of stay in each state.
This is similar to the transition probability function in that it mirrors existing functionality from `flexsurv`, but is more efficient when running estimates for multiple people.

The other new feature is a visualisation of transition probabilities using dynamic predictions, as shown below. Transition probabilities are defined as the probability of being in a given state $j$ at time $t$, given being in state $h$ at time $s$.

$$P_{h,j}(s, t) = \Pr(X(t) = j\ |\ X(s) = h)$$

Standard plots of transition probabilities involve estimating probabilities from a fixed state and time, i.e. keeping $h$ and $s$ fixed, normally at the starting state and $s=0$.
Dynamic predictions instead vary both $h$ and $s$ to produce a more thorough overview of the predicted pathway through the state transition model.
To produce the diagram below, transition probabilties are calculated at 2-year intervals and then combined together to display the flow between the states.

In the actual implementation in `multistateutils::plot_predicted_pathway` the output is an HTML widget, which displays occupancy probabilities on mouse hover, and also lets the states be clicked and dragged around into a more suitable layout.

![](/img/multistateutils_20180607/state_pathway.png)

Upcoming features will include the ability to run full cohort discrete event simulation, which is particularly useful in health modelling. 
It allows researchers to simulate an incident population for a particular disease and view their pathway through states of interest over the disease course, helping, for example, to obtain estimates of costs to the healthcare provider.

Please check out the package [on CRAN](https://cran.r-project.org/web/packages/multistateutils/index.html) and let me know if you have any feedback or suggestions for new features.
