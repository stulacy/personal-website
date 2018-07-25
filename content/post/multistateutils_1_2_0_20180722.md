+++
date = 2018-07-25
draft = false
tags = ["multistateutils", "R", "multi-state modelling"]
title = "multistateutils v1.2.0 released"
math = false
+++
A new version of
[multistateutils](https://stuartlacy.co.uk/2018/06/17/multistateutils-functions-for-using-multi-state-models-in-r/)
has been released [onto
CRAN](https://cran.r-project.org/web/packages/multistateutils/index.html)
containing a few new features. I’ll give a quick overview of them here,
but have a look at the
[vignette](https://cran.r-project.org/web/packages/multistateutils/vignettes/Examples.html)
for more examples.

msprep2
-------

The first is a replacement for the `mstate::msprep` function that
converts data into the long transition-specific format required for
fitting multi-state models. `msprep` requires the input data to be a in
a wide format, where each row corresponds to an individual and each
possible state has a column for entry time and a status indicator.
Getting raw data into this format can be a bit awkward, not least
because times and statuses still need to be provided for states that
aren’t visited. Furthermore, this wide format prevents the use of
reversible Markov chains, those where the same state can be entered more
than once.

`msprep2` aims to improve the usability by specifying input data in a
long format with each row corresponding to an observed state entry time,
which is a more intuitive data format, utilising the [‘tidy data’
philosophy](http://vita.had.co.nz/papers/tidy-data.html). Covariates and
time of last follow-up are passed in separately and are linked to the
state entry table by an id in the same manner used by relational
databases.

Let’s show a quick example using an illness-death model, which is a
three-state system where a patient starts in the *healthy* state and can
either *die* straight away, or obtain an *illness* first.

    library(mstate)
    tmat <- trans.illdeath()
    tmat

    ##          to
    ## from      healthy illness death
    ##   healthy      NA       1     2
    ##   illness      NA      NA     3
    ##   death        NA      NA    NA

Let’s consider 3 individuals:

1.  Dies at 10 days
2.  Becomes ill at 5 days and dies at 12 days
3.  Becomes ill at 8 days and is lost to follow-up at 20 days

In the wide format required by `msprep` this looks as follows. Following
a patient’s trajectory through the system can be confusing owing to the
fact that entry times are provided even for non-visited states.

    example_wide <- data.frame(id=1:3, illness_time=c(10, 5, 8), illness_status=c(0, 1, 1),
                               death_time = c(10, 12, 20), death_status=c(1, 1, 0))
    example_wide

    ##   id illness_time illness_status death_time death_status
    ## 1  1           10              0         10            1
    ## 2  2            5              1         12            1
    ## 3  3            8              1         20            0

In the long format this would look as follows. I believe it’s much
easier to tell which state each patient has visited in this manner.

    example_long <- data.frame(id=c(1, 2, 2, 3), state=c('death', 'illness', 'death', 'illness'),
                               time=c(10, 5, 12, 8))
    example_long

    ##   id   state time
    ## 1  1   death   10
    ## 2  2 illness    5
    ## 3  2   death   12
    ## 4  3 illness    8

Passing the wide table into `msprep` produces the transition-specific
output required for fitting transition-level models.

    msprep(time=c(NA, 'illness_time', 'death_time'),
                  status=c(NA, 'illness_status', 'death_status'), 
                  data=example_wide, 
                  trans=tmat) 

    ## An object of class 'msdata'
    ## 
    ## Data:
    ##   id from to trans Tstart Tstop time status
    ## 1  1    1  2     1      0    10   10      0
    ## 2  1    1  3     2      0    10   10      1
    ## 3  2    1  2     1      0     5    5      1
    ## 4  2    1  3     2      0     5    5      0
    ## 5  2    2  3     3      5    12    7      1
    ## 6  3    1  2     1      0     8    8      1
    ## 7  3    1  3     2      0     8    8      0
    ## 8  3    2  3     3      8    20   12      0

Running the long table through `msprep2` produces the same output
(albeit in a data tibble rather than an msdata object). Note that
censored observations are specified with an additional data frame that
has an id column and the time of last follow-up, which in this example
is patient 3 who was last observed at 20 days. This again is a more
natural way of storing follow-up data, rather than having to associate
it with a state as in the wide format required by `msprep`.

    library(multistateutils)
    cens <- data.frame(id=3, censor_time=20)
    msprep2(example_long, tmat, censors = cens)

    ## # A tibble: 8 x 8
    ##      id  from    to trans Tstart Tstop  time status
    ##   <int> <int> <int> <int>  <dbl> <dbl> <dbl>  <int>
    ## 1     1     1     2     1      0    10    10      0
    ## 2     1     1     3     2      0    10    10      1
    ## 3     2     1     2     1      0     5     5      1
    ## 4     2     1     3     2      0     5     5      0
    ## 5     2     2     3     3      5    12     7      1
    ## 6     3     1     2     1      0     8     8      1
    ## 7     3     1     3     2      0     8     8      0
    ## 8     3     2     3     3      8    20    12      0

I’ll also quickly show an example of a reversible Markov chain by
extending this illness-death model so that the can *recover*, i.e. move
back into healthy from the illness state.

    tmat2 <- matrix(c(NA, 3, NA, 1, NA, NA, 2, 4, NA), nrow=3, ncol=3, 
                    dimnames=list(colnames(tmat), colnames(tmat)))
    tmat2

    ##         healthy illness death
    ## healthy      NA       1     2
    ## illness       3      NA     4
    ## death        NA      NA    NA

I’ll create a dummy dataset with one individual who is ill at 7 days,
recovers at 12 days, is ill once more at 17 dies and finally dies at 22
days.

    example_reverse <- data.frame(id=c(rep(1, 4)),
                                   state=c('illness', 'healthy', 'illness', 'death'),
                                   time=c(7, 12, 17, 22))
    example_reverse

    ##   id   state time
    ## 1  1 illness    7
    ## 2  1 healthy   12
    ## 3  1 illness   17
    ## 4  1   death   22

`msprep2` has no problem converting this into a table of possible
transitions.

    msprep2(example_reverse, tmat2)

    ## # A tibble: 8 x 8
    ##      id  from    to trans Tstart Tstop  time status
    ##   <int> <int> <int> <int>  <dbl> <dbl> <dbl>  <int>
    ## 1     1     1     2     1      0     7     7      1
    ## 2     1     1     3     2      0     7     7      0
    ## 3     1     2     1     3      7    12     5      1
    ## 4     1     2     3     4      7    12     5      0
    ## 5     1     1     2     1     12    17     5      1
    ## 6     1     1     3     2     12    17     5      0
    ## 7     1     2     1     3     17    22     5      0
    ## 8     1     2     3     4     17    22     5      1

cohort_simulation
-------------------------

The second new feature is similar to the existing `predict_transitions`
and `length_of_stay` functions but rather than running discrete event
simulation to obtain estimates for a single person, it runs
the simulation for a (often heterogeneous) group of individuals at the same time. 
This is useful for applications such as health modelling, where 
we are interested in estimating various statistics about an incident disease population, for example: the number of patients who will
receive a certain treatment, or how long patients spend receiving the treatment.
These values are particularly relevant to estimating costs, facilating the evaluation of treatments from a health economics perspective.

I'll demonstrate this function on the `ebmt3` dataset provided by `mstate`, which is an
illness-death model describing recovery after a bone marrow transplant,
where the illness state is termed *platelet recovery*.

    data(ebmt3)
    head(ebmt3)

    ##   id prtime prstat rfstime rfsstat dissub   age            drmatch    tcd
    ## 1  1     23      1     744       0    CML   >40    Gender mismatch No TCD
    ## 2  2     35      1     360       1    CML   >40 No gender mismatch No TCD
    ## 3  3     26      1     135       1    CML   >40 No gender mismatch No TCD
    ## 4  4     22      1     995       0    AML 20-40 No gender mismatch No TCD
    ## 5  5     29      1     422       1    AML 20-40 No gender mismatch No TCD
    ## 6  6     38      1     119       1    ALL   >40 No gender mismatch No TCD

As this data comes in the wide format required by `msprep`, we can easily convert it
into the transition-specific data frame as described in the
previous section.

    long <- msprep(time=c(NA, 'prtime', 'rfstime'),      # Convert data to long
                   status=c(NA, 'prstat', 'rfsstat'), 
                   data=ebmt3, 
                   trans=tmat, 
                   keep=c('age', 'dissub'))

I’ll build a Weibull model for each transition, using `age` and `dissub` as covariates.

    library(flexsurv)
    models <- lapply(1:3, function(i) {
        flexsurvreg(Surv(time, status) ~ age + dissub, data=long, dist='weibull')
    })

The simulation is implemented in `cohort_simulation` and
requires: a list of `flexsurv` models, a transition matrix, and 
a data frame of new cases to simulate.
It can be run in two ways, either until everyone has entered an absorptive state 
or for a set period of time. 
See the
[vignette](https://cran.r-project.org/web/packages/multistateutils/vignettes/Examples.html)
for full examples of both of these use-cases. 

Below I’ll demonstrate
running the simulation until everyone is dead.
Other options allow for the specification of the starting time and state
for each individual, with the default values of 0 and 1 respectively
being used here. The function output is a long data frame where each row
corresponds to an observed state entry.

    sim <- cohort_simulation(models, ebmt3[, c('age', 'dissub')], tmat)
    head(sim)

    ##   id   age dissub   state time
    ## 1  0   >40    CML healthy    0
    ## 2  2   >40    CML healthy    0
    ## 3  6 20-40    CML healthy    0
    ## 4 14   >40    CML healthy    0
    ## 5 30  <=20    ALL healthy    0
    ## 6 62  <=20    ALL healthy    0


The function doesn't calculate any summary statistics for you, but instead provides the raw state entry times allowing you to estimate any quantity you want.
For example, we can confirm that all 2204 individuals are followed through
to death.

    library(dplyr)
    sim %>% 
        group_by(id) %>%
        top_n(1, time) %>%
        ungroup() %>%
        count(state)

    ## # A tibble: 1 x 2
    ##   state     n
    ##   <chr> <int>
    ## 1 death  2204

And can observe that 51% of individuals are estimated to have platelet recovery.

    sum(sim$state == 'illness') / length(unique(sim$id))

    ## [1] 0.5049909

which is very close to the actual proportion in `ebmt3`, which suggests there is
a strong model fit.

    mean(ebmt3$prstat == 1)

    ## [1] 0.5303993

The code below calculates the
time spent in the platelet recovery state for each disease subtype.

    library(tidyr)
    sim %>%
        spread(state, time) %>%
        filter(!is.na(illness)) %>%
        mutate(pr_time = death - illness) %>%
        group_by(dissub) %>%
        summarise(years_pr_death = median(pr_time) / 365.25)

    ## # A tibble: 3 x 2
    ##   dissub years_pr_death
    ##   <fct>           <dbl>
    ## 1 AML              2.95
    ## 2 ALL              2.15
    ## 3 CML              5.00

I'm very interested to hear feedback on how these functions can be improved, either email me or use the issue tracker on the [GitHub repository](https://github.com/stulacy/multistateutils).
